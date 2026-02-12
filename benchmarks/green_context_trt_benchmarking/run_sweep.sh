#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sweep script for Green Context TRT Benchmark
# Runs the benchmark across different model sizes and sample counts,
# collects all output, and prints a comparison summary at the end.
#
# Usage:
#   ./run_sweep.sh                   # Run all configurations via holohub (docker)
#   ./run_sweep.sh --local           # Run binary directly (no docker)
#   ./run_sweep.sh --binary /path/to/green_context_trt_benchmarking
#   ./run_sweep.sh --base-img nvcr.io/nvidia/clara-holoscan/holoscan:v3.6.1-cuda13-dgpu
#   ./run_sweep.sh --dry-run         # Print commands without executing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOLOHUB_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/sweep_results_$(date +%Y%m%d_%H%M%S)"

# ── Defaults ──────────────────────────────────────────────────────────────────
BINARY=""
LOCAL_MODE=false
DRY_RUN=false
BASE_IMG=""
EXTRA_DOCKER_OPTS=""
# Docker container mount point (holohub maps $HOLOHUB_ROOT -> this path inside container)
CONTAINER_HOLOHUB_ROOT="/workspace/holohub"

# Sweep parameters (edit these to customize your sweep)
SAMPLE_COUNTS=(500 1000 2000)
WARMUP_SAMPLES=100
MODE="all"  # baseline | green-context | all

# Model configurations: "label:hidden_size:num_layers:input_size"
# Increase hidden_size / num_layers to create more GPU contention
MODEL_CONFIGS=(
    "small:2048:3:1024"
    "medium:4096:6:1024"
    "large:6144:8:1024"
)

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)        LOCAL_MODE=true; shift ;;
        --binary)       BINARY="$2"; LOCAL_MODE=true; shift 2 ;;
        --base-img)     BASE_IMG="$2"; shift 2 ;;
        --docker-opts)  EXTRA_DOCKER_OPTS="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --samples)
            # Override sample counts: --samples "500 1000"
            IFS=' ' read -ra SAMPLE_COUNTS <<< "$2"; shift 2 ;;
        --models)
            # Override model configs: --models "small:2048:3:1024 large:8192:10:1024"
            IFS=' ' read -ra MODEL_CONFIGS <<< "$2"; shift 2 ;;
        --mode)         MODE="$2"; shift 2 ;;
        --warmup)       WARMUP_SAMPLES="$2"; shift 2 ;;
        -h|--help)
            cat <<'USAGE'
Green Context TRT Benchmark Sweep

Usage: ./run_sweep.sh [OPTIONS]

Options:
  --local                Run binary directly (no docker/holohub wrapper)
  --binary PATH          Path to pre-built binary (implies --local)
  --base-img IMAGE       Container base image for holohub run
  --docker-opts OPTS     Extra docker options
  --samples "N1 N2 ..."  Override sample counts (default: "500 1000 2000")
  --models "CONFIGS"     Override model configs (default: small/medium/large)
                         Format per config: "label:hidden_size:num_layers:input_size"
  --mode MODE            baseline | green-context | all (default: all)
  --warmup N             Warmup samples (default: 100)
  --dry-run              Print commands without executing
  -h, --help             Show this help

Examples:
  # Full sweep on a modern dGPU with docker
  ./run_sweep.sh --base-img nvcr.io/nvidia/clara-holoscan/holoscan:v3.6.1-cuda13-dgpu

  # Quick sweep with fewer configs
  ./run_sweep.sh --samples "1000" --models "medium:4096:6:1024"

  # Run locally with a pre-built binary
  ./run_sweep.sh --binary ./build/benchmarks/green_context_trt_benchmarking/green_context_trt_benchmarking
USAGE
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"

echo "========================================================================"
echo " Green Context TRT Benchmark Sweep"
echo "========================================================================"
echo "  Results directory: $RESULTS_DIR"
echo "  Mode:             $MODE"
echo "  Sample counts:    ${SAMPLE_COUNTS[*]}"
echo "  Warmup samples:   $WARMUP_SAMPLES"
echo "  Model configs:    ${MODEL_CONFIGS[*]}"
echo "  Local mode:       $LOCAL_MODE"
[[ -n "$BASE_IMG" ]] && echo "  Base image:       $BASE_IMG"
echo "========================================================================"
echo ""

# ── Helper: resolve binary path ──────────────────────────────────────────────
find_binary() {
    if [[ -n "$BINARY" ]]; then
        echo "$BINARY"
        return
    fi
    # Common build locations
    local candidates=(
        "$HOLOHUB_ROOT/build/benchmarks/green_context_trt_benchmarking/green_context_trt_benchmarking"
        "$HOLOHUB_ROOT/build/green_context_trt_benchmarking/green_context_trt_benchmarking"
    )
    for c in "${candidates[@]}"; do
        if [[ -x "$c" ]]; then
            echo "$c"
            return
        fi
    done
    echo ""
}

# ── Helper: generate ONNX model ─────────────────────────────────────────────
generate_model() {
    local label="$1" hidden="$2" layers="$3" input_size="$4"
    local model_path="$RESULTS_DIR/model_${label}.onnx"
    local gen_script="$SCRIPT_DIR/generate_onnx_model.py"

    if [[ -f "$model_path" ]]; then
        echo "  [model] Reusing existing $model_path" >&2
        echo "$model_path"
        return
    fi

    echo "  [model] Generating $label model (hidden=$hidden, layers=$layers, input=$input_size)..." >&2

    if [[ ! -f "$gen_script" ]]; then
        echo "ERROR: generate_onnx_model.py not found at $gen_script" >&2
        exit 1
    fi

    if $DRY_RUN; then
        echo "  [dry-run] python3 $gen_script --output $model_path --input-size $input_size --hidden-size $hidden --num-layers $layers" >&2
        echo "$model_path"
        return
    fi

    python3 "$gen_script" \
        --output "$model_path" \
        --input-size "$input_size" \
        --hidden-size "$hidden" \
        --num-layers "$layers" 2>&1 | sed 's/^/    /' >&2

    if [[ ! -f "$model_path" ]]; then
        echo "ERROR: Model generation failed for $label" >&2
        exit 1
    fi

    echo "$model_path"
}

# ── Helper: translate host path to container path ────────────────────────────
host_to_container_path() {
    local host_path="$1"
    # Replace the holohub root prefix with the container mount point
    echo "${host_path/$HOLOHUB_ROOT/$CONTAINER_HOLOHUB_ROOT}"
}

# ── Helper: run one benchmark configuration ──────────────────────────────────
run_benchmark() {
    local label="$1" model_path="$2" input_size="$3" samples="$4"
    local run_id="${label}_samples${samples}"
    local log_file="$RESULTS_DIR/${run_id}.log"

    echo "  [run] $run_id ..."

    if $LOCAL_MODE; then
        local run_args="--samples $samples --warmup-samples $WARMUP_SAMPLES --mode $MODE --input-size $input_size --model-path $model_path"
        local bin
        bin=$(find_binary)
        if $DRY_RUN; then
            echo "  [dry-run] ${bin:-<binary>} $run_args"
            return
        fi
        if [[ -z "$bin" ]]; then
            echo "ERROR: Cannot find benchmark binary. Build first or use --binary." >&2
            exit 1
        fi
        local cmd="$bin $run_args"
        echo "  [cmd] $cmd"
        # Run from binary's directory so it finds the YAML config
        (cd "$(dirname "$bin")" && $cmd) 2>&1 | tee "$log_file"
    else
        # holohub/docker mode: translate host model path to container path
        local container_model_path
        container_model_path=$(host_to_container_path "$model_path")
        local run_args="--samples $samples --warmup-samples $WARMUP_SAMPLES --mode $MODE --input-size $input_size --model-path $container_model_path"

        local docker_opts="--user root"
        [[ -n "$EXTRA_DOCKER_OPTS" ]] && docker_opts="$docker_opts $EXTRA_DOCKER_OPTS"

        local cmd="$HOLOHUB_ROOT/holohub run green_context_trt_benchmarking"
        cmd+=" --docker-opts=\"$docker_opts\""
        [[ -n "$BASE_IMG" ]] && cmd+=" --base-img=$BASE_IMG"
        cmd+=" --run-args=\"$run_args\""

        if $DRY_RUN; then
            echo "  [dry-run] $cmd"
            return
        fi
        echo "  [cmd] $cmd"
        eval "$cmd" 2>&1 | tee "$log_file"
    fi

    echo ""
}

# ── Phase 1: Generate all models ─────────────────────────────────────────────
echo "──── Phase 1: Generating ONNX models ────────────────────────────────────"
declare -A MODEL_PATHS
declare -A MODEL_INPUT_SIZES

for config in "${MODEL_CONFIGS[@]}"; do
    IFS=':' read -r label hidden layers input_size <<< "$config"
    model_path=$(generate_model "$label" "$hidden" "$layers" "$input_size")
    MODEL_PATHS[$label]="$model_path"
    MODEL_INPUT_SIZES[$label]="$input_size"
done
echo ""

# ── Phase 2: Run all benchmark configurations ────────────────────────────────
echo "──── Phase 2: Running benchmarks ────────────────────────────────────────"
TOTAL_RUNS=$(( ${#MODEL_CONFIGS[@]} * ${#SAMPLE_COUNTS[@]} ))
RUN_NUM=0

for config in "${MODEL_CONFIGS[@]}"; do
    IFS=':' read -r label hidden layers input_size <<< "$config"
    model_path="${MODEL_PATHS[$label]}"

    for samples in "${SAMPLE_COUNTS[@]}"; do
        RUN_NUM=$((RUN_NUM + 1))
        echo ""
        echo "──── Run $RUN_NUM / $TOTAL_RUNS: model=$label, samples=$samples ────"
        run_benchmark "$label" "$model_path" "$input_size" "$samples"
    done
done

if $DRY_RUN; then
    echo ""
    echo "[dry-run] No results to summarize."
    exit 0
fi

# ── Phase 3: Parse and summarize results ─────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗"
echo "║                                        SWEEP RESULTS SUMMARY                                               ║"
echo "╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Print header
fmt_hdr="%-8s %7s │ %9s %9s %9s %9s │ %9s %9s %9s %9s │ %8s %8s %8s\n"
fmt_row="%-8s %7d │ %9s %9s %9s %9s │ %9s %9s %9s %9s │ %8s %8s %8s\n"
fmt_miss="%-8s %7d │ %9s %9s %9s %9s │ %9s %9s %9s %9s │ %8s %8s %8s\n"

echo "Launch-Start Latency (μs)"
echo ""
printf "$fmt_hdr" \
    "Model" "Samples" \
    "BL Avg" "BL P50" "BL P95" "BL P99" \
    "GC Avg" "GC P50" "GC P95" "GC P99" \
    "Δ Avg%" "Δ P95%" "Δ P99%"
printf '─%.0s' {1..120}; echo ""

for config in "${MODEL_CONFIGS[@]}"; do
    IFS=':' read -r label hidden layers input_size <<< "$config"

    for samples in "${SAMPLE_COUNTS[@]}"; do
        run_id="${label}_samples${samples}"
        log_file="$RESULTS_DIR/${run_id}.log"

        if [[ ! -f "$log_file" ]]; then
            printf "$fmt_miss" "$label" "$samples" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-"
            continue
        fi

        # Robust awk parser: tracks section (baseline/gc/comparison) and
        # subsection (launch/exec) independently, resetting subsec on section change.
        # The comparison section uses its own cmp_subsec to distinguish
        # "Launch-Start Latency:" from "Kernel Execution Time:".
        eval "$(awk '
        /=== Without Green Context/                       { section="bl"; subsec="" }
        /=== With Green Context ===/                      { section="gc"; subsec="" }
        /Baseline and Green Context Benchmark Comparison/ { section="cmp"; cmp_subsec="" }

        section=="bl" && /CUDA Kernel Launch-Start Time:/ { subsec="launch" }
        section=="bl" && /CUDA Kernel Execution Time:/    { subsec="exec" }
        section=="gc" && /CUDA Kernel Launch-Start Time:/ { subsec="launch" }
        section=="gc" && /CUDA Kernel Execution Time:/    { subsec="exec" }

        section=="cmp" && /Launch-Start Latency:/  { cmp_subsec="launch" }
        section=="cmp" && /Kernel Execution Time:/ { cmp_subsec="exec" }

        # Baseline launch-start stats
        section=="bl" && subsec=="launch" && /Average:/ { gsub(/[^0-9.]/, "", $2); bl_avg=$2 }
        section=="bl" && subsec=="launch" && /P50:/     { gsub(/[^0-9.]/, "", $2); bl_p50=$2 }
        section=="bl" && subsec=="launch" && /P95:/     { gsub(/[^0-9.]/, "", $2); bl_p95=$2 }
        section=="bl" && subsec=="launch" && /P99:/     { gsub(/[^0-9.]/, "", $2); bl_p99=$2 }

        # Green Context launch-start stats
        section=="gc" && subsec=="launch" && /Average:/ { gsub(/[^0-9.]/, "", $2); gc_avg=$2 }
        section=="gc" && subsec=="launch" && /P50:/     { gsub(/[^0-9.]/, "", $2); gc_p50=$2 }
        section=="gc" && subsec=="launch" && /P95:/     { gsub(/[^0-9.]/, "", $2); gc_p95=$2 }
        section=="gc" && subsec=="launch" && /P99:/     { gsub(/[^0-9.]/, "", $2); gc_p99=$2 }

        # Comparison deltas (launch-start only)
        # Extract the value between ( and %) -- portable awk, no gawk needed
        section=="cmp" && cmp_subsec=="launch" && /Average Latency:/ {
            s=$0; sub(/.*\(/, "", s); sub(/%\).*/, "", s); d_avg=s
        }
        section=="cmp" && cmp_subsec=="launch" && /95th Percentile:/ {
            s=$0; sub(/.*\(/, "", s); sub(/%\).*/, "", s); d_p95=s
        }
        section=="cmp" && cmp_subsec=="launch" && /99th Percentile:/ {
            s=$0; sub(/.*\(/, "", s); sub(/%\).*/, "", s); d_p99=s
        }

        END {
            printf "bl_avg=\"%s\"; bl_p50=\"%s\"; bl_p95=\"%s\"; bl_p99=\"%s\"; ", bl_avg, bl_p50, bl_p95, bl_p99
            printf "gc_avg=\"%s\"; gc_p50=\"%s\"; gc_p95=\"%s\"; gc_p99=\"%s\"; ", gc_avg, gc_p50, gc_p95, gc_p99
            printf "d_avg=\"%s\"; d_p95=\"%s\"; d_p99=\"%s\"\n", d_avg, d_p95, d_p99
        }
        ' "$log_file")"

        printf "$fmt_row" \
            "$label" "$samples" \
            "${bl_avg:--}" "${bl_p50:--}" "${bl_p95:--}" "${bl_p99:--}" \
            "${gc_avg:--}" "${gc_p50:--}" "${gc_p95:--}" "${gc_p99:--}" \
            "${d_avg:--}" "${d_p95:--}" "${d_p99:--}"
    done
done

echo ""
echo "TRT Inference Per-Kernel GPU Execution Time (CUPTI, μs)"
echo "  (Positive Δ = GC kernels take longer → GC IS partitioning TRT)"
echo "  (Near-zero Δ = GC may NOT be partitioning TRT)"
echo ""
printf "%-8s %7s │ %9s %9s %9s │ %9s %9s %9s │ %8s %8s %8s │ %8s\n" \
    "Model" "Samples" \
    "BL Avg" "BL P50" "BL P95" \
    "GC Avg" "GC P50" "GC P95" \
    "Δ Avg%" "Δ P50%" "Δ P95%" \
    "Kernels"
printf '─%.0s' {1..115}; echo ""

for config in "${MODEL_CONFIGS[@]}"; do
    IFS=':' read -r label hidden layers input_size <<< "$config"

    for samples in "${SAMPLE_COUNTS[@]}"; do
        run_id="${label}_samples${samples}"
        log_file="$RESULTS_DIR/${run_id}.log"

        if [[ ! -f "$log_file" ]]; then
            printf "%-8s %7d │ %9s %9s %9s │ %9s %9s %9s │ %8s %8s %8s │ %8s\n" \
                "$label" "$samples" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-"
            continue
        fi

        eval "$(awk '
        /TRT Inference Per-Kernel GPU Execution Time/ { section="ikern"; ik_subsec="" }
        section=="ikern" && /Without Green Context/ { ik_subsec="bl" }
        section=="ikern" && /With Green Context/    { ik_subsec="gc" }
        section=="ikern" && /Inference Kernel Execution Time Change/ { ik_subsec="cmp" }

        section=="ikern" && ik_subsec=="bl" && /Average:/ { gsub(/[^0-9.]/, "", $2); ik_bl_avg=$2 }
        section=="ikern" && ik_subsec=="bl" && /P50:/     { gsub(/[^0-9.]/, "", $2); ik_bl_p50=$2 }
        section=="ikern" && ik_subsec=="bl" && /P95:/     { gsub(/[^0-9.]/, "", $2); ik_bl_p95=$2 }
        section=="ikern" && ik_subsec=="bl" && /Kernels:/ { gsub(/[^0-9]/, "", $2); ik_bl_cnt=$2 }

        section=="ikern" && ik_subsec=="gc" && /Average:/ { gsub(/[^0-9.]/, "", $2); ik_gc_avg=$2 }
        section=="ikern" && ik_subsec=="gc" && /P50:/     { gsub(/[^0-9.]/, "", $2); ik_gc_p50=$2 }
        section=="ikern" && ik_subsec=="gc" && /P95:/     { gsub(/[^0-9.]/, "", $2); ik_gc_p95=$2 }

        section=="ikern" && ik_subsec=="cmp" && /Average:/ {
            s=$0; sub(/.*: /, "", s); sub(/%.*/, "", s); ik_d_avg=s
        }
        section=="ikern" && ik_subsec=="cmp" && /P50:/ {
            s=$0; sub(/.*: /, "", s); sub(/%.*/, "", s); ik_d_p50=s
        }
        section=="ikern" && ik_subsec=="cmp" && /P95:/ {
            s=$0; sub(/.*: /, "", s); sub(/%.*/, "", s); ik_d_p95=s
        }

        END {
            printf "ik_bl_avg=\"%s\"; ik_bl_p50=\"%s\"; ik_bl_p95=\"%s\"; ", ik_bl_avg, ik_bl_p50, ik_bl_p95
            printf "ik_gc_avg=\"%s\"; ik_gc_p50=\"%s\"; ik_gc_p95=\"%s\"; ", ik_gc_avg, ik_gc_p50, ik_gc_p95
            printf "ik_d_avg=\"%s\"; ik_d_p50=\"%s\"; ik_d_p95=\"%s\"; ", ik_d_avg, ik_d_p50, ik_d_p95
            printf "ik_bl_cnt=\"%s\"\n", ik_bl_cnt
        }
        ' "$log_file")"

        printf "%-8s %7d │ %9s %9s %9s │ %9s %9s %9s │ %8s %8s %8s │ %8s\n" \
            "$label" "$samples" \
            "${ik_bl_avg:--}" "${ik_bl_p50:--}" "${ik_bl_p95:--}" \
            "${ik_gc_avg:--}" "${ik_gc_p50:--}" "${ik_gc_p95:--}" \
            "${ik_d_avg:--}" "${ik_d_p50:--}" "${ik_d_p95:--}" \
            "${ik_bl_cnt:--}"
    done
done

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo " Full logs: $RESULTS_DIR/"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
ls -lh "$RESULTS_DIR"/*.log 2>/dev/null || echo "  (no log files found)"
