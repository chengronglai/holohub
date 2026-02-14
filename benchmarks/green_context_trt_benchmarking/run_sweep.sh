#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sweep script for Green Context Inference Benchmark
# Runs the benchmark across backends (trt/onnxrt), model sizes, and sample counts.
# Collects all output and prints multi-dimensional comparison summary tables.
#
# Usage:
#   ./run_sweep.sh                   # Run all configurations via holohub (docker)
#   ./run_sweep.sh --local           # Run binary directly (no docker)
#   ./run_sweep.sh --binary /path/to/green_context_trt_benchmarking
#   ./run_sweep.sh --base-img nvcr.io/nvidia/clara-holoscan/holoscan:v3.10.0-cuda13
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
CONTAINER_HOLOHUB_ROOT="/workspace/holohub"

# ── Sweep parameters (edit these to customize your sweep) ─────────────────────
BACKENDS=(trt onnxrt)
SAMPLE_COUNTS=(5000)
REPEATS=5           # Number of times to repeat each configuration
WARMUP_SAMPLES=500
SMS_PER_PARTITION=0 # 0 = auto (half GPU). Set to e.g. 8 or 16 to stress-test SM partitioning.
MODE="all"  # baseline | green-context | all

# Model configurations: "label:hidden_size:num_layers:input_size"
# Increase hidden_size / num_layers to create more GPU contention.
# The label is used in file names and summary tables.
# Note: "small" (2048:3) omitted -- too little contention for meaningful results.
MODEL_CONFIGS=(
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
        --backends)
            IFS=' ' read -ra BACKENDS <<< "$2"; shift 2 ;;
        --repeats)      REPEATS="$2"; shift 2 ;;
        --sms)          SMS_PER_PARTITION="$2"; shift 2 ;;
        --samples)
            IFS=' ' read -ra SAMPLE_COUNTS <<< "$2"; shift 2 ;;
        --models)
            IFS=' ' read -ra MODEL_CONFIGS <<< "$2"; shift 2 ;;
        --mode)         MODE="$2"; shift 2 ;;
        --warmup)       WARMUP_SAMPLES="$2"; shift 2 ;;
        -h|--help)
            cat <<'USAGE'
Green Context Inference Benchmark Sweep

Usage: ./run_sweep.sh [OPTIONS]

Options:
  --local                Run binary directly (no docker/holohub wrapper)
  --binary PATH          Path to pre-built binary (implies --local)
  --base-img IMAGE       Container base image for holohub run
  --docker-opts OPTS     Extra docker options
  --backends "B1 B2"     Backends to sweep (default: "trt onnxrt")
  --repeats N            Repeat each configuration N times (default: 5)
  --sms N                SMs per GC partition (default: 0 = auto/half GPU)
                         Use 8 or 16 to stress-test SM partitioning
  --samples "N1 N2 ..."  Override sample counts (default: "5000")
  --models "CONFIGS"     Override model configs (default: medium/large)
                         Format per config: "label:hidden_size:num_layers:input_size"
  --mode MODE            baseline | green-context | all (default: all)
  --warmup N             Warmup samples (default: 500)
  --dry-run              Print commands without executing
  -h, --help             Show this help

Examples:
  # Full sweep with docker
  ./run_sweep.sh --base-img nvcr.io/nvidia/clara-holoscan/holoscan:v3.10.0-cuda13

  # Quick test: one backend, one model
  ./run_sweep.sh --backends "onnxrt" --models "medium:4096:6:1024" --samples "500"

  # Run locally with a pre-built binary
  ./run_sweep.sh --binary ./build/benchmarks/green_context_trt_benchmarking/green_context_trt_benchmarking

  # Compare backends with medium model
  ./run_sweep.sh --models "medium:4096:6:1024" --backends "trt onnxrt"
USAGE
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"

TOTAL_RUNS=$(( ${#BACKENDS[@]} * ${#MODEL_CONFIGS[@]} * ${#SAMPLE_COUNTS[@]} * REPEATS ))

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║            Green Context Inference Benchmark Sweep                    ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Results directory: $RESULTS_DIR"
echo "  Mode:             $MODE"
echo "  Backends:         ${BACKENDS[*]}"
echo "  Sample counts:    ${SAMPLE_COUNTS[*]}"
echo "  Repeats:          $REPEATS"
echo "  SMs/partition:    ${SMS_PER_PARTITION:-0 (auto)}"
echo "  Warmup samples:   $WARMUP_SAMPLES"
echo "  Model configs:    ${MODEL_CONFIGS[*]}"
echo "  Total runs:       $TOTAL_RUNS"
echo "  Local mode:       $LOCAL_MODE"
[[ -n "$BASE_IMG" ]] && echo "  Base image:       $BASE_IMG"
echo ""

# ── Helper: resolve binary path ──────────────────────────────────────────────
find_binary() {
    if [[ -n "$BINARY" ]]; then
        echo "$BINARY"
        return
    fi
    local candidates=(
        "$HOLOHUB_ROOT/build/benchmarks/green_context_trt_benchmarking/green_context_trt_benchmarking"
        "$HOLOHUB_ROOT/build/green_context_trt_benchmarking/green_context_trt_benchmarking"
        "$HOLOHUB_ROOT/build/green_context_trt_benchmarking/benchmarks/green_context_trt_benchmarking/green_context_trt_benchmarking"
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
    echo "${host_path/$HOLOHUB_ROOT/$CONTAINER_HOLOHUB_ROOT}"
}

# ── Helper: run one benchmark configuration ──────────────────────────────────
run_benchmark() {
    local backend="$1" label="$2" model_path="$3" input_size="$4" samples="$5" repeat="$6"
    local run_id="${backend}_${label}_s${samples}_r${repeat}"
    local log_file="$RESULTS_DIR/${run_id}.log"

    echo "  [run] $run_id ..."

    if $LOCAL_MODE; then
        local run_args="--samples $samples --warmup-samples $WARMUP_SAMPLES --mode $MODE --input-size $input_size --model-path $model_path --backend $backend --sms-per-partition $SMS_PER_PARTITION"
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
        (cd "$(dirname "$bin")" && $cmd) 2>&1 | tee "$log_file"
    else
        local container_model_path
        container_model_path=$(host_to_container_path "$model_path")
        local run_args="--samples $samples --warmup-samples $WARMUP_SAMPLES --mode $MODE --input-size $input_size --model-path $container_model_path --backend $backend --sms-per-partition $SMS_PER_PARTITION"

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

# ── Helper: parse all metrics from one log file ──────────────────────────────
# Uses hierarchical section tracking:
#   major = which titled block we are in (timing / cmp / infer / ikern / ...)
#   side  = "bl" or "gc" (reset on each "Without/With Green Context" line)
#   metric = "launch" or "exec" (within the timing major section)
parse_log() {
    local log_file="$1"
    eval "$(awk '
    # ── Major section headers (mutually exclusive, appear once each) ──
    /Comprehensive Timing Results/                    { major="timing"; side=""; metric="" }
    /Baseline and Green Context Benchmark Comparison/ { major="cmp"; cmp_sub="" }
    /TRT InferenceOp Compute Time.*Wall-Clock/        { major="infer"; side="" }
    /InferenceOp Compute Time Change/                 { major="infer_cmp" }
    /TRT Inference Per-Kernel GPU Execution Time/     { major="ikern"; side="" }
    /Inference Kernel Execution Time Change/          { major="ikern_cmp" }

    # ── BL / GC sub-sections (apply within any major section) ──
    /Without Green Context/ { side="bl" }
    /With Green Context/    { side="gc" }

    # ── Within "Comprehensive Timing Results" ──
    major=="timing" && /CUDA Kernel Launch-Start Time:/ { metric="launch" }
    major=="timing" && /CUDA Kernel Execution Time:/    { metric="exec" }

    major=="timing" && side=="bl" && metric=="launch" && /Average:/ { gsub(/[^0-9.]/, "", $2); bl_launch_avg=$2 }
    major=="timing" && side=="bl" && metric=="launch" && /P50:/     { gsub(/[^0-9.]/, "", $2); bl_launch_p50=$2 }
    major=="timing" && side=="bl" && metric=="launch" && /P95:/     { gsub(/[^0-9.]/, "", $2); bl_launch_p95=$2 }
    major=="timing" && side=="bl" && metric=="launch" && /P99:/     { gsub(/[^0-9.]/, "", $2); bl_launch_p99=$2 }
    major=="timing" && side=="bl" && metric=="launch" && /Max:/     { gsub(/[^0-9.]/, "", $2); bl_launch_max=$2 }

    major=="timing" && side=="gc" && metric=="launch" && /Average:/ { gsub(/[^0-9.]/, "", $2); gc_launch_avg=$2 }
    major=="timing" && side=="gc" && metric=="launch" && /P50:/     { gsub(/[^0-9.]/, "", $2); gc_launch_p50=$2 }
    major=="timing" && side=="gc" && metric=="launch" && /P95:/     { gsub(/[^0-9.]/, "", $2); gc_launch_p95=$2 }
    major=="timing" && side=="gc" && metric=="launch" && /P99:/     { gsub(/[^0-9.]/, "", $2); gc_launch_p99=$2 }
    major=="timing" && side=="gc" && metric=="launch" && /Max:/     { gsub(/[^0-9.]/, "", $2); gc_launch_max=$2 }

    major=="timing" && side=="bl" && metric=="exec" && /Average:/ { gsub(/[^0-9.]/, "", $2); bl_exec_avg=$2 }
    major=="timing" && side=="bl" && metric=="exec" && /P50:/     { gsub(/[^0-9.]/, "", $2); bl_exec_p50=$2 }
    major=="timing" && side=="bl" && metric=="exec" && /P95:/     { gsub(/[^0-9.]/, "", $2); bl_exec_p95=$2 }

    major=="timing" && side=="gc" && metric=="exec" && /Average:/ { gsub(/[^0-9.]/, "", $2); gc_exec_avg=$2 }
    major=="timing" && side=="gc" && metric=="exec" && /P50:/     { gsub(/[^0-9.]/, "", $2); gc_exec_p50=$2 }
    major=="timing" && side=="gc" && metric=="exec" && /P95:/     { gsub(/[^0-9.]/, "", $2); gc_exec_p95=$2 }

    # ── Within "Benchmark Comparison" ──
    major=="cmp" && /Launch-Start Latency:/  { cmp_sub="launch" }
    major=="cmp" && /Kernel Execution Time:/ { cmp_sub="exec" }

    major=="cmp" && cmp_sub=="launch" && /Average Latency:/ {
        s=$0; sub(/.*\(/, "", s); sub(/%\).*/, "", s); d_launch_avg=s
    }
    major=="cmp" && cmp_sub=="launch" && /95th Percentile:/ {
        s=$0; sub(/.*\(/, "", s); sub(/%\).*/, "", s); d_launch_p95=s
    }
    major=="cmp" && cmp_sub=="launch" && /99th Percentile:/ {
        s=$0; sub(/.*\(/, "", s); sub(/%\).*/, "", s); d_launch_p99=s
    }
    major=="cmp" && cmp_sub=="exec" && /Average Duration:/ {
        s=$0; sub(/.*\(/, "", s); sub(/%\).*/, "", s); d_exec_avg=s
    }

    # ── Within "InferenceOp Compute Time (Wall-Clock)" ──
    major=="infer" && side=="bl" && /Average:/ { gsub(/[^0-9.]/, "", $2); bl_infer_avg=$2 }
    major=="infer" && side=="bl" && /P50:/     { gsub(/[^0-9.]/, "", $2); bl_infer_p50=$2 }
    major=="infer" && side=="bl" && /P95:/     { gsub(/[^0-9.]/, "", $2); bl_infer_p95=$2 }
    major=="infer" && side=="bl" && /Samples:/ { gsub(/[^0-9]/, "", $2);  bl_infer_cnt=$2 }

    major=="infer" && side=="gc" && /Average:/ { gsub(/[^0-9.]/, "", $2); gc_infer_avg=$2 }
    major=="infer" && side=="gc" && /P50:/     { gsub(/[^0-9.]/, "", $2); gc_infer_p50=$2 }
    major=="infer" && side=="gc" && /P95:/     { gsub(/[^0-9.]/, "", $2); gc_infer_p95=$2 }
    major=="infer" && side=="gc" && /Samples:/ { gsub(/[^0-9]/, "", $2);  gc_infer_cnt=$2 }

    major=="infer_cmp" && /Average:/ {
        s=$0; sub(/.*: /, "", s); sub(/%.*/, "", s); d_infer_avg=s
    }

    # ── Within "Inference Per-Kernel GPU Execution Time (CUPTI)" ──
    major=="ikern" && side=="bl" && /Average:/ { gsub(/[^0-9.]/, "", $2); ik_bl_avg=$2 }
    major=="ikern" && side=="bl" && /P50:/     { gsub(/[^0-9.]/, "", $2); ik_bl_p50=$2 }
    major=="ikern" && side=="bl" && /P95:/     { gsub(/[^0-9.]/, "", $2); ik_bl_p95=$2 }
    major=="ikern" && side=="bl" && /Kernels:/ { gsub(/[^0-9]/, "", $2);  ik_bl_cnt=$2 }

    major=="ikern" && side=="gc" && /Average:/ { gsub(/[^0-9.]/, "", $2); ik_gc_avg=$2 }
    major=="ikern" && side=="gc" && /P50:/     { gsub(/[^0-9.]/, "", $2); ik_gc_p50=$2 }
    major=="ikern" && side=="gc" && /P95:/     { gsub(/[^0-9.]/, "", $2); ik_gc_p95=$2 }
    major=="ikern" && side=="gc" && /Kernels:/ { gsub(/[^0-9]/, "", $2);  ik_gc_cnt=$2 }

    major=="ikern_cmp" && /Average:/ {
        s=$0; sub(/.*: /, "", s); sub(/%.*/, "", s); ik_d_avg=s
    }
    major=="ikern_cmp" && /P50:/ {
        s=$0; sub(/.*: /, "", s); sub(/%.*/, "", s); ik_d_p50=s
    }
    major=="ikern_cmp" && /P95:/ {
        s=$0; sub(/.*: /, "", s); sub(/%.*/, "", s); ik_d_p95=s
    }

    END {
        # Launch-start
        printf "bl_launch_avg=\"%s\"; bl_launch_p50=\"%s\"; bl_launch_p95=\"%s\"; bl_launch_p99=\"%s\"; bl_launch_max=\"%s\"; ", bl_launch_avg, bl_launch_p50, bl_launch_p95, bl_launch_p99, bl_launch_max
        printf "gc_launch_avg=\"%s\"; gc_launch_p50=\"%s\"; gc_launch_p95=\"%s\"; gc_launch_p99=\"%s\"; gc_launch_max=\"%s\"; ", gc_launch_avg, gc_launch_p50, gc_launch_p95, gc_launch_p99, gc_launch_max
        printf "d_launch_avg=\"%s\"; d_launch_p95=\"%s\"; d_launch_p99=\"%s\"; ", d_launch_avg, d_launch_p95, d_launch_p99
        # Kernel execution
        printf "bl_exec_avg=\"%s\"; bl_exec_p50=\"%s\"; bl_exec_p95=\"%s\"; ", bl_exec_avg, bl_exec_p50, bl_exec_p95
        printf "gc_exec_avg=\"%s\"; gc_exec_p50=\"%s\"; gc_exec_p95=\"%s\"; ", gc_exec_avg, gc_exec_p50, gc_exec_p95
        printf "d_exec_avg=\"%s\"; ", d_exec_avg
        # InferenceOp compute
        printf "bl_infer_avg=\"%s\"; bl_infer_p50=\"%s\"; bl_infer_p95=\"%s\"; bl_infer_cnt=\"%s\"; ", bl_infer_avg, bl_infer_p50, bl_infer_p95, bl_infer_cnt
        printf "gc_infer_avg=\"%s\"; gc_infer_p50=\"%s\"; gc_infer_p95=\"%s\"; gc_infer_cnt=\"%s\"; ", gc_infer_avg, gc_infer_p50, gc_infer_p95, gc_infer_cnt
        printf "d_infer_avg=\"%s\"; ", d_infer_avg
        # Per-kernel inference
        printf "ik_bl_avg=\"%s\"; ik_bl_p50=\"%s\"; ik_bl_p95=\"%s\"; ik_bl_cnt=\"%s\"; ", ik_bl_avg, ik_bl_p50, ik_bl_p95, ik_bl_cnt
        printf "ik_gc_avg=\"%s\"; ik_gc_p50=\"%s\"; ik_gc_p95=\"%s\"; ik_gc_cnt=\"%s\"; ", ik_gc_avg, ik_gc_p50, ik_gc_p95, ik_gc_cnt
        printf "ik_d_avg=\"%s\"; ik_d_p50=\"%s\"; ik_d_p95=\"%s\"\n", ik_d_avg, ik_d_p50, ik_d_p95
    }
    ' "$log_file")"
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
RUN_NUM=0

for backend in "${BACKENDS[@]}"; do
    for config in "${MODEL_CONFIGS[@]}"; do
        IFS=':' read -r label hidden layers input_size <<< "$config"
        model_path="${MODEL_PATHS[$label]}"

        for samples in "${SAMPLE_COUNTS[@]}"; do
            for repeat in $(seq 1 "$REPEATS"); do
                RUN_NUM=$((RUN_NUM + 1))
                echo ""
                echo "──── Run $RUN_NUM / $TOTAL_RUNS: backend=$backend, model=$label, samples=$samples, repeat=$repeat/$REPEATS ────"
                run_benchmark "$backend" "$label" "$model_path" "$input_size" "$samples" "$repeat"
            done
        done
    done
done

if $DRY_RUN; then
    echo ""
    echo "[dry-run] No results to summarize."
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Parse and summarize results
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗"
echo "║                                              SWEEP RESULTS SUMMARY                                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝"
echo ""

# ── Helper: print table header box ────────────────────────────────────────────
render_table() {
    local title="$1" desc="$2"

    echo ""
    echo "┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐"
    echo "│  $title"
    echo "│  $desc"
    echo "└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘"
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# Table 1: Launch-Start Latency (primary metric)
# ─────────────────────────────────────────────────────────────────────────────
render_table \
    "TABLE 1: CUDA Kernel Launch-Start Latency (μs) — PRIMARY METRIC" \
    "Lower = better. Positive Δ% = GC improved scheduling."

printf "%-7s %-8s %3s │ %8s %8s %8s %8s │ %8s %8s %8s %8s │ %8s %8s %8s\n" \
    "Backend" "Model" "Run" \
    "BL Avg" "BL P50" "BL P95" "BL P99" \
    "GC Avg" "GC P50" "GC P95" "GC P99" \
    "Δ Avg%" "Δ P95%" "Δ P99%"
printf '─%.0s' {1..130}; echo ""

for backend in "${BACKENDS[@]}"; do
    for config in "${MODEL_CONFIGS[@]}"; do
        IFS=':' read -r label hidden layers input_size <<< "$config"
        for samples in "${SAMPLE_COUNTS[@]}"; do
            # Accumulators for computing mean across repeats
            sum_bl_avg=0; sum_gc_avg=0; sum_d_avg=0; n_valid=0

            for repeat in $(seq 1 "$REPEATS"); do
                run_id="${backend}_${label}_s${samples}_r${repeat}"
                log_file="$RESULTS_DIR/${run_id}.log"
                if [[ ! -f "$log_file" ]]; then
                    printf "%-7s %-8s  r%d │ %8s %8s %8s %8s │ %8s %8s %8s %8s │ %8s %8s %8s\n" \
                        "$backend" "$label" "$repeat" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-"
                    continue
                fi
                parse_log "$log_file"
                printf "%-7s %-8s  r%d │ %8s %8s %8s %8s │ %8s %8s %8s %8s │ %8s %8s %8s\n" \
                    "$backend" "$label" "$repeat" \
                    "${bl_launch_avg:--}" "${bl_launch_p50:--}" "${bl_launch_p95:--}" "${bl_launch_p99:--}" \
                    "${gc_launch_avg:--}" "${gc_launch_p50:--}" "${gc_launch_p95:--}" "${gc_launch_p99:--}" \
                    "${d_launch_avg:--}" "${d_launch_p95:--}" "${d_launch_p99:--}"

                # Accumulate for mean (only if values are numeric)
                if [[ -n "${bl_launch_avg}" && -n "${gc_launch_avg}" && -n "${d_launch_avg}" ]]; then
                    sum_bl_avg=$(awk "BEGIN{printf \"%.2f\", $sum_bl_avg + ${bl_launch_avg}}")
                    sum_gc_avg=$(awk "BEGIN{printf \"%.2f\", $sum_gc_avg + ${gc_launch_avg}}")
                    sum_d_avg=$(awk "BEGIN{printf \"%.2f\", $sum_d_avg + ${d_launch_avg/+/}}")
                    n_valid=$((n_valid + 1))
                fi
            done

            # Print MEAN row
            if [[ $n_valid -gt 0 ]]; then
                mean_bl=$(awk "BEGIN{printf \"%.2f\", $sum_bl_avg / $n_valid}")
                mean_gc=$(awk "BEGIN{printf \"%.2f\", $sum_gc_avg / $n_valid}")
                mean_d=$(awk "BEGIN{printf \"%.2f\", $sum_d_avg / $n_valid}")
                printf "\e[1m%-7s %-8s MEAN│ %8s %8s %8s %8s │ %8s %8s %8s %8s │ %8s %8s %8s\e[0m\n" \
                    "$backend" "$label" \
                    "$mean_bl" "" "" "" \
                    "$mean_gc" "" "" "" \
                    "$mean_d" "" ""
            fi
            printf '·%.0s' {1..130}; echo ""
        done
    done
    printf '─%.0s' {1..130}; echo ""
done

# ─────────────────────────────────────────────────────────────────────────────
# Table 2: Benchmark Kernel Execution Time
# ─────────────────────────────────────────────────────────────────────────────
render_table \
    "TABLE 2: Benchmark Kernel Execution Time (μs)" \
    "Measures how long the timing kernel itself runs on GPU. Lower with GC = less SM contention."

printf "%-7s %-8s %3s │ %8s %8s %8s │ %8s %8s %8s │ %8s\n" \
    "Backend" "Model" "Run" \
    "BL Avg" "BL P50" "BL P95" \
    "GC Avg" "GC P50" "GC P95" \
    "Δ Avg%"
printf '─%.0s' {1..90}; echo ""

for backend in "${BACKENDS[@]}"; do
    for config in "${MODEL_CONFIGS[@]}"; do
        IFS=':' read -r label hidden layers input_size <<< "$config"
        for samples in "${SAMPLE_COUNTS[@]}"; do
            sum_bl=0; sum_gc=0; sum_d=0; n_valid=0
            for repeat in $(seq 1 "$REPEATS"); do
                run_id="${backend}_${label}_s${samples}_r${repeat}"
                log_file="$RESULTS_DIR/${run_id}.log"
                if [[ ! -f "$log_file" ]]; then
                    printf "%-7s %-8s  r%d │ %8s %8s %8s │ %8s %8s %8s │ %8s\n" \
                        "$backend" "$label" "$repeat" "-" "-" "-" "-" "-" "-" "-"
                    continue
                fi
                parse_log "$log_file"
                printf "%-7s %-8s  r%d │ %8s %8s %8s │ %8s %8s %8s │ %8s\n" \
                    "$backend" "$label" "$repeat" \
                    "${bl_exec_avg:--}" "${bl_exec_p50:--}" "${bl_exec_p95:--}" \
                    "${gc_exec_avg:--}" "${gc_exec_p50:--}" "${gc_exec_p95:--}" \
                    "${d_exec_avg:--}"
                if [[ -n "${bl_exec_avg}" && -n "${gc_exec_avg}" && -n "${d_exec_avg}" ]]; then
                    sum_bl=$(awk "BEGIN{printf \"%.2f\", $sum_bl + ${bl_exec_avg}}")
                    sum_gc=$(awk "BEGIN{printf \"%.2f\", $sum_gc + ${gc_exec_avg}}")
                    sum_d=$(awk "BEGIN{printf \"%.2f\", $sum_d + ${d_exec_avg/+/}}")
                    n_valid=$((n_valid + 1))
                fi
            done
            if [[ $n_valid -gt 0 ]]; then
                printf "\e[1m%-7s %-8s MEAN│ %8s %8s %8s │ %8s %8s %8s │ %8s\e[0m\n" \
                    "$backend" "$label" \
                    "$(awk "BEGIN{printf \"%.2f\", $sum_bl/$n_valid}")" "" "" \
                    "$(awk "BEGIN{printf \"%.2f\", $sum_gc/$n_valid}")" "" "" \
                    "$(awk "BEGIN{printf \"%.2f\", $sum_d/$n_valid}")"
            fi
            printf '·%.0s' {1..90}; echo ""
        done
    done
    printf '─%.0s' {1..90}; echo ""
done

# ─────────────────────────────────────────────────────────────────────────────
# Table 3: InferenceOp Compute Time (wall-clock)
# ─────────────────────────────────────────────────────────────────────────────
render_table \
    "TABLE 3: InferenceOp Compute Time — Wall-Clock (μs)" \
    "Wall-clock time of InferenceOp::compute(). Positive Δ = GC slows inference (expected with fewer SMs)."

printf "%-7s %-8s %3s │ %8s %8s %8s %6s │ %8s %8s %8s %6s │ %8s\n" \
    "Backend" "Model" "Run" \
    "BL Avg" "BL P50" "BL P95" "BL N" \
    "GC Avg" "GC P50" "GC P95" "GC N" \
    "Δ Avg%"
printf '─%.0s' {1..105}; echo ""

for backend in "${BACKENDS[@]}"; do
    for config in "${MODEL_CONFIGS[@]}"; do
        IFS=':' read -r label hidden layers input_size <<< "$config"
        for samples in "${SAMPLE_COUNTS[@]}"; do
            sum_bl=0; sum_gc=0; sum_d=0; n_valid=0
            for repeat in $(seq 1 "$REPEATS"); do
                run_id="${backend}_${label}_s${samples}_r${repeat}"
                log_file="$RESULTS_DIR/${run_id}.log"
                if [[ ! -f "$log_file" ]]; then
                    printf "%-7s %-8s  r%d │ %8s %8s %8s %6s │ %8s %8s %8s %6s │ %8s\n" \
                        "$backend" "$label" "$repeat" "-" "-" "-" "-" "-" "-" "-" "-" "-"
                    continue
                fi
                parse_log "$log_file"
                printf "%-7s %-8s  r%d │ %8s %8s %8s %6s │ %8s %8s %8s %6s │ %8s\n" \
                    "$backend" "$label" "$repeat" \
                    "${bl_infer_avg:--}" "${bl_infer_p50:--}" "${bl_infer_p95:--}" "${bl_infer_cnt:--}" \
                    "${gc_infer_avg:--}" "${gc_infer_p50:--}" "${gc_infer_p95:--}" "${gc_infer_cnt:--}" \
                    "${d_infer_avg:--}"
                if [[ -n "${bl_infer_avg}" && -n "${gc_infer_avg}" && -n "${d_infer_avg}" ]]; then
                    sum_bl=$(awk "BEGIN{printf \"%.2f\", $sum_bl + ${bl_infer_avg}}")
                    sum_gc=$(awk "BEGIN{printf \"%.2f\", $sum_gc + ${gc_infer_avg}}")
                    sum_d=$(awk "BEGIN{printf \"%.2f\", $sum_d + ${d_infer_avg/+/}}")
                    n_valid=$((n_valid + 1))
                fi
            done
            if [[ $n_valid -gt 0 ]]; then
                printf "\e[1m%-7s %-8s MEAN│ %8s %8s %8s %6s │ %8s %8s %8s %6s │ %8s\e[0m\n" \
                    "$backend" "$label" \
                    "$(awk "BEGIN{printf \"%.2f\", $sum_bl/$n_valid}")" "" "" "" \
                    "$(awk "BEGIN{printf \"%.2f\", $sum_gc/$n_valid}")" "" "" "" \
                    "$(awk "BEGIN{printf \"%.2f\", $sum_d/$n_valid}")"
            fi
            printf '·%.0s' {1..105}; echo ""
        done
    done
    printf '─%.0s' {1..105}; echo ""
done

# ─────────────────────────────────────────────────────────────────────────────
# Table 4: Inference Per-Kernel GPU Execution Time (CUPTI)
# ─────────────────────────────────────────────────────────────────────────────
render_table \
    "TABLE 4: Inference Per-Kernel GPU Execution Time — CUPTI (μs)" \
    "Near-zero Δ = inference kernels NOT partitioned (bypass GC). Positive Δ = GC IS partitioning."

printf "%-7s %-8s %3s │ %8s %8s %8s %7s │ %8s %8s %8s %7s │ %7s %7s %7s\n" \
    "Backend" "Model" "Run" \
    "BL Avg" "BL P50" "BL P95" "BL #K" \
    "GC Avg" "GC P50" "GC P95" "GC #K" \
    "Δ Avg%" "Δ P50%" "Δ P95%"
printf '─%.0s' {1..120}; echo ""

for backend in "${BACKENDS[@]}"; do
    for config in "${MODEL_CONFIGS[@]}"; do
        IFS=':' read -r label hidden layers input_size <<< "$config"
        for samples in "${SAMPLE_COUNTS[@]}"; do
            sum_d=0; n_valid=0
            for repeat in $(seq 1 "$REPEATS"); do
                run_id="${backend}_${label}_s${samples}_r${repeat}"
                log_file="$RESULTS_DIR/${run_id}.log"
                if [[ ! -f "$log_file" ]]; then
                    printf "%-7s %-8s  r%d │ %8s %8s %8s %7s │ %8s %8s %8s %7s │ %7s %7s %7s\n" \
                        "$backend" "$label" "$repeat" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-"
                    continue
                fi
                parse_log "$log_file"
                printf "%-7s %-8s  r%d │ %8s %8s %8s %7s │ %8s %8s %8s %7s │ %7s %7s %7s\n" \
                    "$backend" "$label" "$repeat" \
                    "${ik_bl_avg:--}" "${ik_bl_p50:--}" "${ik_bl_p95:--}" "${ik_bl_cnt:--}" \
                    "${ik_gc_avg:--}" "${ik_gc_p50:--}" "${ik_gc_p95:--}" "${ik_gc_cnt:--}" \
                    "${ik_d_avg:--}" "${ik_d_p50:--}" "${ik_d_p95:--}"
                if [[ -n "${ik_d_avg}" ]]; then
                    sum_d=$(awk "BEGIN{printf \"%.2f\", $sum_d + ${ik_d_avg/+/}}")
                    n_valid=$((n_valid + 1))
                fi
            done
            if [[ $n_valid -gt 0 ]]; then
                printf "\e[1m%-7s %-8s MEAN│ %8s %8s %8s %7s │ %8s %8s %8s %7s │ %7s %7s %7s\e[0m\n" \
                    "$backend" "$label" \
                    "" "" "" "" \
                    "" "" "" "" \
                    "$(awk "BEGIN{printf \"%.2f\", $sum_d/$n_valid}")" "" ""
            fi
            printf '·%.0s' {1..120}; echo ""
        done
    done
    printf '─%.0s' {1..120}; echo ""
done

# ─────────────────────────────────────────────────────────────────────────────
# Quick comparison: MEAN launch-start across repeats, one row per backend x model
# ─────────────────────────────────────────────────────────────────────────────
if [[ ${#BACKENDS[@]} -ge 2 && ${#SAMPLE_COUNTS[@]} -eq 1 ]]; then
    samples="${SAMPLE_COUNTS[0]}"
    echo ""
    echo "┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐"
    echo "│  QUICK COMPARISON: Backend vs Backend — MEAN Launch-Start Avg μs over $REPEATS repeats ($samples samples each)     │"
    echo "└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘"
    echo ""
    hdr="%-8s"
    for backend in "${BACKENDS[@]}"; do
        hdr+=" │ %8s %8s %8s"
    done
    hdr+="\n"
    hdr_args=("Model")
    for backend in "${BACKENDS[@]}"; do
        hdr_args+=("${backend} BL" "${backend} GC" "${backend} Δ%")
    done
    printf "$hdr" "${hdr_args[@]}"
    printf '─%.0s' {1..$(( 10 + ${#BACKENDS[@]} * 30 ))}; echo ""

    for config in "${MODEL_CONFIGS[@]}"; do
        IFS=':' read -r label hidden layers input_size <<< "$config"
        row_args=("$label")
        for backend in "${BACKENDS[@]}"; do
            sum_bl=0; sum_gc=0; sum_d=0; n_valid=0
            for repeat in $(seq 1 "$REPEATS"); do
                run_id="${backend}_${label}_s${samples}_r${repeat}"
                log_file="$RESULTS_DIR/${run_id}.log"
                if [[ -f "$log_file" ]]; then
                    parse_log "$log_file"
                    if [[ -n "${bl_launch_avg}" && -n "${gc_launch_avg}" && -n "${d_launch_avg}" ]]; then
                        sum_bl=$(awk "BEGIN{printf \"%.2f\", $sum_bl + ${bl_launch_avg}}")
                        sum_gc=$(awk "BEGIN{printf \"%.2f\", $sum_gc + ${gc_launch_avg}}")
                        sum_d=$(awk "BEGIN{printf \"%.2f\", $sum_d + ${d_launch_avg/+/}}")
                        n_valid=$((n_valid + 1))
                    fi
                fi
            done
            if [[ $n_valid -gt 0 ]]; then
                row_args+=("$(awk "BEGIN{printf \"%.2f\", $sum_bl/$n_valid}")")
                row_args+=("$(awk "BEGIN{printf \"%.2f\", $sum_gc/$n_valid}")")
                row_args+=("$(awk "BEGIN{printf \"%.2f\", $sum_d/$n_valid}")")
            else
                row_args+=("-" "-" "-")
            fi
        done
        printf "$hdr" "${row_args[@]}"
    done
fi

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo " Full logs: $RESULTS_DIR/"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
ls -lh "$RESULTS_DIR"/*.log 2>/dev/null || echo "  (no log files found)"
