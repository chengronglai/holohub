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
SAMPLE_COUNTS=(1000)
REPEATS=3
WARMUP_SAMPLES=100
MODE="all"  # baseline | green-context | all

# SM partition sizes to sweep. 0 = auto (half GPU).
SMS_CONFIGS=(8 16 32 48 0)

# Model configurations: "label:hidden_size:num_layers:input_size"
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
        --sms)          IFS=' ' read -ra SMS_CONFIGS <<< "$2"; shift 2 ;;
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
  --sms "N1 N2 ..."      SM partition sizes to sweep (default: "8 16 32 48 0")
                         0 = auto (half GPU). Multiple values to find sweet spot
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

TOTAL_RUNS=$(( ${#BACKENDS[@]} * ${#MODEL_CONFIGS[@]} * ${#SAMPLE_COUNTS[@]} * ${#SMS_CONFIGS[@]} * REPEATS ))

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║            Green Context Inference Benchmark Sweep                    ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Results directory: $RESULTS_DIR"
echo "  Mode:             $MODE"
echo "  Backends:         ${BACKENDS[*]}"
echo "  SM partitions:    ${SMS_CONFIGS[*]}"
echo "  Sample counts:    ${SAMPLE_COUNTS[*]}"
echo "  Repeats:          $REPEATS"
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
    local backend="$1" label="$2" model_path="$3" input_size="$4" samples="$5" sms="$6" repeat="$7"
    local sms_label; [[ "$sms" == "0" ]] && sms_label="auto" || sms_label="${sms}sm"
    local run_id="${backend}_${label}_${sms_label}_s${samples}_r${repeat}"
    local log_file="$RESULTS_DIR/${run_id}.log"

    echo "  [run] $run_id ..."

    if $LOCAL_MODE; then
        local run_args="--samples $samples --warmup-samples $WARMUP_SAMPLES --mode $MODE --input-size $input_size --model-path $model_path --backend $backend --sms-per-partition $sms"
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
        local run_args="--samples $samples --warmup-samples $WARMUP_SAMPLES --mode $MODE --input-size $input_size --model-path $container_model_path --backend $backend --sms-per-partition $sms"

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
            for sms in "${SMS_CONFIGS[@]}"; do
                local_sms_label=$([[ "$sms" == "0" ]] && echo "auto" || echo "${sms}sm")
                for repeat in $(seq 1 "$REPEATS"); do
                    RUN_NUM=$((RUN_NUM + 1))
                    echo ""
                    echo "──── Run $RUN_NUM / $TOTAL_RUNS: backend=$backend, model=$label, sms=$local_sms_label, repeat=$repeat/$REPEATS ────"
                    run_benchmark "$backend" "$label" "$model_path" "$input_size" "$samples" "$sms" "$repeat"
                done
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

printf "%-7s %-8s %5s %3s │ %8s %8s %8s %8s │ %8s %8s %8s %8s │ %8s %8s %8s\n" \
    "Backend" "Model" "SMs" "Run" \
    "BL Avg" "BL P50" "BL P95" "BL P99" \
    "GC Avg" "GC P50" "GC P95" "GC P99" \
    "Δ Avg%" "Δ P95%" "Δ P99%"
printf '─%.0s' {1..138}; echo ""

for backend in "${BACKENDS[@]}"; do
    for config in "${MODEL_CONFIGS[@]}"; do
        IFS=':' read -r label hidden layers input_size <<< "$config"
        for samples in "${SAMPLE_COUNTS[@]}"; do
            for sms in "${SMS_CONFIGS[@]}"; do
                sms_label=$([[ "$sms" == "0" ]] && echo "auto" || echo "$sms")
                sum_bl_avg=0; sum_gc_avg=0; sum_d_avg=0; n_valid=0

                for repeat in $(seq 1 "$REPEATS"); do
                    run_id="${backend}_${label}_${sms_label/auto/auto}_s${samples}_r${repeat}"
                    [[ "$sms" != "0" ]] && run_id="${backend}_${label}_${sms}sm_s${samples}_r${repeat}"
                    log_file="$RESULTS_DIR/${run_id}.log"
                    if [[ ! -f "$log_file" ]]; then
                        printf "%-7s %-8s %5s  r%d │ %8s %8s %8s %8s │ %8s %8s %8s %8s │ %8s %8s %8s\n" \
                            "$backend" "$label" "$sms_label" "$repeat" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-"
                        continue
                    fi
                    parse_log "$log_file"
                    printf "%-7s %-8s %5s  r%d │ %8s %8s %8s %8s │ %8s %8s %8s %8s │ %8s %8s %8s\n" \
                        "$backend" "$label" "$sms_label" "$repeat" \
                        "${bl_launch_avg:--}" "${bl_launch_p50:--}" "${bl_launch_p95:--}" "${bl_launch_p99:--}" \
                        "${gc_launch_avg:--}" "${gc_launch_p50:--}" "${gc_launch_p95:--}" "${gc_launch_p99:--}" \
                        "${d_launch_avg:--}" "${d_launch_p95:--}" "${d_launch_p99:--}"

                    if [[ -n "${bl_launch_avg}" && -n "${gc_launch_avg}" && -n "${d_launch_avg}" ]]; then
                        sum_bl_avg=$(awk "BEGIN{printf \"%.2f\", $sum_bl_avg + ${bl_launch_avg}}")
                        sum_gc_avg=$(awk "BEGIN{printf \"%.2f\", $sum_gc_avg + ${gc_launch_avg}}")
                        sum_d_avg=$(awk "BEGIN{printf \"%.2f\", $sum_d_avg + ${d_launch_avg/+/}}")
                        n_valid=$((n_valid + 1))
                    fi
                done

                if [[ $n_valid -gt 0 ]]; then
                    mean_bl=$(awk "BEGIN{printf \"%.2f\", $sum_bl_avg / $n_valid}")
                    mean_gc=$(awk "BEGIN{printf \"%.2f\", $sum_gc_avg / $n_valid}")
                    mean_d=$(awk "BEGIN{printf \"%.2f\", $sum_d_avg / $n_valid}")
                    printf "\e[1m%-7s %-8s %5s MEAN│ %8s %8s %8s %8s │ %8s %8s %8s %8s │ %8s %8s %8s\e[0m\n" \
                        "$backend" "$label" "$sms_label" \
                        "$mean_bl" "" "" "" \
                        "$mean_gc" "" "" "" \
                        "$mean_d" "" ""
                fi
                printf '·%.0s' {1..138}; echo ""
            done
        done
    done
    printf '─%.0s' {1..138}; echo ""
done

# ─────────────────────────────────────────────────────────────────────────────
# Tables 2-4: MEAN-only summary across SM partition sizes
# (Per-repeat detail is in Table 1; these show compact MEAN per SM config)
# ─────────────────────────────────────────────────────────────────────────────

# Helper: compute MEAN of a variable across repeats for a given config
# Usage: compute_mean backend label sms samples variable_name
# Prints the mean value or "-"
compute_mean() {
    local be="$1" lb="$2" sm="$3" sa="$4" var="$5"
    local sm_lbl; [[ "$sm" == "0" ]] && sm_lbl="auto" || sm_lbl="${sm}sm"
    local sum=0 n=0
    for r in $(seq 1 "$REPEATS"); do
        local rid="${be}_${lb}_${sm_lbl}_s${sa}_r${r}"
        local lf="$RESULTS_DIR/${rid}.log"
        [[ -f "$lf" ]] || continue
        parse_log "$lf"
        local val="${!var}"
        if [[ -n "$val" ]]; then
            sum=$(awk "BEGIN{printf \"%.2f\", $sum + ${val/+/}}")
            n=$((n + 1))
        fi
    done
    [[ $n -gt 0 ]] && awk "BEGIN{printf \"%.2f\", $sum / $n}" || echo "-"
}

render_table \
    "TABLE 2: MEAN Launch-Start Latency by SM Partition Size (μs)" \
    "Shows how partition size affects scheduling. Find the sweet spot for each backend."

printf "%-7s %-8s %5s │ %9s %9s %9s │ %9s\n" \
    "Backend" "Model" "SMs" "BL Avg" "GC Avg" "Δ Avg%" "Infer Δ%"
printf '─%.0s' {1..65}; echo ""

for backend in "${BACKENDS[@]}"; do
    for config in "${MODEL_CONFIGS[@]}"; do
        IFS=':' read -r label hidden layers input_size <<< "$config"
        for samples in "${SAMPLE_COUNTS[@]}"; do
            for sms in "${SMS_CONFIGS[@]}"; do
                sms_label=$([[ "$sms" == "0" ]] && echo "auto" || echo "$sms")
                m_bl=$(compute_mean "$backend" "$label" "$sms" "$samples" "bl_launch_avg")
                m_gc=$(compute_mean "$backend" "$label" "$sms" "$samples" "gc_launch_avg")
                m_d=$(compute_mean "$backend" "$label" "$sms" "$samples" "d_launch_avg")
                m_ik=$(compute_mean "$backend" "$label" "$sms" "$samples" "ik_d_avg")
                printf "%-7s %-8s %5s │ %9s %9s %9s │ %9s\n" \
                    "$backend" "$label" "$sms_label" \
                    "$m_bl" "$m_gc" "$m_d" "$m_ik"
            done
            printf '·%.0s' {1..65}; echo ""
        done
    done
    printf '─%.0s' {1..65}; echo ""
done

render_table \
    "TABLE 3: MEAN InferenceOp Compute Time by SM Partition Size (μs)" \
    "Wall-clock InferenceOp::compute(). Positive Δ = GC slows inference (fewer SMs)."

printf "%-7s %-8s %5s │ %9s %9s %9s\n" \
    "Backend" "Model" "SMs" "BL Avg" "GC Avg" "Δ Avg%"
printf '─%.0s' {1..50}; echo ""

for backend in "${BACKENDS[@]}"; do
    for config in "${MODEL_CONFIGS[@]}"; do
        IFS=':' read -r label hidden layers input_size <<< "$config"
        for samples in "${SAMPLE_COUNTS[@]}"; do
            for sms in "${SMS_CONFIGS[@]}"; do
                sms_label=$([[ "$sms" == "0" ]] && echo "auto" || echo "$sms")
                m_bl=$(compute_mean "$backend" "$label" "$sms" "$samples" "bl_infer_avg")
                m_gc=$(compute_mean "$backend" "$label" "$sms" "$samples" "gc_infer_avg")
                m_d=$(compute_mean "$backend" "$label" "$sms" "$samples" "d_infer_avg")
                printf "%-7s %-8s %5s │ %9s %9s %9s\n" \
                    "$backend" "$label" "$sms_label" "$m_bl" "$m_gc" "$m_d"
            done
            printf '·%.0s' {1..50}; echo ""
        done
    done
    printf '─%.0s' {1..50}; echo ""
done

render_table \
    "TABLE 4: MEAN Inference Per-Kernel Execution Time by SM Partition Size (μs)" \
    "Positive Δ = GC partitioning inference to fewer SMs. Near-zero = kernels not saturating."

printf "%-7s %-8s %5s │ %9s %9s %9s\n" \
    "Backend" "Model" "SMs" "BL Avg" "GC Avg" "Δ Avg%"
printf '─%.0s' {1..50}; echo ""

for backend in "${BACKENDS[@]}"; do
    for config in "${MODEL_CONFIGS[@]}"; do
        IFS=':' read -r label hidden layers input_size <<< "$config"
        for samples in "${SAMPLE_COUNTS[@]}"; do
            for sms in "${SMS_CONFIGS[@]}"; do
                sms_label=$([[ "$sms" == "0" ]] && echo "auto" || echo "$sms")
                m_bl=$(compute_mean "$backend" "$label" "$sms" "$samples" "ik_bl_avg")
                m_gc=$(compute_mean "$backend" "$label" "$sms" "$samples" "ik_gc_avg")
                m_d=$(compute_mean "$backend" "$label" "$sms" "$samples" "ik_d_avg")
                printf "%-7s %-8s %5s │ %9s %9s %9s\n" \
                    "$backend" "$label" "$sms_label" "$m_bl" "$m_gc" "$m_d"
            done
            printf '·%.0s' {1..50}; echo ""
        done
    done
    printf '─%.0s' {1..50}; echo ""
done

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo " Full logs: $RESULTS_DIR/"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
ls -lh "$RESULTS_DIR"/*.log 2>/dev/null || echo "  (no log files found)"
