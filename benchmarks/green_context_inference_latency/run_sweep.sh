#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -u
set -o pipefail

APP="green_context_inference_latency"
BASE_IMG="nvcr.io/nvidia/clara-holoscan/holoscan:v3.10.0-cuda13"

SAMPLES=2000
REPEATS=2
FREQ_HZ=1000
MODES="all"

# Measured model (small control-loop policy -- kept fixed across sweep)
M_INPUT=64
M_HIDDEN=256
M_LAYERS=3

OUT_DIR="benchmarks/green_context_inference_latency/sweep_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_DIR}"
RESULTS_CSV="${OUT_DIR}/latency_results.csv"

# Sweep configurations: vary SM split and contending model complexity.
# Contending model knobs: input_size, hidden_size, num_layers.
# Larger hidden_size / more layers = heavier GPU contention.
#
# name|measured_sms|contending_sms|c_input|c_hidden|c_layers
CONFIGS=(
  # --- Split 8 / 128  ---
  "split_8_128_light|8|128|512|1024|3"
  "split_8_128_medium|8|128|1024|4096|6"
  "split_8_128_heavy|8|128|1024|8192|8"

  # --- Split 16 / 120  ---
  "split_16_120_light|16|120|512|1024|3"
  "split_16_120_medium|16|120|1024|4096|6"
  "split_16_120_heavy|16|120|1024|8192|8"

  # --- Split 32 / 104  ---
  "split_32_104_light|32|104|512|1024|3"
  "split_32_104_medium|32|104|1024|4096|6"
  "split_32_104_heavy|32|104|1024|8192|8"

  # --- Split 68 / 68  ---
  "split_68_68_light|68|68|512|1024|3"
  "split_68_68_medium|68|68|1024|4096|6"
  "split_68_68_heavy|68|68|1024|8192|8"

  # --- Split 104 / 32  ---
  "split_104_32_light|104|32|512|1024|3"
  "split_104_32_medium|104|32|1024|4096|6"
  "split_104_32_heavy|104|32|1024|8192|8"
)

BACKENDS=("trt")

echo "config,backend,repeat,measured_sms,contending_sms,c_input,c_hidden,c_layers,baseline_avg_us,baseline_std_us,baseline_p95_us,baseline_p99_us,gc_avg_us,gc_std_us,gc_p95_us,gc_p99_us,avg_change_pct,std_change_pct,p99_change_pct,status" > "${RESULTS_CSV}"

extract_e2e_metrics_to_csv_row() {
  local logfile="$1"
  awk '
    BEGIN {
      in_e2e = 0;
      section = "";
      b_avg = b_std = b_p95 = b_p99 = "";
      g_avg = g_std = g_p95 = g_p99 = "";
    }
    /End-to-End Pipeline Latency/ { in_e2e = 1; next }
    /Contending Inference Pipeline Throughput/ { in_e2e = 0; next }
    in_e2e && /^=== Comparison/ { section = ""; next }
    in_e2e && /^=== Baseline ===/ { section = "baseline"; next }
    in_e2e && /^=== Green Context ===/ { section = "gc"; next }
    in_e2e && section == "baseline" && /^  Average:/ && b_avg == "" { b_avg = $2; next }
    in_e2e && section == "baseline" && /^  Std Dev:/ && b_std == "" { b_std = $3; next }
    in_e2e && section == "baseline" && /^  P95:/ && b_p95 == "" { b_p95 = $2; next }
    in_e2e && section == "baseline" && /^  P99:/ && b_p99 == "" { b_p99 = $2; next }
    in_e2e && section == "gc" && /^  Average:/ && g_avg == "" { g_avg = $2; next }
    in_e2e && section == "gc" && /^  Std Dev:/ && g_std == "" { g_std = $3; next }
    in_e2e && section == "gc" && /^  P95:/ && g_p95 == "" { g_p95 = $2; next }
    in_e2e && section == "gc" && /^  P99:/ && g_p99 == "" { g_p99 = $2; next }
    END {
      printf "%s,%s,%s,%s,%s,%s,%s,%s\n", b_avg, b_std, b_p95, b_p99, g_avg, g_std, g_p95, g_p99
    }
  ' "${logfile}"
}

is_number() {
  [[ "$1" =~ ^-?[0-9]+([.][0-9]+)?$ ]]
}

pct_change() {
  local baseline="$1"
  local gc="$2"
  awk -v bl="${baseline}" -v gc_val="${gc}" 'BEGIN {
    if (bl == 0) { printf "NA"; }
    else { printf "%.2f", ((gc_val - bl) / bl) * 100.0; }
  }'
}

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r cfg_name measured_sms contending_sms c_input c_hidden c_layers <<< "${cfg}"

  for backend in "${BACKENDS[@]}"; do
    for run_idx in $(seq 1 "${REPEATS}"); do
      run_tag="${cfg_name}_${backend}_r${run_idx}"
      log_file="${OUT_DIR}/${run_tag}.log"

      run_args="--backend ${backend} --frequency-hz ${FREQ_HZ} --mode ${MODES} --samples ${SAMPLES}"
      run_args+=" --measured-input-size ${M_INPUT} --measured-hidden-size ${M_HIDDEN} --measured-layers ${M_LAYERS}"
      run_args+=" --contending-input-size ${c_input} --contending-hidden-size ${c_hidden} --contending-layers ${c_layers}"
      run_args+=" --measured-sms ${measured_sms} --contending-sms ${contending_sms}"

      echo "--------------------------------------------------------------------------------"
      echo "Running ${run_tag}"
      echo "run-args: ${run_args}"
      echo "log: ${log_file}"

      if ./holohub run "${APP}" \
        --docker-opts="--user root" \
        --base-img="${BASE_IMG}" \
        --run-args="${run_args}" 2>&1 | tee "${log_file}"; then

        metrics="$(extract_e2e_metrics_to_csv_row "${log_file}")"
        IFS=',' read -r b_avg b_std b_p95 b_p99 g_avg g_std g_p95 g_p99 <<< "${metrics}"
        if is_number "${b_avg}" && is_number "${b_std}" && is_number "${b_p95}" && is_number "${b_p99}" && \
           is_number "${g_avg}" && is_number "${g_std}" && is_number "${g_p95}" && is_number "${g_p99}"; then
          avg_change_pct="$(pct_change "${b_avg}" "${g_avg}")"
          std_change_pct="$(pct_change "${b_std}" "${g_std}")"
          p99_change_pct="$(pct_change "${b_p99}" "${g_p99}")"
          echo "${cfg_name},${backend},${run_idx},${measured_sms},${contending_sms},${c_input},${c_hidden},${c_layers},${b_avg},${b_std},${b_p95},${b_p99},${g_avg},${g_std},${g_p95},${g_p99},${avg_change_pct},${std_change_pct},${p99_change_pct},OK" >> "${RESULTS_CSV}"
        else
          echo "${cfg_name},${backend},${run_idx},${measured_sms},${contending_sms},${c_input},${c_hidden},${c_layers},,,,,,,,,,,,PARSE_FAIL" >> "${RESULTS_CSV}"
        fi
      else
        echo "${cfg_name},${backend},${run_idx},${measured_sms},${contending_sms},${c_input},${c_hidden},${c_layers},,,,,,,,,,,,FAIL" >> "${RESULTS_CSV}"
      fi
    done
  done
done

echo
echo "Sweep complete. Results CSV:"
echo "  ${RESULTS_CSV}"
echo
echo "End-to-end latency table:"
column -s, -t "${RESULTS_CSV}"
