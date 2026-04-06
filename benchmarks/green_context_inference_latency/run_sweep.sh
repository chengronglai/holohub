#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -u
set -o pipefail

APP="green_context_inference_latency"
BASE_IMG="nvcr.io/nvidia/clara-holoscan/holoscan:v4.0.0-cuda13"

SAMPLES=1000
REPEATS=10
FREQ_HZ=1000
MODES="all"

M_INPUT=64
M_HIDDEN=256
M_LAYERS=3

OUT_DIR="benchmarks/green_context_inference_latency/sweep_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_DIR}"
RESULTS_CSV="${OUT_DIR}/latency_results.csv"

# SM split configurations: name|measured_sms|contending_sms|c_input|c_hidden|c_layers
CONFIGS=(
  "split_16_120_light|16|120|512|1024|3"
  "split_16_120_heavy|16|120|1024|8192|10"

  "split_32_104_light|32|104|512|1024|3"
  "split_32_104_heavy|32|104|1024|8192|10"
)

BACKENDS=("trt" "onnxrt")

# none = no pinning (--user root), SCHED_FIFO = RT pinned (--privileged)
PIN_POLICIES=("none" "SCHED_FIFO")

CSV_HEADER="config,backend,pin_policy,repeat,measured_sms,contending_sms,c_input,c_hidden,c_layers"
CSV_HEADER+=",e2e_bl_avg,e2e_bl_std,e2e_bl_p95,e2e_bl_p99,e2e_gc_avg,e2e_gc_std,e2e_gc_p95,e2e_gc_p99"
CSV_HEADER+=",txp_bl_avg,txp_bl_std,txp_bl_p95,txp_bl_p99,txp_gc_avg,txp_gc_std,txp_gc_p95,txp_gc_p99"
CSV_HEADER+=",e2e_avg_chg_pct,e2e_p99_chg_pct,txp_avg_chg_pct,status"
echo "${CSV_HEADER}" > "${RESULTS_CSV}"

check_rt_prereqs() {
  local rt_runtime
  rt_runtime="$(< /proc/sys/kernel/sched_rt_runtime_us)"
  if [[ "${rt_runtime}" != "-1" ]]; then
    echo "ERROR: RT sweep requires unlimited host RT runtime."
    echo "Run: sudo sysctl -w kernel.sched_rt_runtime_us=-1"
    exit 1
  fi
}

# Extract stats from a named section (first arg = start pattern, second = stop pattern)
extract_section_metrics() {
  local logfile="$1"
  local start_pat="$2"
  local stop_pat="$3"
  awk -v start="${start_pat}" -v stop="${stop_pat}" '
    BEGIN {
      in_sec = 0; section = "";
      b_avg = b_std = b_p95 = b_p99 = "";
      g_avg = g_std = g_p95 = g_p99 = "";
    }
    $0 ~ start { in_sec = 1; next }
    $0 ~ stop  { in_sec = 0; next }
    in_sec && /^=== Comparison/ { section = ""; next }
    in_sec && /^=== Baseline ===/ { section = "baseline"; next }
    in_sec && /^=== Green Context ===/ { section = "gc"; next }
    in_sec && section == "baseline" && /^  Average:/ && b_avg == "" { b_avg = $2; next }
    in_sec && section == "baseline" && /^  Std Dev:/ && b_std == "" { b_std = $3; next }
    in_sec && section == "baseline" && /^  P95:/ && b_p95 == "" { b_p95 = $2; next }
    in_sec && section == "baseline" && /^  P99:/ && b_p99 == "" { b_p99 = $2; next }
    in_sec && section == "gc" && /^  Average:/ && g_avg == "" { g_avg = $2; next }
    in_sec && section == "gc" && /^  Std Dev:/ && g_std == "" { g_std = $3; next }
    in_sec && section == "gc" && /^  P95:/ && g_p95 == "" { g_p95 = $2; next }
    in_sec && section == "gc" && /^  P99:/ && g_p99 == "" { g_p99 = $2; next }
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
    for pin_policy in "${PIN_POLICIES[@]}"; do
      for run_idx in $(seq 1 "${REPEATS}"); do
        run_tag="${cfg_name}_${backend}_${pin_policy}_r${run_idx}"
        log_file="${OUT_DIR}/${run_tag}.log"

        run_args="--backend ${backend} --frequency-hz ${FREQ_HZ} --mode ${MODES} --samples ${SAMPLES}"
        run_args+=" --measured-input-size ${M_INPUT} --measured-hidden-size ${M_HIDDEN} --measured-layers ${M_LAYERS}"
        run_args+=" --contending-input-size ${c_input} --contending-hidden-size ${c_hidden} --contending-layers ${c_layers}"
        run_args+=" --measured-sms ${measured_sms} --contending-sms ${contending_sms}"

        if [[ "${pin_policy}" == "none" ]]; then
          docker_opts="--user root"
        else
          check_rt_prereqs
          run_args+=" --pin-measured-pipeline --scheduling-policy ${pin_policy}"
          docker_opts="--privileged --user root --ulimit rtprio=99"
        fi

        echo "--------------------------------------------------------------------------------"
        echo "Running ${run_tag}"
        echo "  backend=${backend}  pin=${pin_policy}  sms=${measured_sms}/${contending_sms}  model=${c_hidden}h${c_layers}l"
        echo "  log: ${log_file}"

        if ./holohub run "${APP}" \
          --docker-opts="${docker_opts}" \
          --base-img="${BASE_IMG}" \
          --run-args="${run_args}" 2>&1 | tee "${log_file}"; then

          e2e="$(extract_section_metrics "${log_file}" "End-to-End Pipeline Latency" "TxOp Firing Period")"
          txp="$(extract_section_metrics "${log_file}" "TxOp Firing Period" "Contending Inference Pipeline")"
          IFS=',' read -r e_ba e_bs e_bp95 e_bp99 e_ga e_gs e_gp95 e_gp99 <<< "${e2e}"
          IFS=',' read -r t_ba t_bs t_bp95 t_bp99 t_ga t_gs t_gp95 t_gp99 <<< "${txp}"
          if is_number "${e_ba}" && is_number "${e_bp99}" && is_number "${t_ba}"; then
            e2e_avg_chg="$(pct_change "${e_ba}" "${e_ga}")"
            e2e_p99_chg="$(pct_change "${e_bp99}" "${e_gp99}")"
            txp_avg_chg="$(pct_change "${t_ba}" "${t_ga}")"
            row="${cfg_name},${backend},${pin_policy},${run_idx},${measured_sms},${contending_sms},${c_input},${c_hidden},${c_layers}"
            row+=",${e_ba},${e_bs},${e_bp95},${e_bp99},${e_ga},${e_gs},${e_gp95},${e_gp99}"
            row+=",${t_ba},${t_bs},${t_bp95},${t_bp99},${t_ga},${t_gs},${t_gp95},${t_gp99}"
            row+=",${e2e_avg_chg},${e2e_p99_chg},${txp_avg_chg},OK"
            echo "${row}" >> "${RESULTS_CSV}"
          else
            echo "${cfg_name},${backend},${pin_policy},${run_idx},${measured_sms},${contending_sms},${c_input},${c_hidden},${c_layers},,,,,,,,,,,,,,,,,,,,PARSE_FAIL" >> "${RESULTS_CSV}"
          fi
        else
          echo "${cfg_name},${backend},${pin_policy},${run_idx},${measured_sms},${contending_sms},${c_input},${c_hidden},${c_layers},,,,,,,,,,,,,,,,,,,,FAIL" >> "${RESULTS_CSV}"
        fi
      done
    done
  done
done

echo
echo "Sweep complete. Results CSV:"
echo "  ${RESULTS_CSV}"
echo
echo "End-to-end latency table:"
column -s, -t "${RESULTS_CSV}"
