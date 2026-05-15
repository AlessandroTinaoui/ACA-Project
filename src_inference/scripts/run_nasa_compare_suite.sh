#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "./.venv/bin/python" ]]; then
    PYTHON_BIN="./.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

CONFIG_PATH="${1:-src_train/configs/nasa/default.toml}"
NUM_INPUTS="${2:-0}"
FP32_RUN_NAME="${3:-nasa_fp32_full}"
Q16_RUN_NAME="${4:-nasa_quant_w16a16_full}"
Q8_RUN_NAME="${5:-nasa_quant_w8a8_full}"
TI16_RUN_NAME="${6:-nasa_true_int_w16a16_full}"
TI8_RUN_NAME="${7:-nasa_true_int_w8a8_full}"

resolve_num_inputs() {
  local requested="$1"
  if [[ "$requested" != "0" ]]; then
    echo "$requested"
    return
  fi

  "$PYTHON_BIN" - <<'PY'
import numpy as np
from pathlib import Path

y_test = Path("datasets/NASA/processed/Y_test.npy")
if not y_test.exists():
    raise SystemExit("Error: datasets/NASA/processed/Y_test.npy not found.")
print(int(np.load(y_test).reshape(-1).shape[0]))
PY
}

NUM_INPUTS="$(resolve_num_inputs "$NUM_INPUTS")"

echo "[1/6] Train NASA fp32"
"$PYTHON_BIN" src_train/kan_models/models/nasa/main.py --config "$CONFIG_PATH"

echo "[2/6] Run NASA fp32 on gem5"
bash src_inference/scripts/run_nasa_cache.sh nasa "$NUM_INPUTS" "$FP32_RUN_NAME"

echo "[3/6] Quantize NASA w16/a16"
"$PYTHON_BIN" src_train/kan_models/models/nasa/quantize.py \
  --config "$CONFIG_PATH" \
  --w-bit 16 \
  --a-bit 16 \
  --device cpu

echo "[4/8] Run NASA w16/a16 on gem5"
bash src_inference/scripts/run_nasa_quant_cache.sh 16 \
  "../artifacts/nasa_kan/nasa_kan_quantkan_uniform_w16a16_pc_export.json" \
  "$NUM_INPUTS" \
  "$Q16_RUN_NAME"

echo "[5/8] Run NASA true-int16 on gem5"
bash src_inference/scripts/run_nasa_true_int_cache.sh 16 \
  "../artifacts/nasa_kan/nasa_kan_quantkan_uniform_w16a16_pc_export.json" \
  "$NUM_INPUTS" \
  "$TI16_RUN_NAME"

echo "[6/8] Quantize NASA w8/a8"
"$PYTHON_BIN" src_train/kan_models/models/nasa/quantize.py \
  --config "$CONFIG_PATH" \
  --w-bit 8 \
  --a-bit 8 \
  --device cpu

echo "[7/8] Run NASA w8/a8 on gem5"
bash src_inference/scripts/run_nasa_quant_cache.sh 8 \
  "../artifacts/nasa_kan/nasa_kan_quantkan_uniform_w8a8_pc_export.json" \
  "$NUM_INPUTS" \
  "$Q8_RUN_NAME"

echo "[8/8] Run NASA true-int8 on gem5"
bash src_inference/scripts/run_nasa_true_int_cache.sh 8 \
  "../artifacts/nasa_kan/nasa_kan_quantkan_uniform_w8a8_pc_export.json" \
  "$NUM_INPUTS" \
  "$TI8_RUN_NAME"

FP32_METRICS="artifacts/nasa_kan/metrics.json"
FP32_SUMMARY="src_inference/simulation_metrics/cache_l1_l2/${FP32_RUN_NAME}/riscv_se_cache_l1_l2_${FP32_RUN_NAME}_summary.md"
FP32_SIMOUT="src_inference/results/cache/${FP32_RUN_NAME}/simout"
Q16_METRICS="artifacts/nasa_kan/nasa_kan_quantkan_uniform_w16a16_pc_metrics.json"
Q16_SUMMARY="src_inference/simulation_metrics/cache_l1_l2_quant_int16/${Q16_RUN_NAME}/riscv_se_cache_l1_l2_quant_int16_${Q16_RUN_NAME}_summary.md"
Q16_SIMOUT="src_inference/results/cache_quant_int16/${Q16_RUN_NAME}/simout"
Q8_METRICS="artifacts/nasa_kan/nasa_kan_quantkan_uniform_w8a8_pc_metrics.json"
Q8_SUMMARY="src_inference/simulation_metrics/cache_l1_l2_quant_int8/${Q8_RUN_NAME}/riscv_se_cache_l1_l2_quant_int8_${Q8_RUN_NAME}_summary.md"
Q8_SIMOUT="src_inference/results/cache_quant_int8/${Q8_RUN_NAME}/simout"
TI16_SUMMARY="src_inference/simulation_metrics/cache_l1_l2_true_int16/${TI16_RUN_NAME}/riscv_se_cache_l1_l2_true_int16_${TI16_RUN_NAME}_summary.md"
TI8_SUMMARY="src_inference/simulation_metrics/cache_l1_l2_true_int8/${TI8_RUN_NAME}/riscv_se_cache_l1_l2_true_int8_${TI8_RUN_NAME}_summary.md"
TI16_SIMOUT="src_inference/results/cache_true_int16/${TI16_RUN_NAME}/simout"
TI8_SIMOUT="src_inference/results/cache_true_int8/${TI8_RUN_NAME}/simout"
COMPARE_OUT="src_inference/simulation_metrics/nasa_compare/nasa_fp32_vs_quant_full.md"

"$PYTHON_BIN" src_inference/scripts/compare_nasa_runs.py \
  --fp32-metrics "$FP32_METRICS" \
  --fp32-summary "$FP32_SUMMARY" \
  --fp32-simout "$FP32_SIMOUT" \
  --q16-metrics "$Q16_METRICS" \
  --q16-summary "$Q16_SUMMARY" \
  --q16-simout "$Q16_SIMOUT" \
  --q8-metrics "$Q8_METRICS" \
  --q8-summary "$Q8_SUMMARY" \
  --q8-simout "$Q8_SIMOUT" \
  --ti16-summary "$TI16_SUMMARY" \
  --ti8-summary "$TI8_SUMMARY" \
  --ti16-simout "$TI16_SIMOUT" \
  --ti8-simout "$TI8_SIMOUT" \
  --out "$COMPARE_OUT"

echo
echo "Completed NASA suite."
echo "Comparison report: $COMPARE_OUT"
