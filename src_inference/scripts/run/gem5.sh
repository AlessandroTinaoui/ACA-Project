#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

MODE="${1:-fp32}"
if [[ $# -ge 1 ]]; then
  shift
fi

GEM5_BIN="../gem5/build/RISCV/gem5.opt"

default_quant_export() {
  case "$1" in
    8)
      echo "../artifacts/nasa_kan/nasa_kan_quantkan_uniform_w8a8_pc_export.json"
      ;;
    16)
      echo "../artifacts/nasa_kan/nasa_kan_quantkan_uniform_w16a16_pc_export.json"
      ;;
    *)
      echo "Unsupported quant bit-width: $1" >&2
      exit 1
      ;;
  esac
}

resolve_existing_path() {
  local path="$1"
  if [[ -z "$path" ]]; then
    return
  fi
  if [[ -f "$path" ]]; then
    echo "$path"
  elif [[ -f "../$path" ]]; then
    echo "../$path"
  else
    echo "$path"
  fi
}

resolve_num_inputs() {
  local requested="$1"
  if [[ "$requested" != "0" ]]; then
    echo "$requested"
    return
  fi

  python3 - <<'PY'
import numpy as np
from pathlib import Path

y_test = Path("../datasets/NASA/processed/Y_test.npy")
if not y_test.exists():
    raise SystemExit("Error: ../datasets/NASA/processed/Y_test.npy not found.")
print(int(np.load(y_test).reshape(-1).shape[0]))
PY
}

run_gem5() {
  local binary="$1"
  local num_inputs="$2"
  local outdir="$3"
  local report_path="$4"
  local summary_path="$5"
  local run_tag="$6"

  if [[ ! -x "$GEM5_BIN" ]]; then
    echo "Error: gem5 binary not found or not executable: $GEM5_BIN" >&2
    echo "Build gem5 for RISC-V first, then rerun this script." >&2
    exit 1
  fi

  mkdir -p "$outdir" "$(dirname "$report_path")"

  echo "Running KAN gem5 simulation"
  echo "Binary: $binary"
  echo "Output directory: $outdir"

  "$GEM5_BIN" \
    --outdir="$outdir" \
    gem5-configs/riscv_cache.py \
    --binary "$binary" \
    --num-inputs "$num_inputs" | tee "$outdir/simout"

  python3 scripts/stats/collect.py \
    --stats "$outdir/stats.txt" \
    --config "$outdir/config.json" \
    --out "$report_path" \
    --summary-out "$summary_path" \
    --format md \
    --stats-section first \
    --title "gem5 report $run_tag"
}

case "$MODE" in
  fp32)
    MODEL_ARG="${1:-default}"
    NUM_INPUTS="$(resolve_num_inputs "${2:-0}")"
    RUN_NAME="${3:-kan_fp32}"
    MODE_TAG="cache_l1_l2"
    RUN_TAG="riscv_se_${MODE_TAG}_${RUN_NAME}"

    bash scripts/build/riscv.sh fp32 "$MODEL_ARG" "$NUM_INPUTS"
    run_gem5 \
      "build/riscv/kan_riscv" \
      "$NUM_INPUTS" \
      "results/cache/$RUN_NAME" \
      "simulation_metrics/$MODE_TAG/$RUN_NAME/${RUN_TAG}_report.md" \
      "simulation_metrics/$MODE_TAG/$RUN_NAME/${RUN_TAG}_summary.md" \
      "$RUN_TAG"
    ;;
  quant|true-int)
    BITS="${1:-16}"
    EXPORT_JSON="${2:-}"
    NUM_INPUTS="$(resolve_num_inputs "${3:-0}")"
    RUN_NAME="${4:-}"
    if [[ -z "$EXPORT_JSON" ]]; then
      EXPORT_JSON="$(default_quant_export "$BITS")"
    else
      EXPORT_JSON="$(resolve_existing_path "$EXPORT_JSON")"
    fi
    if [[ ! -f "$EXPORT_JSON" ]]; then
      echo "Error: KAN export not found: $EXPORT_JSON" >&2
      exit 1
    fi

    if [[ "$MODE" == "quant" ]]; then
      [[ -n "$RUN_NAME" ]] || RUN_NAME="quant_int${BITS}"
      MODE_TAG="cache_l1_l2_quant_int${BITS}"
      BINARY="build/riscv/kan_quant_int${BITS}_riscv"
      OUTDIR="results/cache_quant_int${BITS}/$RUN_NAME"
    else
      [[ -n "$RUN_NAME" ]] || RUN_NAME="true_int${BITS}"
      MODE_TAG="cache_l1_l2_true_int${BITS}"
      BINARY="build/riscv/kan_true_int${BITS}_riscv"
      OUTDIR="results/cache_true_int${BITS}/$RUN_NAME"
    fi
    RUN_TAG="riscv_se_${MODE_TAG}_${RUN_NAME}"

    bash scripts/build/riscv.sh "$MODE" "$BITS" "$EXPORT_JSON" "$NUM_INPUTS"
    run_gem5 \
      "$BINARY" \
      "$NUM_INPUTS" \
      "$OUTDIR" \
      "simulation_metrics/$MODE_TAG/$RUN_NAME/${RUN_TAG}_report.md" \
      "simulation_metrics/$MODE_TAG/$RUN_NAME/${RUN_TAG}_summary.md" \
      "$RUN_TAG"
    ;;
  *)
    echo "Usage: $0 {fp32|quant|true-int} [args...]" >&2
    exit 1
    ;;
esac
