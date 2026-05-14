#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BITS="${1:-16}"
EXPORT_JSON="${2:-}"
NUM_INPUTS="${3:-128}"
LUT_SIZE="${4:-4096}"
GEM5_BIN="../gem5/build/RISCV/gem5.opt"

default_export_path() {
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

if [[ -z "$EXPORT_JSON" ]]; then
  EXPORT_JSON="$(default_export_path "$BITS")"
fi

if [[ ! -x "$GEM5_BIN" ]]; then
  echo "Error: gem5 binary not found or not executable: $GEM5_BIN" >&2
  echo "Build gem5 for RISC-V first, then rerun this script." >&2
  exit 1
fi

if [[ ! -f "$EXPORT_JSON" ]]; then
  echo "Error: quantized NASA export not found: $EXPORT_JSON" >&2
  exit 1
fi

bash scripts/build_nasa_quant_riscv.sh "$BITS" "$EXPORT_JSON" "$NUM_INPUTS" "$LUT_SIZE"

MODEL_STEM="$(basename "$EXPORT_JSON" .json)"
BINARY="build/riscv/nasa_kan_demo_quant_int${BITS}_riscv"
OUTDIR="results/cache_quant_int${BITS}/$MODEL_STEM"
ARTIFACTS_ROOT="simulation_metrics"
MODE_TAG="cache_l1_l2_quant_int${BITS}"
RUN_TAG="riscv_se_${MODE_TAG}_${MODEL_STEM}"
REPORT_DIR="$ARTIFACTS_ROOT/$MODE_TAG/$MODEL_STEM"
REPORT_PATH="$REPORT_DIR/${RUN_TAG}_report.md"
SUMMARY_PATH="$REPORT_DIR/${RUN_TAG}_summary.md"

mkdir -p "$OUTDIR"
mkdir -p "$REPORT_DIR"

echo "Running cached quantized NASA simulation with model: $EXPORT_JSON"
echo "Output directory: $OUTDIR"

"$GEM5_BIN" \
  --outdir="$OUTDIR" \
  gem5-configs/riscv_cache.py \
  --binary "$BINARY" \
  --num-inputs "$NUM_INPUTS" | tee "$OUTDIR/simout"

python3 scripts/collect_stats.py \
  --stats "$OUTDIR/stats.txt" \
  --config "$OUTDIR/config.json" \
  --out "$REPORT_PATH" \
  --summary-out "$SUMMARY_PATH" \
  --format md \
  --stats-section first \
  --title "gem5 report $RUN_TAG"
