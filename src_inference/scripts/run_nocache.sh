#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GEM5_BIN="../gem5/build/RISCV/gem5.opt"
BINARY="build/riscv/kan_demo_riscv"
MODEL_ARG="${1:-1x4x1}"
NUM_INPUTS="${2:-1024}"

resolve_model_json() {
  case "$1" in
    mini|mini_kan|1x1)
      echo "model/1x1"
      ;;
    1x2x1)
      echo "model/1x2x1"
      ;;
    1x4x1)
      echo "model/1x4x1"
      ;;
    1x8x1)
      echo "model/1x8x1"
      ;;
    *)
      echo "$1"
      ;;
  esac
}

MODEL_JSON="$(resolve_model_json "$MODEL_ARG")"
if [[ -d "$MODEL_JSON" ]]; then
  mapfile -t JSON_FILES < <(find "$MODEL_JSON" -maxdepth 1 -type f -name '*.json' | sort)
  if [[ "${#JSON_FILES[@]}" -eq 0 ]]; then
    echo "Error: no JSON export found in model directory: $MODEL_JSON" >&2
    exit 1
  fi
  if [[ "${#JSON_FILES[@]}" -gt 1 ]]; then
    echo "Error: multiple JSON exports found in $MODEL_JSON; pass one JSON path explicitly." >&2
    printf '  %s\n' "${JSON_FILES[@]}" >&2
    exit 1
  fi
  MODEL_JSON="${JSON_FILES[0]}"
  MODEL_STEM="$(basename "$(dirname "$MODEL_JSON")")"
else
  MODEL_STEM="$(basename "$MODEL_JSON" .json)"
fi
OUTDIR="results/nocache/$MODEL_STEM"
ARTIFACTS_ROOT="simulation_metrics"
MODE_TAG="nocache"
RUN_TAG="riscv_se_${MODE_TAG}_${MODEL_STEM}"
REPORT_DIR="$ARTIFACTS_ROOT/$MODE_TAG/$MODEL_STEM"
REPORT_PATH="$REPORT_DIR/${RUN_TAG}_report.md"
SUMMARY_PATH="$REPORT_DIR/${RUN_TAG}_summary.md"

if [[ ! -x "$GEM5_BIN" ]]; then
  echo "Error: gem5 binary not found or not executable: $GEM5_BIN" >&2
  echo "Build gem5 for RISC-V first, then rerun this script." >&2
  exit 1
fi

if [[ ! -f "$MODEL_JSON" ]]; then
  echo "Error: model JSON not found: $MODEL_JSON" >&2
  exit 1
fi

python3 scripts/json_to_header.py "$MODEL_JSON"
bash scripts/build_riscv.sh

mkdir -p "$OUTDIR"
mkdir -p "$REPORT_DIR"

echo "Running no-cache simulation with model: $MODEL_JSON"
echo "Output directory: $OUTDIR"

"$GEM5_BIN" \
  --outdir="$OUTDIR" \
  gem5-configs/riscv_nocache.py \
  --binary "$BINARY" \
  --num-inputs "$NUM_INPUTS"

python3 scripts/collect_stats.py \
  --stats "$OUTDIR/stats.txt" \
  --config "$OUTDIR/config.json" \
  --out "$REPORT_PATH" \
  --summary-out "$SUMMARY_PATH" \
  --format md \
  --title "gem5 report $RUN_TAG"
