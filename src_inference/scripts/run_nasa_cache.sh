#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GEM5_BIN="../gem5/build/RISCV/gem5.opt"
MODEL_ARG="${1:-nasa}"
NUM_INPUTS="${2:-0}"
RUN_NAME="${3:-nasa_kan}"

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

NUM_INPUTS="$(resolve_num_inputs "$NUM_INPUTS")"

if [[ ! -x "$GEM5_BIN" ]]; then
  echo "Error: gem5 binary not found or not executable: $GEM5_BIN" >&2
  echo "Build gem5 for RISC-V first, then rerun this script." >&2
  exit 1
fi

bash scripts/build_nasa_riscv.sh "$MODEL_ARG" "$NUM_INPUTS"

OUTDIR="results/cache/$RUN_NAME"
ARTIFACTS_ROOT="simulation_metrics"
MODE_TAG="cache_l1_l2"
RUN_TAG="riscv_se_${MODE_TAG}_${RUN_NAME}"
REPORT_DIR="$ARTIFACTS_ROOT/$MODE_TAG/$RUN_NAME"
REPORT_PATH="$REPORT_DIR/${RUN_TAG}_report.md"
SUMMARY_PATH="$REPORT_DIR/${RUN_TAG}_summary.md"

mkdir -p "$OUTDIR"
mkdir -p "$REPORT_DIR"

echo "Running cached NASA fp32 simulation with model: $MODEL_ARG"
echo "Output directory: $OUTDIR"

"$GEM5_BIN" \
  --outdir="$OUTDIR" \
  gem5-configs/riscv_cache.py \
  --binary build/riscv/nasa_kan_demo_riscv \
  --num-inputs "$NUM_INPUTS" | tee "$OUTDIR/simout"

python3 scripts/collect_stats.py \
  --stats "$OUTDIR/stats.txt" \
  --config "$OUTDIR/config.json" \
  --out "$REPORT_PATH" \
  --summary-out "$SUMMARY_PATH" \
  --format md \
  --stats-section first \
  --title "gem5 report $RUN_TAG"

