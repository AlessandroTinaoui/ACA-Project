#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GEM5_BIN="../gem5/build/RISCV/gem5.opt"
BINARY="build/riscv/mlp_demo_riscv"
MODEL_JSON="${1:-model/mlp}"
NUM_INPUTS="${2:-1024}"
OUTDIR="results/mlp_l1_l2"

if [[ ! -x "$GEM5_BIN" ]]; then
  echo "Error: gem5 binary not found or not executable: $GEM5_BIN" >&2
  echo "Build gem5 for RISC-V first, then rerun this script." >&2
  exit 1
fi

if [[ -d "$MODEL_JSON" ]]; then
  mapfile -t JSON_FILES < <(find "$MODEL_JSON" -maxdepth 1 -type f -name '*.json' | sort)
  if [[ "${#JSON_FILES[@]}" -eq 0 ]]; then
    echo "Error: no MLP JSON export found in model directory: $MODEL_JSON" >&2
    exit 1
  fi
  if [[ "${#JSON_FILES[@]}" -gt 1 ]]; then
    echo "Error: multiple MLP JSON exports found in $MODEL_JSON; pass one JSON path explicitly." >&2
    printf '  %s\n' "${JSON_FILES[@]}" >&2
    exit 1
  fi
  MODEL_JSON="${JSON_FILES[0]}"
fi

if [[ ! -f "$MODEL_JSON" ]]; then
  echo "Error: MLP model JSON not found: $MODEL_JSON" >&2
  exit 1
fi

python3 scripts/json_to_mlp_header.py "$MODEL_JSON"
bash scripts/build_mlp_riscv.sh

mkdir -p "$OUTDIR"

echo "Running MLP L1+L2 simulation with model: $MODEL_JSON"
echo "Output directory: $OUTDIR"

"$GEM5_BIN" \
  --outdir="$OUTDIR" \
  gem5-configs/riscv_mlp_l1_l2.py \
  --binary "$BINARY" \
  --num-inputs "$NUM_INPUTS" | tee "$OUTDIR/simout"
