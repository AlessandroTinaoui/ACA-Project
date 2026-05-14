#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BITS="${1:-16}"
EXPORT_JSON="${2:-}"
MAX_SAMPLES="${3:-128}"
LUT_SIZE="${4:-4096}"
GEM5_ROOT="../gem5"
M5OPS_RISCV_SRC="$GEM5_ROOT/util/m5/src/abi/riscv/m5op.S"

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

if ! command -v riscv64-linux-gnu-gcc >/dev/null 2>&1; then
  echo "Error: riscv64-linux-gnu-gcc not found." >&2
  echo "Install the RISC-V cross compiler, for example:" >&2
  echo "  sudo apt install gcc-riscv64-linux-gnu" >&2
  exit 1
fi

if [[ ! -f "$M5OPS_RISCV_SRC" ]]; then
  echo "Error: gem5 RISC-V m5ops source not found: $M5OPS_RISCV_SRC" >&2
  echo "Check that the gem5 submodule is available next to src_inference." >&2
  exit 1
fi

if [[ -z "$EXPORT_JSON" ]]; then
  EXPORT_JSON="$(default_export_path "$BITS")"
fi

PREDICTIONS_CSV="${EXPORT_JSON%_export.json}_test_predictions.csv"

mkdir -p build/riscv

python3 scripts/json_to_quant_header.py "$EXPORT_JSON" --bits "$BITS" --lut-size "$LUT_SIZE"
if [[ -f "$PREDICTIONS_CSV" ]]; then
  python3 scripts/nasa_test_to_header.py --max-samples "$MAX_SAMPLES" --predictions "$PREDICTIONS_CSV"
else
  python3 scripts/nasa_test_to_header.py --max-samples "$MAX_SAMPLES"
fi

riscv64-linux-gnu-gcc -O2 -static -Wall -Wextra \
  -DKAN_ENABLE_GEM5_M5OPS=1 \
  -Iinclude -I"$GEM5_ROOT/include" \
  src/nasa_quant_main.c src/kan_quant_inference.c \
  "$M5OPS_RISCV_SRC" \
  -lm \
  -o "build/riscv/nasa_kan_demo_quant_int${BITS}_riscv"

echo "Built build/riscv/nasa_kan_demo_quant_int${BITS}_riscv"
