#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GEM5_ROOT="../gem5"
M5OPS_RISCV_SRC="$GEM5_ROOT/util/m5/src/abi/riscv/m5op.S"

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

mkdir -p build/riscv

riscv64-linux-gnu-gcc -O2 -static -Wall -Wextra \
  -DKAN_ENABLE_GEM5_M5OPS=1 \
  -Iinclude -I"$GEM5_ROOT/include" \
  src/main.c src/bspline.c src/kan_inference.c \
  "$M5OPS_RISCV_SRC" \
  -lm \
  -o build/riscv/kan_demo_riscv

echo "Built build/riscv/kan_demo_riscv"
