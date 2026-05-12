#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v riscv64-linux-gnu-gcc >/dev/null 2>&1; then
  echo "Error: riscv64-linux-gnu-gcc not found." >&2
  echo "Install the RISC-V cross compiler, for example:" >&2
  echo "  sudo apt install gcc-riscv64-linux-gnu" >&2
  exit 1
fi

mkdir -p build/riscv

riscv64-linux-gnu-gcc -O2 -static -Wall -Wextra -Iinclude \
  src/main.c src/bspline.c src/kan_inference.c \
  -lm \
  -o build/riscv/kan_demo_riscv

echo "Built build/riscv/kan_demo_riscv"
