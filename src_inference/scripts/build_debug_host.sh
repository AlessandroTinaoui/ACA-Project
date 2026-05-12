#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p build/host

if [[ ! -f include/mlp_model.h ]]; then
  python3 scripts/json_to_mlp_header.py model/mlp
fi

gcc -O2 -Wall -Wextra -Iinclude \
  src/debug_compare.c src/bspline.c src/kan_inference.c src/mlp_inference.c \
  -lm \
  -o build/host/debug_compare_host

echo "Built build/host/debug_compare_host"
