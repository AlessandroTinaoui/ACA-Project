#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f include/mlp_model.h ]]; then
  python3 scripts/json_to_mlp_header.py
fi

mkdir -p build/host

gcc -O2 -Wall -Wextra -Iinclude \
  src/mlp_main.c src/mlp_inference.c \
  -lm \
  -o build/host/mlp_demo_host

echo "Built build/host/mlp_demo_host"
