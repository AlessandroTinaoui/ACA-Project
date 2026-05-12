#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p build/host

gcc -O2 -Wall -Wextra -Iinclude \
  src/main.c src/bspline.c src/kan_inference.c \
  -lm \
  -o build/host/kan_demo_host

echo "Built build/host/kan_demo_host"
