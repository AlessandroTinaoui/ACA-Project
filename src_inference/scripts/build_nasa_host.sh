#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_ARG="${1:-nasa}"
MAX_SAMPLES="${2:-0}"

mkdir -p build/host

python3 scripts/json_to_header.py "$MODEL_ARG"
python3 scripts/nasa_test_to_header.py --max-samples "$MAX_SAMPLES"

gcc -O2 -Wall -Wextra -Iinclude \
  src/nasa_main.c src/bspline.c src/kan_inference.c \
  -lm \
  -o build/host/nasa_kan_demo_host

echo "Built build/host/nasa_kan_demo_host"
