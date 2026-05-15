#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BITS="${1:-16}"
EXPORT_JSON="${2:-}"
MAX_SAMPLES="${3:-128}"

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

if [[ -z "$EXPORT_JSON" ]]; then
  EXPORT_JSON="$(default_export_path "$BITS")"
fi

PREDICTIONS_CSV="${EXPORT_JSON%_export.json}_test_predictions.csv"

mkdir -p build/host

python3 scripts/json_to_true_int_header.py "$EXPORT_JSON" --bits "$BITS"
if [[ -f "$PREDICTIONS_CSV" ]]; then
  python3 scripts/nasa_test_to_header.py --max-samples "$MAX_SAMPLES" --predictions "$PREDICTIONS_CSV"
else
  python3 scripts/nasa_test_to_header.py --max-samples "$MAX_SAMPLES"
fi

gcc -O2 -Wall -Wextra -Iinclude \
  src/nasa_true_int_main.c src/kan_true_int_inference.c \
  -lm \
  -o "build/host/nasa_kan_demo_true_int${BITS}_host"

echo "Built build/host/nasa_kan_demo_true_int${BITS}_host"

