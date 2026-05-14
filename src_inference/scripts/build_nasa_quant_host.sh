#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BITS="${1:-16}"
EXPORT_JSON="${2:-}"
MAX_SAMPLES="${3:-128}"
LUT_SIZE="${4:-4096}"

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

python3 scripts/json_to_quant_header.py "$EXPORT_JSON" --bits "$BITS" --lut-size "$LUT_SIZE"
if [[ -f "$PREDICTIONS_CSV" ]]; then
  python3 scripts/nasa_test_to_header.py --max-samples "$MAX_SAMPLES" --predictions "$PREDICTIONS_CSV"
else
  python3 scripts/nasa_test_to_header.py --max-samples "$MAX_SAMPLES"
fi

gcc -O2 -Wall -Wextra -Iinclude \
  src/nasa_quant_main.c src/kan_quant_inference.c \
  -lm \
  -o "build/host/nasa_kan_demo_quant_int${BITS}_host"

echo "Built build/host/nasa_kan_demo_quant_int${BITS}_host"
