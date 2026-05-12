#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

KAN_MODEL="${1:-1x4x1}"
NUM_INPUTS="${2:-1024}"
MLP_MODEL="${3:-model/mlp}"

KAN_JSON="$KAN_MODEL"
case "$KAN_MODEL" in
  mini|mini_kan|1x1)
    KAN_JSON="model/1x1"
    ;;
  1x2x1)
    KAN_JSON="model/1x2x1"
    ;;
  1x4x1)
    KAN_JSON="model/1x4x1"
    ;;
  1x8x1)
    KAN_JSON="model/1x8x1"
    ;;
esac

if [[ -d "$KAN_JSON" ]]; then
  KAN_STEM="$(basename "$KAN_JSON")"
else
  KAN_STEM="$(basename "$KAN_JSON" .json)"
fi
KAN_DIR="results/cache/$KAN_STEM"
MLP_DIR="results/mlp_l1_l2"

echo "==> Running KAN"
echo "    model: $KAN_MODEL"
echo "    N:     $NUM_INPUTS"
bash scripts/run_cache.sh "$KAN_MODEL" "$NUM_INPUTS"

echo
echo "==> Running MLP"
echo "    model: $MLP_MODEL"
echo "    N:     $NUM_INPUTS"
bash scripts/run_mlp_l1_l2.sh "$MLP_MODEL" "$NUM_INPUTS"

echo
echo "==> Comparing KAN vs MLP"
python3 scripts/compare_kan_mlp.py \
  --kan-dir "$KAN_DIR" \
  --mlp-dir "$MLP_DIR"
