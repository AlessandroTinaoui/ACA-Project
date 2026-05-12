#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

NUM_INPUTS="${1:-1024}"
MLP_MODEL="${2:-model/mlp}"
KAN_MODELS=(1x1 1x2x1 1x4x1 1x8x1)

for model in "${KAN_MODELS[@]}"; do
  echo
  echo "==> Running KAN $model"
  bash scripts/run_cache.sh "$model" "$NUM_INPUTS"
done

echo
echo "==> Running MLP"
bash scripts/run_mlp_l1_l2.sh "$MLP_MODEL" "$NUM_INPUTS"

echo
echo "==> Comparing all models"
python3 scripts/compare_all_models.py --kan-models "${KAN_MODELS[@]}" --mlp-dir results/mlp_l1_l2
