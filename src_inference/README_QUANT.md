# Quantized KAN Pipeline

This repository keeps the `fp32` path separate from the quantized one.

## QuantKAN PTQ

Generate a quantized export from the trained checkpoint:

```bash
cd /home/alessandro/Projects/ACA-Project
./.venv/bin/python src_train/kan_models/nasa/quantize.py \
  --config src_train/configs/train_config.toml \
  --w-bit 16 \
  --a-bit 16 \
  --device cpu
```

This produces files under `artifacts/nasa_kan/`, including:

```text
nasa_kan_quantkan_uniform_w16a16_pc_export.json
nasa_kan_quantkan_uniform_w16a16_pc_test_predictions.csv
nasa_kan_quantkan_uniform_w16a16_pc_metrics.json
```

## gem5 Run

Run the quantized RISC-V binary with caches:

```bash
bash scripts/run/gem5.sh quant 16 \
  ../artifacts/nasa_kan/nasa_kan_quantkan_uniform_w16a16_pc_export.json \
  128
```

Results go to:

```text
results/cache_quant_int16/<export_stem>/
simulation_metrics/cache_l1_l2_quant_int16/<export_stem>/
```

## true-int path

Run it on `gem5`:

```bash
bash scripts/run/gem5.sh true-int 16 \
  ../artifacts/nasa_kan/nasa_kan_quantkan_uniform_w16a16_pc_export.json \
  128
```

Results go to:

```text
results/cache_true_int16/<run_name>/
simulation_metrics/cache_l1_l2_true_int16/<run_name>/
```

## Full comparison suite

Run fp32, PTQ `w16/a16`, PTQ `w8/a8`, `true-int16`, and `true-int8` end-to-end:

```bash
cd /home/alessandro/Projects/ACA-Project
bash src_inference/scripts/run/suite.sh \
  src_train/configs/train_config.toml \
  0
```

`0` means the full test set. The suite writes a compact comparison to:

```text
src_inference/simulation_metrics/compare/fp32_vs_quant_full.md
```
