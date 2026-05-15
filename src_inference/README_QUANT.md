# Quantized KAN Pipeline

This repository keeps the `fp32` NASA path separate from the quantized one.

## NASA QuantKAN PTQ

Generate a quantized NASA export from the trained checkpoint:

```bash
cd /home/alessandro/Projects/ACA-Project
./.venv/bin/python src_train/kan_models/models/nasa/quantize.py \
  --config src_train/configs/nasa/default.toml \
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

## Host Build

Build the quantized NASA host binary:

```bash
cd /home/alessandro/Projects/ACA-Project/src_inference
bash scripts/build_nasa_quant_host.sh 16 \
  ../artifacts/nasa_kan/nasa_kan_quantkan_uniform_w16a16_pc_export.json \
  128
./build/host/nasa_kan_demo_quant_int16_host 128
```

Arguments:

- first: quant bit-width, `8` or `16`
- second: quantized export JSON
- third: max NASA samples embedded in `nasa_test_data.h`

## gem5 Run

Run the quantized NASA binary with caches:

```bash
bash scripts/run_nasa_quant_cache.sh 16 \
  ../artifacts/nasa_kan/nasa_kan_quantkan_uniform_w16a16_pc_export.json \
  128
```

Results go to:

```text
results/cache_quant_int16/<export_stem>/
simulation_metrics/cache_l1_l2_quant_int16/<export_stem>/
```

## NASA true-int path

Build the separate integer-only-ish NASA backend that keeps activations in integer form between layers:

```bash
cd /home/alessandro/Projects/ACA-Project/src_inference
bash scripts/build_nasa_true_int_host.sh 16 \
  ../artifacts/nasa_kan/nasa_kan_quantkan_uniform_w16a16_pc_export.json \
  128
./build/host/nasa_kan_demo_true_int16_host 128
```

Run it on `gem5`:

```bash
bash scripts/run_nasa_true_int_cache.sh 16 \
  ../artifacts/nasa_kan/nasa_kan_quantkan_uniform_w16a16_pc_export.json \
  128
```

Results go to:

```text
results/cache_true_int16/<run_name>/
simulation_metrics/cache_l1_l2_true_int16/<run_name>/
```

## Full NASA comparison suite

Run fp32, PTQ `w16/a16`, PTQ `w8/a8`, `true-int16`, and `true-int8` end-to-end:

```bash
cd /home/alessandro/Projects/ACA-Project
bash src_inference/scripts/run_nasa_compare_suite.sh \
  src_train/configs/nasa/default.toml \
  0
```

`0` means the full NASA test set. The suite writes a compact comparison to:

```text
src_inference/simulation_metrics/nasa_compare/nasa_fp32_vs_quant_full.md
```
