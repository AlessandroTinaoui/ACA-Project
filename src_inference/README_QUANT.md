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
  128 \
  4096
./build/host/nasa_kan_demo_quant_int16_host 128
```

Arguments:

- first: quant bit-width, `8` or `16`
- second: quantized export JSON
- third: max NASA samples embedded in `nasa_test_data.h`
- fourth: LUT size used for the cubic basis lookup
  the default is `4096`

## gem5 Run

Run the quantized NASA binary with caches:

```bash
bash scripts/run_nasa_quant_cache.sh 16 \
  ../artifacts/nasa_kan/nasa_kan_quantkan_uniform_w16a16_pc_export.json \
  128 \
  4096
```

Results go to:

```text
results/cache_quant_int16/<export_stem>/
simulation_metrics/cache_l1_l2_quant_int16/<export_stem>/
```
