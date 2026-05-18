# ACA-Project

Training e inferenza per un solo modello: KAN regressivo NASA C-MAPSS RUL.

## Struttura

```text
datasets/NASA/      dataset e preprocessing NASA
src_train/          training, quantizzazione PTQ, config TOML
src_inference/      inferenza C, header generation, build RISC-V, gem5
artifacts/nasa_kan/ checkpoint, export JSON, metriche e predizioni generate
```

## Requisiti

Python 3.11+:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Per `src_inference` servono anche:

- `riscv64-linux-gnu-gcc` per build RISC-V
- `gem5/build/RISCV/gem5.opt`

## Training NASA

Preprocessing:

```bash
python3 datasets/NASA/preprocessing.py
```

Training fp32:

```bash
make nasa
```

oppure:

```bash
python3 src_train/kan_models/nasa/train.py --config src_train/configs/train_config.toml
```

Quantizzazione PTQ:

```bash
make nasa-quant NASA_QUANT_W_BITS=16 NASA_QUANT_A_BITS=16
```

Output principali:

```text
artifacts/nasa_kan/model.pt
artifacts/nasa_kan/nasa_kan_riscv_export.json
artifacts/nasa_kan/metrics.json
artifacts/nasa_kan/test_predictions.csv
artifacts/nasa_kan/nasa_kan_quantkan_uniform_w16a16_pc_export.json
```

## Inferenza NASA

Gli header C dei modelli e dei dati test sono generati dagli script e non vanno committati:

```text
src_inference/include/kan_model.h
src_inference/include/kan_model_quant.h
src_inference/include/kan_model_true_int.h
src_inference/include/rul_test_data.h
```

Build + run gem5 fp32:

```bash
make nasa-gem5 NASA_NUM_INPUTS=128
```

Run gem5 PTQ:

```bash
make nasa-quant-gem5 NASA_QUANT_W_BITS=16 NASA_NUM_INPUTS=128
```

Run gem5 true-int:

```bash
make nasa-true-int NASA_QUANT_W_BITS=16 NASA_NUM_INPUTS=128
```

Suite completa fp32 + PTQ + true-int:

```bash
make nasa-suite NASA_NUM_INPUTS=128
```

Per uso normale usa i target `make`; gli script diretti restano disponibili per debug:

```bash
bash src_inference/scripts/build/riscv.sh fp32 default 128
bash src_inference/scripts/build/riscv.sh quant 16 artifacts/nasa_kan/nasa_kan_quantkan_uniform_w16a16_pc_export.json 128
bash src_inference/scripts/build/riscv.sh true-int 16 artifacts/nasa_kan/nasa_kan_quantkan_uniform_w16a16_pc_export.json 128
```

```bash
bash src_inference/scripts/run/gem5.sh fp32 default 128
bash src_inference/scripts/run/gem5.sh quant 16 artifacts/nasa_kan/nasa_kan_quantkan_uniform_w16a16_pc_export.json 128
bash src_inference/scripts/run/gem5.sh true-int 16 artifacts/nasa_kan/nasa_kan_quantkan_uniform_w16a16_pc_export.json 128
```

## Makefile

```bash
make help
```

Target principali:

```text
install
nasa
nasa-quant
nasa-gem5
nasa-quant-gem5
nasa-true-int
nasa-suite
clean
```

## gem5

I run gem5 usano:

```text
gem5/build/RISCV/gem5.opt
```

Build tipica del submodule/cartella `gem5`:

```bash
cd gem5
scons build/RISCV/gem5.opt -j"$(nproc)"
cd ..
```
