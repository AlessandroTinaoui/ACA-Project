# ACA-Project

Repository unico per:

- training dei modelli in `src_train`
- export JSON dei modelli
- conversione degli export in header C
- build host e RISC-V in `src_inference`
- simulazione su gem5
- confronto KAN vs MLP

## Struttura

```text
datasets/        dataset organizzati per sottocartella
src_train/       training KAN/MLP, config TOML, metriche e plot
src_inference/   inferenza C, conversione JSON->header, build RISC-V, gem5
artifacts/       export e checkpoint generati dal training
```

## Requisiti

Python 3.11+.

Dipendenze Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Per `src_inference` servono anche:

- `gcc` per build host
- `riscv64-linux-gnu-gcc` per build RISC-V
- `gem5` compilato in `gem5/build/RISCV/gem5.opt`

## Compilare gem5 per RISC-V

I target gem5 di questo repository si aspettano il binario qui:

```text
gem5/build/RISCV/gem5.opt
```

Quindi, dalla root di `ACA-Project`, `gem5` deve essere presente come submodule/cartella interna:

```text
./gem5
```

### 1. Installare le dipendenze di build

Su Ubuntu/Debian, una base ragionevole e`:

```bash
sudo apt update
sudo apt install -y build-essential git python3-dev python3-venv scons \
  m4 zlib1g zlib1g-dev libprotobuf-dev protobuf-compiler \
  libgoogle-perftools-dev libboost-all-dev pkg-config
```

### 2. Entrare nella repo gem5

```bash
cd gem5
```

### 3. Installare le dipendenze Python di gem5

Se vuoi usare un virtualenv dedicato:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Compilare il target RISC-V

Build ottimizzata:

```bash
scons build/RISCV/gem5.opt -j"$(nproc)"
```

Build debug, se ti serve:

```bash
scons build/RISCV/gem5.debug -j"$(nproc)"
```

### 5. Verificare il binario

```bash
ls -l build/RISCV/gem5.opt
```

Poi torna nella root di questo progetto:

```bash
cd ..
```

Da qui i target come:

```bash
make gem5-kan-cache
make gem5-mlp
make gem5-all
```

troveranno automaticamente `gem5/build/RISCV/gem5.opt`.

## Makefile di Root

Il punto di ingresso consigliato è:

```bash
make help
```

Target principali:

```text
install                  installa le dipendenze Python
train-fun-kan            train KAN 1D e export JSON
train-fun-mlp            train MLP 1D e export JSON
compare-fun-predictions  confronto Python-side tra export KAN e MLP
sync-inference-kan       copia un export KAN in src_inference/model/<arch>/
sync-inference-mlp       copia l'export MLP in src_inference/model/mlp/
conic                    esegue il workflow coniche dal TOML unificato
tabular-credit           training del modello credit-default
tabular-stroke           training del modello stroke
nasa                     training KAN regressivo NASA C-MAPSS RUL
inference-host           build host KAN
inference-riscv          build RISC-V KAN
gem5-kan-cache           run gem5 KAN con L1+L2
gem5-kan-l1              run gem5 KAN con sola L1
gem5-kan-nocache         run gem5 KAN senza cache
gem5-mlp                 run gem5 MLP con L1+L2
gem5-compare-one         confronto KAN vs MLP su risultati esistenti
gem5-compare-all         confronto di tutti i modelli
gem5-plots               genera CSV e PNG delle metriche
gem5-all                 esegue tutte le simulazioni standard e il confronto finale
```

## Training Modelli 1D

### KAN 1D

Config di default:

```text
src_train/kan_models/models/functions/fun_1_d/params.toml
```

Esecuzione:

```bash
make train-fun-kan
```

Oppure direttamente:

```bash
python3 src_train/kan_models/models/functions/fun_1_d/main.py \
  --params src_train/kan_models/models/functions/fun_1_d/params.toml
```

Output di default:

```text
artifacts/fun_1_d/kan_1_4_1/mini_kan_riscv_export.json
artifacts/fun_1_d/kan_1_4_1/mini_kan_metrics.json
artifacts/fun_1_d/kan_1_4_1/mini_kan_checkpoint.pt
artifacts/fun_1_d/kan_1_4_1/mini_kan_fit.png
```

Per cambiare architettura modifica `width` nel TOML. Il path finale usa il formato `kan_<width_con_underscore>`, per esempio:

- `[1, 1]` -> `artifacts/fun_1_d/kan_1_1/`
- `[1, 2, 1]` -> `artifacts/fun_1_d/kan_1_2_1/`
- `[1, 4, 1]` -> `artifacts/fun_1_d/kan_1_4_1/`
- `[1, 8, 1]` -> `artifacts/fun_1_d/kan_1_8_1/`

### MLP 1D

Config di default:

```text
src_train/kan_models/models/functions/sine_1d_mlp/params.toml
```

Esecuzione:

```bash
make train-fun-mlp
```

Oppure direttamente:

```bash
python3 src_train/kan_models/models/functions/sine_1d_mlp/main.py \
  --params src_train/kan_models/models/functions/sine_1d_mlp/params.toml
```

Output di default:

```text
artifacts/sine_1d_mlp/mlp_riscv_export.json
artifacts/sine_1d_mlp/mlp_metrics.json
artifacts/sine_1d_mlp/mlp_checkpoint.pt
artifacts/sine_1d_mlp/mlp_fit.png
artifacts/sine_1d_mlp/mlp_loss.png
```

### Confronto prediction-side prima di gem5

```bash
make compare-fun-predictions
```

Questo script confronta target, KAN ricostruito da export e MLP ricostruito da export.

## Deploy degli Export in `src_inference`

Per usare su C/gem5 un export addestrato in `src_train`, copialo nella struttura attesa da `src_inference/model`.

KAN 1x4x1:

```bash
make sync-inference-kan KAN_MODEL=1x4x1
```

MLP:

```bash
make sync-inference-mlp
```

Se l'export sorgente è custom, puoi sovrascrivere il path:

```bash
make sync-inference-kan \
  KAN_MODEL=1x2x1 \
  KAN_EXPORT_SRC=artifacts/fun_1_d/kan_1_2_1/mini_kan_riscv_export.json
```

## Training Coniche

Entry point unificato:

```bash
make conic
```

Equivale a:

```bash
python3 src_train/kan_models/models/conic/main.py \
  --config src_train/configs/conic/default.toml
```

Il TOML gestisce tre modalità:

- `mode = "baseline"`
- `mode = "pruning"`
- `mode = "continual"`

Dataset usato dal config di default:

```text
datasets/conic/Conic-Section_dataset.csv
```

Output tipici:

```text
src_train/conic_metrics.csv
src_train/conic_class_tests.csv
src_train/conic_run_config.json
src_train/conic_plots/
src_train/continual_metrics.csv
src_train/continual_plots/
src_train/continual_metrics_reversed.csv
src_train/continual_plots_reversed/
```

Per lanciare direttamente un continual run:

```bash
python3 src_train/kan_models/models/conic/main.py \
  --config src_train/configs/conic/default.toml
```

La variante (`standard` o `reversed`) si controlla dal TOML di default.

## Training Tabellare

Credit default:

```bash
make tabular-credit
```

oppure:

```bash
python3 src_train/kan_models/models/tabular/credit_default/main.py \
  --config src_train/configs/credit_default/default.toml
```

Stroke:

```bash
make tabular-stroke
```

oppure:

```bash
python3 src_train/kan_models/models/tabular/stroke/main.py \
  --config src_train/configs/stroke/pruning.toml
```

Dataset:

```text
datasets/credit_default/default of credit card clients.csv
datasets/stroke/healthcare-dataset-stroke-data.csv
```

## Training NASA C-MAPSS

Preprocessing FD001:

```bash
python3 datasets/NASA/preprocessing.py
```

Training KAN regressivo:

```bash
make nasa
```

oppure:

```bash
python3 src_train/kan_models/models/nasa/main.py \
  --config src_train/configs/nasa/default.toml
```

Dataset preprocessato:

```text
datasets/NASA/processed/train_windows.csv
datasets/NASA/processed/test_windows.csv
```

Output:

```text
artifacts/nasa_kan/model.pt
artifacts/nasa_kan/metrics.json
artifacts/nasa_kan/history.csv
artifacts/nasa_kan/test_predictions.csv
artifacts/nasa_kan/training_history.png
```

## Inference C e gem5

### Build host

KAN:

```bash
make inference-host
./src_inference/build/host/kan_demo_host 1024
```

MLP:

```bash
make inference-mlp-host
./src_inference/build/host/mlp_demo_host 1024
```

### Build RISC-V

KAN:

```bash
make inference-riscv
```

MLP:

```bash
make inference-mlp-riscv
```

### Run gem5

KAN con L1+L2:

```bash
make gem5-kan-cache KAN_MODEL=1x4x1 N=1024
```

KAN con sola L1:

```bash
make gem5-kan-l1 KAN_MODEL=1x4x1 N=1024
```

KAN senza cache:

```bash
make gem5-kan-nocache KAN_MODEL=1x4x1 N=1024
```

MLP con L1+L2:

```bash
make gem5-mlp N=1024
```

Workflow completo standard:

```bash
make gem5-all N=1024
```

### Confronti e plot

Confronto singolo KAN vs MLP:

```bash
make gem5-compare-one KAN_MODEL=1x4x1
```

Confronto tutti i modelli:

```bash
make gem5-compare-all
```

Plot metriche:

```bash
make gem5-plots
```

Output:

```text
src_inference/results/
src_inference/simulation_metrics/
src_inference/plots/model_metrics/metrics.csv
src_inference/plots/model_metrics/*.png
```

## Path Importanti

Training 1D:

```text
src_train/kan_models/models/functions/fun_1_d/params.toml
src_train/kan_models/models/functions/sine_1d_mlp/params.toml
```

Training coniche:

```text
src_train/configs/conic/default.toml
```

Export consumati da `src_inference`:

```text
src_inference/model/1x1/
src_inference/model/1x2x1/
src_inference/model/1x4x1/
src_inference/model/1x8x1/
src_inference/model/mlp/
```

Script chiave gem5:

```text
src_inference/scripts/run_cache.sh
src_inference/scripts/run_l1.sh
src_inference/scripts/run_nocache.sh
src_inference/scripts/run_mlp_l1_l2.sh
src_inference/scripts/compare_all_models.py
src_inference/scripts/plot_model_metrics.py
```

## Note Pratiche

- Gli script Python in `src_inference/scripts` ora risolvono i path rispetto a `src_inference`, quindi funzionano anche lanciati dalla root del repository.
- Il `Makefile` di root non sostituisce i singoli script: li incapsula.
- Se `gem5.opt` non esiste in `gem5/build/RISCV/`, i target gem5 falliranno fino a quando non compili gem5.
