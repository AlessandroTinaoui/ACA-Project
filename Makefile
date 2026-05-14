SHELL := /bin/bash

PYTHON ?= python3
PIP ?= pip
N ?= 1024
KAN_MODEL ?= 1x4x1

FUN_KAN_PARAMS ?= src_train/kan_models/models/functions/fun_1_d/params.toml
FUN_MLP_PARAMS ?= src_train/kan_models/models/functions/sine_1d_mlp/params.toml
CONIC_CONFIG ?= src_train/configs/conic/default.toml
CREDIT_CONFIG ?= src_train/configs/credit_default/default.toml
STROKE_CONFIG ?= src_train/configs/stroke/pruning.toml
NASA_CONFIG ?= src_train/configs/nasa/default.toml
NASA_QUANT_W_BITS ?= 16
NASA_QUANT_A_BITS ?= 16

KAN_MODEL_SLUG = $(subst x,_,$(KAN_MODEL))
KAN_EXPORT_SRC ?= artifacts/fun_1_d/kan_$(KAN_MODEL_SLUG)/mini_kan_riscv_export.json
MLP_EXPORT_SRC ?= artifacts/sine_1d_mlp/mlp_riscv_export.json

.PHONY: help install train-fun-kan train-fun-mlp compare-fun-predictions \
	sync-inference-kan sync-inference-mlp conic tabular-credit tabular-stroke nasa nasa-quant \
	inference-host inference-riscv inference-mlp-host inference-mlp-riscv \
	gem5-kan-cache gem5-kan-l1 gem5-kan-nocache gem5-mlp \
	gem5-compare-one gem5-compare-all gem5-plots gem5-all clean

help:
	@printf "%s\n" \
	"install                  Install Python dependencies from requirements.txt" \
	"train-fun-kan            Train/export the 1D KAN model" \
	"train-fun-mlp            Train/export the 1D MLP baseline" \
	"compare-fun-predictions  Compare exported KAN/MLP predictions in Python" \
	"sync-inference-kan       Copy one KAN export into src_inference/model/<KAN_MODEL>/" \
	"sync-inference-mlp       Copy the MLP export into src_inference/model/mlp/" \
	"conic                    Run the unified conic experiment" \
	"tabular-credit           Run the credit-default tabular experiment" \
	"tabular-stroke           Run the stroke tabular experiment" \
	"nasa                     Run the NASA C-MAPSS RUL KAN regression" \
	"nasa-quant               Apply QuantKAN PTQ to the NASA KAN checkpoint" \
	"inference-host           Build the host KAN binary" \
	"inference-riscv          Build the RISC-V KAN binary" \
	"inference-mlp-host       Build the host MLP binary" \
	"inference-mlp-riscv      Build the RISC-V MLP binary" \
	"gem5-kan-cache           Run gem5 KAN with L1+L2 cache" \
	"gem5-kan-l1              Run gem5 KAN with L1 only" \
	"gem5-kan-nocache         Run gem5 KAN without cache" \
	"gem5-mlp                 Run gem5 MLP with L1+L2 cache" \
	"gem5-compare-one         Compare KAN_MODEL results against MLP results" \
	"gem5-compare-all         Compare all cached KAN runs against the MLP run" \
	"gem5-plots               Generate metrics CSV and PNG plots" \
	"gem5-all                 Run all standard gem5 comparisons" \
	"clean                    Clean src_inference build/results outputs"

install:
	$(PIP) install -r requirements.txt

train-fun-kan:
	$(PYTHON) src_train/kan_models/models/functions/fun_1_d/main.py --params $(FUN_KAN_PARAMS)

train-fun-mlp:
	$(PYTHON) src_train/kan_models/models/functions/sine_1d_mlp/main.py --params $(FUN_MLP_PARAMS)

compare-fun-predictions:
	$(PYTHON) src_train/kan_models/models/functions/compare_fun_1_d_predictions.py

sync-inference-kan:
	test -f "$(KAN_EXPORT_SRC)"
	mkdir -p "src_inference/model/$(KAN_MODEL)"
	cp "$(KAN_EXPORT_SRC)" "src_inference/model/$(KAN_MODEL)/mini_kan_riscv_export.json"
	@echo "Copied $(KAN_EXPORT_SRC) -> src_inference/model/$(KAN_MODEL)/mini_kan_riscv_export.json"

sync-inference-mlp:
	test -f "$(MLP_EXPORT_SRC)"
	mkdir -p src_inference/model/mlp
	cp "$(MLP_EXPORT_SRC)" "src_inference/model/mlp/mlp_export.json"
	@echo "Copied $(MLP_EXPORT_SRC) -> src_inference/model/mlp/mlp_export.json"

conic:
	$(PYTHON) src_train/kan_models/models/conic/main.py --config $(CONIC_CONFIG)

tabular-credit:
	$(PYTHON) src_train/kan_models/models/tabular/credit_default/main.py --config $(CREDIT_CONFIG)

tabular-stroke:
	$(PYTHON) src_train/kan_models/models/tabular/stroke/main.py --config $(STROKE_CONFIG)

nasa:
	$(PYTHON) src_train/kan_models/models/nasa/main.py --config $(NASA_CONFIG)

nasa-quant:
	$(PYTHON) src_train/kan_models/models/nasa/quantize.py --config $(NASA_CONFIG) --w-bit $(NASA_QUANT_W_BITS) --a-bit $(NASA_QUANT_A_BITS)

inference-host:
	bash src_inference/scripts/build_host.sh

inference-riscv:
	bash src_inference/scripts/build_riscv.sh

inference-mlp-host:
	bash src_inference/scripts/build_mlp_host.sh

inference-mlp-riscv:
	bash src_inference/scripts/build_mlp_riscv.sh

gem5-kan-cache:
	bash src_inference/scripts/run_cache.sh $(KAN_MODEL) $(N)

gem5-kan-l1:
	bash src_inference/scripts/run_l1.sh $(KAN_MODEL) $(N)

gem5-kan-nocache:
	bash src_inference/scripts/run_nocache.sh $(KAN_MODEL) $(N)

gem5-mlp:
	bash src_inference/scripts/run_mlp_l1_l2.sh model/mlp $(N)

gem5-compare-one:
	$(PYTHON) src_inference/scripts/compare_kan_mlp.py \
		--kan-dir src_inference/results/cache/$(KAN_MODEL) \
		--mlp-dir src_inference/results/mlp_l1_l2

gem5-compare-all:
	$(PYTHON) src_inference/scripts/compare_all_models.py

gem5-plots:
	$(PYTHON) src_inference/scripts/plot_model_metrics.py

gem5-all:
	bash src_inference/scripts/run_all_models_compare.sh $(N) model/mlp

clean:
	$(MAKE) -C src_inference clean
