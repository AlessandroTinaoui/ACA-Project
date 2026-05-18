SHELL := /bin/bash

PYTHON ?= python3
PIP ?= pip

NASA_CONFIG ?= src_train/configs/train_config.toml
NASA_QUANT_W_BITS ?= 16
NASA_QUANT_A_BITS ?= 16
NASA_NUM_INPUTS ?= 0

.PHONY: help install nasa nasa-quant nasa-gem5 nasa-quant-gem5 nasa-true-int nasa-suite clean

help:
	@printf "%s\n" \
	"Targets:" \
	"install                  Install Python dependencies from requirements.txt" \
	"nasa                     Run the NASA C-MAPSS RUL KAN regression" \
	"nasa-quant               Apply QuantKAN PTQ to the NASA KAN checkpoint" \
	"nasa-gem5                Build and run NASA fp32 on gem5" \
	"nasa-quant-gem5          Run NASA PTQ on gem5" \
	"nasa-true-int            Run the NASA true-int gem5 path" \
	"nasa-suite               Run fp32, PTQ, and true-int NASA gem5 comparisons" \
	"clean                    Clean src_inference build/results outputs" \
	"" \
	"Options:" \
	"PYTHON=python3           Python interpreter used for training targets" \
	"PIP=pip                  Pip executable used by install" \
	"NASA_CONFIG=$(NASA_CONFIG)" \
	"                         TOML config used by nasa, nasa-quant, and nasa-suite" \
	"NASA_QUANT_W_BITS=$(NASA_QUANT_W_BITS)" \
	"                         Weight bit-width used by nasa-quant and quantized gem5 targets" \
	"NASA_QUANT_A_BITS=$(NASA_QUANT_A_BITS)" \
	"                         Activation bit-width used by nasa-quant" \
	"NASA_NUM_INPUTS=$(NASA_NUM_INPUTS)" \
	"                         Number of test samples for gem5 targets; 0 means full test set" \
	"" \
	"Examples:" \
	"make nasa NASA_CONFIG=src_train/configs/train_config.toml" \
	"make nasa-quant NASA_QUANT_W_BITS=8 NASA_QUANT_A_BITS=8" \
	"make nasa-gem5 NASA_NUM_INPUTS=128" \
	"make nasa-true-int NASA_QUANT_W_BITS=16 NASA_NUM_INPUTS=128" \
	"make nasa-suite NASA_NUM_INPUTS=0"

install:
	$(PIP) install -r requirements.txt

nasa:
	$(PYTHON) src_train/kan_models/nasa/train.py --config $(NASA_CONFIG)

nasa-quant:
	$(PYTHON) src_train/kan_models/nasa/quantize.py --config $(NASA_CONFIG) --w-bit $(NASA_QUANT_W_BITS) --a-bit $(NASA_QUANT_A_BITS)

nasa-gem5:
	bash src_inference/scripts/run/gem5.sh fp32 default $(NASA_NUM_INPUTS)

nasa-quant-gem5:
	bash src_inference/scripts/run/gem5.sh quant $(NASA_QUANT_W_BITS) "" $(NASA_NUM_INPUTS)

nasa-true-int:
	bash src_inference/scripts/run/gem5.sh true-int $(NASA_QUANT_W_BITS) "" $(NASA_NUM_INPUTS)

nasa-suite:
	bash src_inference/scripts/run/suite.sh $(NASA_CONFIG) $(NASA_NUM_INPUTS)

clean:
	rm -rf src_inference/build src_inference/results src_inference/simulation_metrics
	rm -f src_inference/include/kan_model.h src_inference/include/kan_model_quant.h src_inference/include/kan_model_true_int.h src_inference/include/rul_test_data.h src_inference/include/test_data.h
	find src_train src_inference -type d -name "__pycache__" -exec rm -rf {} +
	find src_train src_inference -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
