.PHONY: clean distclean data-cleanup cache-cleanup run-all-models

N ?= 1024
MLP_MODEL ?= model/mlp

clean:
	rm -rf build
	$(MAKE) data-cleanup
	$(MAKE) cache-cleanup

data-cleanup:
	rm -rf results simulation_metrics

cache-cleanup:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

run-all-models:
	bash scripts/run_all_models_compare.sh $(N) $(MLP_MODEL)

distclean: clean
	rm -f include/kan_model.h include/kan_model_quant.h include/kan_model_true_int.h include/mlp_model.h include/nasa_test_data.h
