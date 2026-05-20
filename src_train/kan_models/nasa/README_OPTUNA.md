# NASA KAN Optuna tuning

Run hyperparameter search without changing the existing training entrypoint:

```bash
pip install -r requirements-optuna.txt
.venv/bin/python src_train/kan_models/nasa/tune_optuna.py \
  --config src_train/configs/train_config.toml \
  --trials 30 \
  --epochs 25 \
  --patience 5 \
  --tune-raw-preprocessing \
  --retrain-best
```

The search writes only under `artifacts/nasa_optuna` by default.

The default search space includes:

- model hyperparameters: hidden layers, hidden widths, grid size, spline degree, noise scale;
- training hyperparameters: learning rate, weight decay, batch size, loss, Huber beta, grid update schedule;
- train-time preprocessing: `last`, `summary`, `short_flatten`, summary statistics, target scaling, validation split;
- raw preprocessing when `--tune-raw-preprocessing` is enabled: sliding-window size and RUL cap.

The raw scaler and dropped columns are fixed to the existing FD001 preprocessing defaults:

- scaler: `MinMaxScaler(feature_range=(-1, 1))`;
- dropped columns: operating settings plus constant FD001 sensors.

## Editing Search Spaces

Search spaces are defined directly in `tune_optuna.py`.

- Hidden layer count and width choices: edit `suggest_hidden_layers()`.
- Summary statistics choices: edit `suggest_summary_statistics()`.
- Raw preprocessing search: edit `raw_window_size` and `raw_rul_cap` in `suggest_trial_experiment()`.
- Train-time preprocessing modes: edit `allowed_modes` in `suggest_trial_experiment()`.
- Model parameters: edit `model_grid`, `model_k`, and `model_noise_scale` in `suggest_trial_experiment()`.
- Training parameters: edit `training_batch_size`, `training_learning_rate`, `training_weight_decay`, `training_loss`, and grid update values in `suggest_trial_experiment()`.

Use the Optuna helper matching the parameter type:

```python
trial.suggest_categorical("name", [choice_a, choice_b])
trial.suggest_int("name", low, high)
trial.suggest_float("name", low, high, log=True)
```

When using persistent storage with `--storage`, use a new `--study-name` after changing categorical choices. Optuna stores the old distributions and can reject incompatible changes.

Main outputs:

- `trials.csv`: all trial parameters and metrics;
- `best_trial.json`: best Optuna trial with params and metrics;
- `best_config.json`: full resolved config for the best trial;
- `best_model/`: normal training artifacts, only when `--retrain-best` is passed.
