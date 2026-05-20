"""Optuna hyperparameter tuning for the NASA C-MAPSS KAN regressor.

This script intentionally leaves the existing training and preprocessing
entrypoints unchanged. Each trial builds an in-memory copy of the base config,
optionally writes trial-local preprocessed data, trains a model, and stores the
Optuna results under a separate output directory.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kan_models.nasa.train import (
    DEFAULT_CONFIG_PATH,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
    build_model,
    load_config,
    load_dataset,
    save_artifacts,
    train_model,
)
from kan_models.utils import PROJECT_ROOT, detect_device, write_json


NASA_DIR = PROJECT_ROOT / "datasets" / "NASA"
RAW_DIR = NASA_DIR / "raw" / "CMAPSSData"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "nasa_optuna"
DEFAULT_RAW_SCALER = "minmax_minus_one_one"
DEFAULT_RAW_DROP_PROFILE = "fd001_default"

BASE_COLUMNS = [
    "Engine_ID",
    "Cycle",
    "Op_Setting_1",
    "Op_Setting_2",
    "Op_Setting_3",
]
SENSOR_COLUMNS = [f"Sensor_{idx}" for idx in range(1, 22)]
COLUMNS = BASE_COLUMNS + SENSOR_COLUMNS

FD001_DEFAULT_DROP_COLUMNS = [
    "Op_Setting_1",
    "Op_Setting_2",
    "Op_Setting_3",
    "Sensor_1",
    "Sensor_5",
    "Sensor_16",
    "Sensor_18",
    "Sensor_19",
]
FD001_CONSTANT_SENSOR_COLUMNS = [
    "Sensor_1",
    "Sensor_5",
    "Sensor_16",
    "Sensor_18",
    "Sensor_19",
]
OPERATING_SETTING_COLUMNS = ["Op_Setting_1", "Op_Setting_2", "Op_Setting_3"]


def load_cmapss_file(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, sep=r"\s+", header=None)
    if data.shape[1] != len(COLUMNS):
        raise ValueError(f"{path.name} has {data.shape[1]} columns, expected {len(COLUMNS)}.")
    data.columns = COLUMNS
    return data


def add_train_rul(data: pd.DataFrame, cap: int) -> pd.DataFrame:
    data = data.copy()
    max_cycle = data.groupby("Engine_ID")["Cycle"].transform("max")
    data["RUL"] = (max_cycle - data["Cycle"]).clip(upper=cap)
    return data


def add_test_rul(data: pd.DataFrame, rul_path: Path, cap: int) -> pd.DataFrame:
    data = data.copy()
    final_rul = pd.read_csv(rul_path, sep=r"\s+", header=None).iloc[:, 0]
    engine_ids = sorted(data["Engine_ID"].unique())
    if len(final_rul) != len(engine_ids):
        raise ValueError(
            f"{rul_path.name} has {len(final_rul)} RUL values, "
            f"but the test set contains {len(engine_ids)} engines."
        )
    final_rul_by_engine = dict(zip(engine_ids, final_rul.astype(float)))
    max_cycle_by_engine = data.groupby("Engine_ID")["Cycle"].transform("max")
    data["Final_RUL"] = data["Engine_ID"].map(final_rul_by_engine)
    data["RUL"] = (data["Final_RUL"] + max_cycle_by_engine - data["Cycle"]).clip(upper=cap)
    return data.drop(columns=["Final_RUL"])


def drop_columns_for_profile(profile: str) -> list[str]:
    if profile == "fd001_default":
        return FD001_DEFAULT_DROP_COLUMNS
    if profile == "keep_operating_settings":
        return FD001_CONSTANT_SENSOR_COLUMNS
    if profile == "keep_all_sensors":
        return OPERATING_SETTING_COLUMNS
    raise ValueError(f"Unsupported drop profile: {profile}")


def normalize_features(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    scaler_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_data = train_data.copy()
    test_data = test_data.copy()
    feature_columns = [
        column
        for column in train_data.columns
        if column not in ("Engine_ID", "Cycle", "RUL")
    ]

    if scaler_name == "minmax_minus_one_one":
        scaler = MinMaxScaler(feature_range=(-1, 1))
    elif scaler_name == "minmax_zero_one":
        scaler = MinMaxScaler(feature_range=(0, 1))
    else:
        raise ValueError(f"Unsupported scaler: {scaler_name}")

    train_data[feature_columns] = scaler.fit_transform(train_data[feature_columns])
    test_data[feature_columns] = scaler.transform(test_data[feature_columns])
    return train_data, test_data, feature_columns


def extract_sliding_windows(
    data: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_windows: list[np.ndarray] = []
    y_values: list[float] = []

    for _, engine_data in data.groupby("Engine_ID", sort=True):
        engine_data = engine_data.sort_values("Cycle")
        features = engine_data[feature_columns].to_numpy(dtype=np.float32)
        rul = engine_data["RUL"].to_numpy(dtype=np.float32)
        if len(engine_data) < window_size:
            continue
        for start_idx in range(0, len(engine_data) - window_size + 1):
            end_idx = start_idx + window_size
            x_windows.append(features[start_idx:end_idx])
            y_values.append(rul[end_idx - 1])

    if not x_windows:
        raise ValueError(f"No windows were generated with window_size={window_size}.")

    return np.asarray(x_windows, dtype=np.float32), np.asarray(y_values, dtype=np.float32)


def flattened_window_columns(feature_columns: list[str], window_size: int) -> list[str]:
    return [
        f"t{time_idx:02d}_{feature}"
        for time_idx in range(window_size)
        for feature in feature_columns
    ]


def save_preprocessed_outputs(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_columns: list[str],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "x_train": output_dir / "X_train.npy",
        "y_train": output_dir / "Y_train.npy",
        "x_test": output_dir / "X_test.npy",
        "y_test": output_dir / "Y_test.npy",
        "train_csv": output_dir / "train_windows.csv",
        "test_csv": output_dir / "test_windows.csv",
    }

    np.save(paths["x_train"], x_train)
    np.save(paths["y_train"], y_train)
    np.save(paths["x_test"], x_test)
    np.save(paths["y_test"], y_test)

    window_columns = flattened_window_columns(feature_columns, x_train.shape[1])
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    train_windows = pd.DataFrame(x_train_flat, columns=window_columns)
    train_windows["RUL"] = y_train
    test_windows = pd.DataFrame(x_test_flat, columns=window_columns)
    test_windows["RUL"] = y_test
    train_windows.to_csv(paths["train_csv"], index=False)
    test_windows.to_csv(paths["test_csv"], index=False)
    return paths


def preprocess_raw_trial(
    raw_dir: Path,
    output_dir: Path,
    *,
    window_size: int,
    rul_cap: int,
    scaler_name: str,
    drop_profile: str,
) -> dict[str, Path]:
    train_path = raw_dir / "train_FD001.txt"
    test_path = raw_dir / "test_FD001.txt"
    rul_path = raw_dir / "RUL_FD001.txt"

    train_data = add_train_rul(load_cmapss_file(train_path), cap=rul_cap)
    test_data = add_test_rul(load_cmapss_file(test_path), rul_path, cap=rul_cap)

    drop_columns = drop_columns_for_profile(drop_profile)
    train_data = train_data.drop(columns=drop_columns)
    test_data = test_data.drop(columns=drop_columns)
    train_data, test_data, feature_columns = normalize_features(train_data, test_data, scaler_name)
    x_train, y_train = extract_sliding_windows(train_data, feature_columns, window_size)
    x_test, y_test = extract_sliding_windows(test_data, feature_columns, window_size)
    return save_preprocessed_outputs(x_train, y_train, x_test, y_test, feature_columns, output_dir)


def monitor_direction(monitor: str) -> str:
    return "maximize" if monitor.endswith("r2") or monitor.endswith("accuracy_within_tolerance") else "minimize"


def monitor_value(metrics: dict[str, Any], monitor: str) -> float:
    split_name, metric_name = monitor.split("_", 1) if "_" in monitor else ("val", monitor)
    return float(metrics[split_name][metric_name])


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def make_raw_config(
    base: ExperimentConfig,
    data: DataConfig,
    model: ModelConfig,
    training: TrainingConfig,
    output: OutputConfig,
) -> dict[str, Any]:
    raw_config = copy.deepcopy(base.raw_config)
    raw_config["data"] = {
        **raw_config.get("data", {}),
        "train_csv_path": str(data.train_csv_path),
        "test_csv_path": str(data.test_csv_path),
        "target_column": data.target_column,
        "window_mode": data.window_mode,
        "window_size": data.window_size,
        "summary_statistics": list(data.summary_statistics),
        "x_train_npy_path": str(data.x_train_npy_path) if data.x_train_npy_path else "",
        "y_train_npy_path": str(data.y_train_npy_path) if data.y_train_npy_path else "",
        "x_test_npy_path": str(data.x_test_npy_path) if data.x_test_npy_path else "",
        "y_test_npy_path": str(data.y_test_npy_path) if data.y_test_npy_path else "",
        "target_scale": data.target_scale,
        "validation_size": data.validation_size,
        "random_seed": data.random_seed,
        "shuffle": data.shuffle,
        "max_train_samples": data.max_train_samples or 0,
        "max_test_samples": data.max_test_samples or 0,
    }
    raw_config["model"] = {
        **raw_config.get("model", {}),
        "hidden_layers": list(model.hidden_layers),
        "grid": model.grid,
        "k": model.k,
        "noise_scale": model.noise_scale,
        "symbolic_enabled": model.symbolic_enabled,
        "auto_save": model.auto_save,
        "device": model.device,
    }
    raw_config["training"] = {
        **raw_config.get("training", {}),
        "epochs": training.epochs,
        "batch_size": training.batch_size,
        "eval_batch_size": training.eval_batch_size,
        "learning_rate": training.learning_rate,
        "weight_decay": training.weight_decay,
        "grid_update_epochs": training.grid_update_epochs,
        "grid_update_every": training.grid_update_every,
        "patience": training.patience,
        "min_delta": training.min_delta,
        "log_every": training.log_every,
        "optimizer": training.optimizer,
        "loss": training.loss,
        "huber_beta": training.huber_beta,
        "monitor": training.monitor,
    }
    raw_config["output"] = {
        **raw_config.get("output", {}),
        "output_dir": str(output.output_dir),
        "model_filename": output.model_filename,
        "export_json_filename": output.export_json_filename,
        "metrics_filename": output.metrics_filename,
        "history_filename": output.history_filename,
        "predictions_filename": output.predictions_filename,
        "config_snapshot_filename": output.config_snapshot_filename,
        "plot_filename": output.plot_filename,
    }
    return raw_config


def suggest_summary_statistics(trial: Any) -> list[str]:
    summary_statistics = [
        statistic
        for statistic in ("mean", "var", "trend")
        if trial.suggest_categorical(f"data_summary_{statistic}", [True, False])
    ]
    if summary_statistics:
        return summary_statistics
    return [trial.suggest_categorical("data_summary_fallback", ["mean", "var", "trend"])]


def suggest_hidden_layers(trial: Any) -> list[int]:
    layer_count = trial.suggest_int("model_hidden_layer_count", 1, 3)
    hidden_layers: list[int] = []
    for layer_idx in range(layer_count):
        hidden_layers.append(
            trial.suggest_categorical(
                f"model_hidden_{layer_idx}",
                [4, 8, 12, 16, 24, 32, 48, 64],
            )
        )
    return hidden_layers


def suggest_trial_experiment(
    trial: Any,
    base: ExperimentConfig,
    args: argparse.Namespace,
    *,
    trial_label: str,
    seed_offset: int | None = None,
) -> tuple[ExperimentConfig, dict[str, Any]]:
    metadata: dict[str, Any] = {}

    data = base.data
    if args.tune_raw_preprocessing:
        raw_window_size = trial.suggest_categorical("raw_window_size", [10, 15, 20, 30, 40])
        raw_rul_cap = trial.suggest_categorical("raw_rul_cap", [100, 125, 150, 200])
        processed_dir = args.output_dir / "trial_preprocessing" / trial_label
        paths = preprocess_raw_trial(
            args.raw_dir,
            processed_dir,
            window_size=raw_window_size,
            rul_cap=raw_rul_cap,
            scaler_name=DEFAULT_RAW_SCALER,
            drop_profile=DEFAULT_RAW_DROP_PROFILE,
        )
        metadata["raw_preprocessing"] = {
            "window_size": raw_window_size,
            "rul_cap": raw_rul_cap,
            "scaler": DEFAULT_RAW_SCALER,
            "drop_profile": DEFAULT_RAW_DROP_PROFILE,
            "processed_dir": processed_dir,
        }
        data = replace(
            data,
            train_csv_path=paths["train_csv"],
            test_csv_path=paths["test_csv"],
            x_train_npy_path=paths["x_train"],
            y_train_npy_path=paths["y_train"],
            x_test_npy_path=paths["x_test"],
            y_test_npy_path=paths["y_test"],
        )

    allowed_modes = ["last", "summary", "short_flatten"]
    if args.allow_full_flatten:
        allowed_modes.append("flatten")
    window_mode = trial.suggest_categorical("data_window_mode", allowed_modes)
    window_size = data.window_size
    summary_statistics = list(data.summary_statistics)
    if window_mode == "summary":
        summary_statistics = suggest_summary_statistics(trial)
    elif window_mode == "short_flatten":
        max_short_window = 30
        if args.tune_raw_preprocessing:
            max_short_window = int(metadata["raw_preprocessing"]["window_size"])
        window_size = trial.suggest_int("data_short_window_size", 3, max_short_window)

    data = replace(
        data,
        window_mode=window_mode,
        window_size=window_size,
        summary_statistics=summary_statistics,
        target_scale=trial.suggest_float("data_target_scale", 80.0, 180.0),
        validation_size=trial.suggest_float("data_validation_size", 0.10, 0.25),
        random_seed=args.seed + (trial.number if seed_offset is None else seed_offset),
    )

    model = replace(
        base.model,
        hidden_layers=suggest_hidden_layers(trial),
        grid=trial.suggest_int("model_grid", 3, 8),
        k=trial.suggest_int("model_k", 2, 4),
        noise_scale=trial.suggest_float("model_noise_scale", 0.0, 0.12),
        device=args.device or base.model.device,
    )

    batch_size = trial.suggest_categorical("training_batch_size", [32, 64, 128, 256])
    loss = trial.suggest_categorical("training_loss", ["mse", "huber"])
    training = replace(
        base.training,
        epochs=args.epochs or base.training.epochs,
        batch_size=batch_size,
        eval_batch_size=max(base.training.eval_batch_size, batch_size * 2),
        learning_rate=trial.suggest_float("training_learning_rate", 1e-4, 8e-3, log=True),
        weight_decay=trial.suggest_float("training_weight_decay", 1e-7, 1e-3, log=True),
        grid_update_epochs=trial.suggest_categorical("training_grid_update_epochs", [0, 1, 3, 5]),
        grid_update_every=trial.suggest_categorical("training_grid_update_every", [10, 25, 50]),
        patience=args.patience or base.training.patience,
        loss=loss,
        huber_beta=trial.suggest_float("training_huber_beta", 0.01, 0.25) if loss == "huber" else base.training.huber_beta,
        log_every=max(1, args.log_every),
    )

    output = replace(
        base.output,
        output_dir=args.output_dir / "best_model",
        config_snapshot_filename="config.json",
    )
    raw_config = make_raw_config(base, data, model, training, output)
    experiment = ExperimentConfig(
        data=data,
        model=model,
        training=training,
        evaluation=base.evaluation,
        output=output,
        raw_config=raw_config,
        config_path=base.config_path,
    )
    metadata["config"] = raw_config
    return experiment, metadata


def run_trial(trial: Any, base: ExperimentConfig, args: argparse.Namespace) -> float:
    experiment, metadata = suggest_trial_experiment(
        trial,
        base,
        args,
        trial_label=f"trial_{trial.number:04d}",
    )
    trial.set_user_attr("config", json_safe(metadata["config"]))
    if "raw_preprocessing" in metadata:
        trial.set_user_attr("raw_preprocessing", json_safe(metadata["raw_preprocessing"]))

    dataset = load_dataset(experiment.data)
    device = detect_device(experiment.model.device)
    model = build_model(
        input_dim=dataset.input_dim,
        model_config=experiment.model,
        seed=experiment.data.random_seed,
        device=device,
    )

    stdout_context = contextlib.nullcontext()
    if not args.verbose_trials:
        stdout_context = contextlib.redirect_stdout(io.StringIO())

    with stdout_context:
        _, metrics, _, _ = train_model(model, dataset, experiment, device)

    value = monitor_value(metrics, experiment.training.monitor)
    trial.set_user_attr("input_dim", dataset.input_dim)
    trial.set_user_attr("best_epoch", metrics["best_epoch"])
    trial.set_user_attr("trained_epochs", metrics["trained_epochs"])
    trial.set_user_attr("train_metrics", metrics["train"])
    trial.set_user_attr("val_metrics", metrics["val"])
    trial.set_user_attr("test_metrics", metrics["test"])
    trial.set_user_attr("monitor", experiment.training.monitor)

    if not math.isfinite(value):
        raise ValueError(f"Objective produced a non-finite value: {value}")
    return value


def save_study_outputs(study: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_csv_path = output_dir / "trials.csv"
    study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs")).to_csv(
        trials_csv_path,
        index=False,
    )
    best = study.best_trial
    write_json(
        output_dir / "best_trial.json",
        {
            "number": best.number,
            "value": best.value,
            "params": best.params,
            "user_attrs": json_safe(best.user_attrs),
        },
    )
    best_config = best.user_attrs.get("config")
    if best_config is not None:
        write_json(output_dir / "best_config.json", json_safe(best_config))


def retrain_best(study: Any, base: ExperimentConfig, args: argparse.Namespace, optuna_module: Any) -> None:
    fixed_trial = optuna_module.trial.FixedTrial(study.best_trial.params)
    experiment, metadata = suggest_trial_experiment(
        fixed_trial,
        base,
        args,
        trial_label="best",
        seed_offset=study.best_trial.number,
    )
    best_config_path = args.output_dir / "best_config.json"
    experiment = replace(
        experiment,
        config_path=best_config_path,
        raw_config=metadata["config"],
        output=replace(experiment.output, output_dir=args.output_dir / "best_model"),
    )

    dataset = load_dataset(experiment.data)
    device = detect_device(experiment.model.device)
    model = build_model(
        input_dim=dataset.input_dim,
        model_config=experiment.model,
        seed=experiment.data.random_seed,
        device=device,
    )
    model, metrics, history, test_predictions = train_model(model, dataset, experiment, device)
    save_artifacts(experiment, dataset, model, metrics, history, test_predictions)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune NASA KAN hyperparameters with Optuna.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Base training TOML config.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for tuning outputs.")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials.")
    parser.add_argument("--timeout", type=int, default=None, help="Optional study timeout in seconds.")
    parser.add_argument("--study-name", default="nasa-kan-optuna", help="Optuna study name.")
    parser.add_argument("--storage", default=None, help="Optional Optuna storage URL, e.g. sqlite:///study.db.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used by the sampler and trial configs.")
    parser.add_argument("--epochs", type=int, default=None, help="Override trial epochs.")
    parser.add_argument("--patience", type=int, default=None, help="Override trial early-stopping patience.")
    parser.add_argument("--device", default=None, help="Override model device, e.g. cpu, cuda, or auto.")
    parser.add_argument("--log-every", type=int, default=9999, help="Training log frequency inside each trial.")
    parser.add_argument(
        "--tune-raw-preprocessing",
        action="store_true",
        help="Tune raw FD001 preprocessing too: window size and RUL cap.",
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Directory containing raw FD001 files.")
    parser.add_argument(
        "--allow-full-flatten",
        action="store_true",
        help="Allow full flattened windows in the train-time preprocessing search space.",
    )
    parser.add_argument(
        "--retrain-best",
        action="store_true",
        help="Train the best configuration once more and save normal training artifacts.",
    )
    parser.add_argument(
        "--verbose-trials",
        action="store_true",
        help="Keep per-epoch training logs visible during Optuna trials.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.raw_dir = args.raw_dir.resolve()

    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing optional dependency: optuna. Install it with "
            "`pip install optuna` or `pip install -r requirements-optuna.txt`."
        ) from exc

    if args.trials <= 0:
        raise ValueError("--trials must be positive.")
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs must be positive when provided.")
    if args.patience is not None and args.patience <= 0:
        raise ValueError("--patience must be positive when provided.")
    if args.tune_raw_preprocessing and not args.raw_dir.exists():
        raise FileNotFoundError(f"Raw NASA directory not found: {args.raw_dir}")

    base = load_config(args.config)
    direction = monitor_direction(base.training.monitor)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=min(5, args.trials))
    study = optuna.create_study(
        study_name=args.study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )
    study.optimize(
        lambda trial: run_trial(trial, base, args),
        n_trials=args.trials,
        timeout=args.timeout,
        gc_after_trial=True,
        catch=(ValueError, RuntimeError),
    )

    save_study_outputs(study, args.output_dir)
    if args.retrain_best:
        retrain_best(study, base, args, optuna)

    print("\nOptuna tuning completed")
    print(f"Study: {args.study_name}")
    print(f"Direction: {direction}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value ({base.training.monitor}): {study.best_value:.6f}")
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
