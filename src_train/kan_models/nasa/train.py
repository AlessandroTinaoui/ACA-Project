"""Compact NASA C-MAPSS KAN training pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kan_models.utils import (
    CONFIGS_DIR,
    KAN,
    clone_state_dict,
    configure_matplotlib,
    detect_device,
    load_toml,
    resolve_path,
    serialize_width,
    write_json,
)

configure_matplotlib()

import matplotlib.pyplot as plt


DEFAULT_CONFIG_PATH = CONFIGS_DIR / "train_config.toml"


@dataclass
class DataConfig:
    train_csv_path: Path
    test_csv_path: Path
    target_column: str = "RUL"
    window_mode: str = "last"
    window_size: int = 5
    summary_statistics: list[str] = field(default_factory=lambda: ["mean", "var", "trend"])
    x_train_npy_path: Path | None = None
    y_train_npy_path: Path | None = None
    x_test_npy_path: Path | None = None
    y_test_npy_path: Path | None = None
    target_scale: float = 125.0
    validation_size: float = 0.15
    random_seed: int = 42
    shuffle: bool = True
    max_train_samples: int | None = None
    max_test_samples: int | None = None


@dataclass
class ModelConfig:
    hidden_layers: list[int]
    grid: int
    k: int
    noise_scale: float = 0.05
    symbolic_enabled: bool = False
    auto_save: bool = False
    device: str = "cpu"


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    grid_update_epochs: int
    grid_update_every: int
    patience: int
    min_delta: float
    log_every: int
    optimizer: str = "adam"
    loss: str = "mse"
    huber_beta: float = 0.05
    monitor: str = "val_mae"


@dataclass
class EvaluationConfig:
    rul_tolerance: float = 10.0


@dataclass
class OutputConfig:
    output_dir: Path
    model_filename: str = "model.pt"
    export_json_filename: str = "nasa_kan_riscv_export.json"
    metrics_filename: str = "metrics.json"
    history_filename: str = "history.csv"
    predictions_filename: str = "test_predictions.csv"
    config_snapshot_filename: str = "config.toml"
    plot_filename: str = "training_history.png"


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    output: OutputConfig
    raw_config: dict[str, Any]
    config_path: Path


@dataclass
class NasaDataset:
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    feature_names: list[str]
    target_scale: float
    source: str

    @property
    def input_dim(self) -> int:
        return int(self.X_train.shape[1])


def _optional_path(config_dir: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    return resolve_path(config_dir, value)


def _optional_positive_int(value: object) -> int | None:
    if value in (None, 0, "0", ""):
        return None
    result = int(value)
    if result <= 0:
        raise ValueError(f"Expected a positive integer or 0, got {value!r}.")
    return result


def _required_section(raw_config: dict[str, Any], name: str) -> dict[str, Any]:
    section = raw_config.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing [{name}] section in NASA config.")
    return section


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> ExperimentConfig:
    """Load a NASA C-MAPSS regression config from TOML."""
    resolved_path, raw_config = load_toml(config_path)
    config_dir = resolved_path.parent

    data_section = _required_section(raw_config, "data")
    model_section = _required_section(raw_config, "model")
    training_section = _required_section(raw_config, "training")
    evaluation_section = raw_config.get("evaluation", {})
    output_section = _required_section(raw_config, "output")

    data = DataConfig(
        train_csv_path=resolve_path(config_dir, data_section["train_csv_path"]),
        test_csv_path=resolve_path(config_dir, data_section["test_csv_path"]),
        target_column=data_section.get("target_column", "RUL"),
        window_mode=data_section.get("window_mode", "last"),
        window_size=int(data_section.get("window_size", 5)),
        summary_statistics=data_section.get("summary_statistics", ["mean", "var", "trend"]),
        x_train_npy_path=_optional_path(config_dir, data_section.get("x_train_npy_path")),
        y_train_npy_path=_optional_path(config_dir, data_section.get("y_train_npy_path")),
        x_test_npy_path=_optional_path(config_dir, data_section.get("x_test_npy_path")),
        y_test_npy_path=_optional_path(config_dir, data_section.get("y_test_npy_path")),
        target_scale=float(data_section.get("target_scale", 125.0)),
        validation_size=float(data_section.get("validation_size", 0.15)),
        random_seed=int(data_section.get("random_seed", 42)),
        shuffle=bool(data_section.get("shuffle", True)),
        max_train_samples=_optional_positive_int(data_section.get("max_train_samples")),
        max_test_samples=_optional_positive_int(data_section.get("max_test_samples")),
    )
    if data.target_scale <= 0:
        raise ValueError("data.target_scale must be positive.")
    if not 0.0 < data.validation_size < 1.0:
        raise ValueError("data.validation_size must be between 0 and 1.")
    if data.window_size <= 0:
        raise ValueError("data.window_size must be positive.")
    if data.window_mode not in {"last", "flatten", "summary", "short_flatten"}:
        raise ValueError("data.window_mode must be one of 'last', 'flatten', 'summary', or 'short_flatten'.")

    allowed_summary_statistics = {"mean", "var", "trend"}
    unknown_summary_statistics = set(data.summary_statistics).difference(allowed_summary_statistics)
    if unknown_summary_statistics:
        raise ValueError(f"Unsupported summary statistics: {sorted(unknown_summary_statistics)}.")

    model = ModelConfig(**model_section)
    if not model.hidden_layers or any(hidden <= 0 for hidden in model.hidden_layers):
        raise ValueError("model.hidden_layers must contain positive integers.")

    training = TrainingConfig(**training_section)
    if training.epochs <= 0 or training.batch_size <= 0 or training.eval_batch_size <= 0:
        raise ValueError("training.epochs, training.batch_size and training.eval_batch_size must be positive.")
    if training.grid_update_every <= 0:
        raise ValueError("training.grid_update_every must be positive.")

    return ExperimentConfig(
        data=data,
        model=model,
        training=training,
        evaluation=EvaluationConfig(**evaluation_section),
        output=OutputConfig(
            output_dir=resolve_path(config_dir, output_section["output_dir"]),
            model_filename=output_section.get("model_filename", "model.pt"),
            export_json_filename=output_section.get("export_json_filename", "nasa_kan_riscv_export.json"),
            metrics_filename=output_section.get("metrics_filename", "metrics.json"),
            history_filename=output_section.get("history_filename", "history.csv"),
            predictions_filename=output_section.get("predictions_filename", "test_predictions.csv"),
            config_snapshot_filename=output_section.get("config_snapshot_filename", "config.toml"),
            plot_filename=output_section.get("plot_filename", "training_history.png"),
        ),
        raw_config=raw_config,
        config_path=resolved_path,
    )


def _limit_samples(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    rng: np.random.Generator,
    shuffle: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or max_samples >= len(X):
        return X, y

    indices = np.arange(len(X))
    if shuffle:
        rng.shuffle(indices)
    indices = indices[:max_samples]
    return X[indices], y[indices]


def _last_timestep_columns(feature_names: list[str]) -> list[str]:
    timestep_by_column: dict[str, int] = {}
    for column in feature_names:
        prefix, _, _ = column.partition("_")
        if len(prefix) == 3 and prefix.startswith("t") and prefix[1:].isdigit():
            timestep_by_column[column] = int(prefix[1:])

    if not timestep_by_column:
        raise ValueError("window_mode='last' requires flattened columns named like t00_Sensor_2.")

    last_timestep = max(timestep_by_column.values())
    return [column for column in feature_names if timestep_by_column.get(column) == last_timestep]


def _parse_window_layout(feature_names: list[str]) -> tuple[list[int], list[str], list[str]]:
    parsed: dict[tuple[int, str], str] = {}
    sensor_order_by_timestep: dict[int, list[str]] = {}

    for column in feature_names:
        prefix, _, sensor = column.partition("_")
        if len(prefix) != 3 or not prefix.startswith("t") or not prefix[1:].isdigit() or not sensor:
            continue
        timestep = int(prefix[1:])
        parsed[(timestep, sensor)] = column
        sensor_order_by_timestep.setdefault(timestep, []).append(sensor)

    if not parsed:
        raise ValueError("Window features must be named like t00_Sensor_2.")

    timesteps = sorted({timestep for timestep, _ in parsed})
    first_timestep = timesteps[0]
    sensor_names = sensor_order_by_timestep[first_timestep]
    ordered_columns: list[str] = []

    for timestep in timesteps:
        for sensor in sensor_names:
            column = parsed.get((timestep, sensor))
            if column is None:
                raise ValueError(f"Missing window column for timestep t{timestep:02d} and sensor {sensor}.")
            ordered_columns.append(column)

    return timesteps, sensor_names, ordered_columns


def _short_window_columns(feature_names: list[str], window_size: int) -> list[str]:
    timesteps, sensor_names, _ = _parse_window_layout(feature_names)
    selected_timesteps = timesteps[-min(window_size, len(timesteps)) :]
    return [f"t{timestep:02d}_{sensor}" for timestep in selected_timesteps for sensor in sensor_names]


def _select_feature_columns(path: Path, target_column: str, window_mode: str, window_size: int) -> list[str]:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    if target_column not in columns:
        raise ValueError(f"Target column '{target_column}' not found in {path}.")

    feature_names = [column for column in columns if column != target_column]
    if window_mode in {"flatten", "summary"}:
        return feature_names
    if window_mode == "short_flatten":
        return _short_window_columns(feature_names, window_size)
    return _last_timestep_columns(feature_names)


def _window_matrix(frame: pd.DataFrame, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    timesteps, sensor_names, ordered_columns = _parse_window_layout(feature_names)
    values = frame[ordered_columns].to_numpy(dtype=np.float32)
    return values.reshape(len(frame), len(timesteps), len(sensor_names)), sensor_names


def _trend(values: np.ndarray) -> np.ndarray:
    timesteps = np.arange(values.shape[1], dtype=np.float32)
    centered = timesteps - timesteps.mean()
    denominator = np.sum(centered**2)
    if denominator == 0:
        return np.zeros((values.shape[0], values.shape[2]), dtype=np.float32)
    return np.sum(values * centered.reshape(1, -1, 1), axis=1) / denominator


def _summarize_windows(
    values: np.ndarray,
    sensor_names: list[str],
    summary_statistics: list[str],
) -> tuple[np.ndarray, list[str]]:
    arrays: list[np.ndarray] = []
    feature_names: list[str] = []

    for statistic in summary_statistics:
        if statistic == "mean":
            arrays.append(values.mean(axis=1))
        elif statistic == "var":
            arrays.append(values.var(axis=1))
        elif statistic == "trend":
            arrays.append(_trend(values))
        else:
            raise ValueError(f"Unsupported summary statistic: {statistic}")
        feature_names.extend([f"{sensor}_{statistic}" for sensor in sensor_names])

    return np.concatenate(arrays, axis=1).astype(np.float32), feature_names


def _load_csv(
    path: Path,
    target_column: str,
    window_mode: str,
    window_size: int,
    summary_statistics: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_names = _select_feature_columns(path, target_column, window_mode, window_size)
    frame = pd.read_csv(path, usecols=[*feature_names, target_column])
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' not found in {path}.")

    if window_mode == "summary":
        window_values, sensor_names = _window_matrix(frame, feature_names)
        X, feature_names = _summarize_windows(window_values, sensor_names, summary_statistics)
    else:
        X = frame[feature_names].to_numpy(dtype=np.float32)
    y = frame[target_column].to_numpy(dtype=np.float32)
    return X, y, feature_names


def _load_npy(
    x_path: Path,
    y_path: Path,
    window_mode: str,
    window_size: int,
    summary_statistics: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.float32).reshape(-1)

    if X.ndim == 3:
        _, actual_window_size, feature_count = X.shape
        if window_mode == "flatten":
            feature_names = [
                f"t{time_idx:02d}_feature_{feature_idx:02d}"
                for time_idx in range(actual_window_size)
                for feature_idx in range(feature_count)
            ]
            X = X.reshape(X.shape[0], -1)
        elif window_mode == "short_flatten":
            selected_window_size = min(window_size, actual_window_size)
            start_timestep = actual_window_size - selected_window_size
            feature_names = [
                f"t{time_idx:02d}_feature_{feature_idx:02d}"
                for time_idx in range(start_timestep, actual_window_size)
                for feature_idx in range(feature_count)
            ]
            X = X[:, -selected_window_size:, :].reshape(X.shape[0], -1)
        elif window_mode == "summary":
            sensor_names = [f"feature_{feature_idx:02d}" for feature_idx in range(feature_count)]
            X, feature_names = _summarize_windows(X, sensor_names, summary_statistics)
        else:
            feature_names = [f"t{actual_window_size - 1:02d}_feature_{feature_idx:02d}" for feature_idx in range(feature_count)]
            X = X[:, -1, :]
    elif X.ndim == 2:
        feature_names = [f"feature_{idx:03d}" for idx in range(X.shape[1])]
    else:
        raise ValueError(f"Unsupported NumPy feature shape {X.shape} in {x_path}.")

    if len(X) != len(y):
        raise ValueError(f"Feature/target length mismatch: {x_path} has {len(X)}, {y_path} has {len(y)}.")
    return X, y, feature_names


def _load_split(
    csv_path: Path,
    target_column: str,
    window_mode: str,
    window_size: int,
    summary_statistics: list[str],
    x_npy_path: Path | None,
    y_npy_path: Path | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if x_npy_path is not None and y_npy_path is not None and x_npy_path.exists() and y_npy_path.exists():
        return _load_npy(x_npy_path, y_npy_path, window_mode, window_size, summary_statistics)
    if csv_path.exists():
        return _load_csv(csv_path, target_column, window_mode, window_size, summary_statistics)
    raise FileNotFoundError(
        "Could not find a usable NASA split. Expected either "
        f"both {x_npy_path} and {y_npy_path}, or {csv_path}."
    )


def _train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    validation_size: float,
    rng: np.random.Generator,
    shuffle: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(X))
    if shuffle:
        rng.shuffle(indices)

    val_count = max(1, int(round(len(indices) * validation_size)))
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    if len(train_indices) == 0:
        raise ValueError("Validation split consumed all samples; reduce data.validation_size.")

    return X[train_indices], X[val_indices], y[train_indices], y[val_indices]


def load_dataset(config: DataConfig) -> NasaDataset:
    """Load processed NASA windows and return scaled torch tensors."""
    rng = np.random.default_rng(config.random_seed)
    X_source, y_source, feature_names = _load_split(
        config.train_csv_path,
        config.target_column,
        config.window_mode,
        config.window_size,
        config.summary_statistics,
        config.x_train_npy_path,
        config.y_train_npy_path,
    )
    X_test, y_test, test_feature_names = _load_split(
        config.test_csv_path,
        config.target_column,
        config.window_mode,
        config.window_size,
        config.summary_statistics,
        config.x_test_npy_path,
        config.y_test_npy_path,
    )

    if len(feature_names) != len(test_feature_names):
        raise ValueError(
            "Train/test feature dimension mismatch: "
            f"{len(feature_names)} train features vs {len(test_feature_names)} test features."
        )

    X_source, y_source = _limit_samples(X_source, y_source, config.max_train_samples, rng, config.shuffle)
    X_test, y_test = _limit_samples(X_test, y_test, config.max_test_samples, rng, config.shuffle)
    X_train, X_val, y_train, y_val = _train_val_split(
        X_source,
        y_source,
        config.validation_size,
        rng,
        config.shuffle,
    )

    scale = float(config.target_scale)
    return NasaDataset(
        X_train=torch.tensor(X_train, dtype=torch.float32),
        y_train=torch.tensor(y_train.reshape(-1, 1) / scale, dtype=torch.float32),
        X_val=torch.tensor(X_val, dtype=torch.float32),
        y_val=torch.tensor(y_val.reshape(-1, 1) / scale, dtype=torch.float32),
        X_test=torch.tensor(X_test, dtype=torch.float32),
        y_test=torch.tensor(y_test.reshape(-1, 1) / scale, dtype=torch.float32),
        feature_names=feature_names,
        target_scale=scale,
        source="npy" if config.x_train_npy_path and config.x_train_npy_path.exists() else "csv",
    )


def build_model(input_dim: int, model_config: ModelConfig, seed: int, device: torch.device) -> KAN:
    """Build the configured KAN regressor."""
    width = [input_dim, *model_config.hidden_layers, 1]
    return KAN(
        width=width,
        grid=model_config.grid,
        k=model_config.k,
        noise_scale=model_config.noise_scale,
        seed=seed,
        symbolic_enabled=model_config.symbolic_enabled,
        auto_save=model_config.auto_save,
        device=device,
    )


def build_optimizer(model: KAN, config: TrainingConfig) -> torch.optim.Optimizer:
    optimizer_name = config.optimizer.lower()
    if optimizer_name != "adam":
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def compute_loss(prediction: torch.Tensor, target: torch.Tensor, config: TrainingConfig) -> torch.Tensor:
    loss_name = config.loss.lower()
    if loss_name == "mse":
        return F.mse_loss(prediction, target)
    if loss_name in {"smooth_l1", "huber"}:
        return F.smooth_l1_loss(prediction, target, beta=config.huber_beta)
    raise ValueError(f"Unsupported regression loss: {config.loss}")


def batch_iterator(size: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    indices = np.arange(size, dtype=np.int64)
    rng.shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, size, batch_size)]


@torch.no_grad()
def predict(model: KAN, features: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model(features)


@torch.no_grad()
def predict_batched(model: KAN, features: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Predict without materializing KAN edge activations for the whole split."""
    model.eval()
    predictions: list[torch.Tensor] = []
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size]
        predictions.append(model(batch).detach().cpu())
    return torch.cat(predictions, dim=0)


def regression_metrics(
    y_true_scaled: torch.Tensor,
    y_pred_scaled: torch.Tensor,
    target_scale: float,
    tolerance: float,
) -> dict[str, float]:
    y_true = y_true_scaled.detach().cpu().numpy().reshape(-1) * target_scale
    y_pred = y_pred_scaled.detach().cpu().numpy().reshape(-1) * target_scale
    errors = y_pred - y_true
    mse = float(np.mean(errors**2))
    mae = float(np.mean(np.abs(errors)))
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    residual = float(np.sum(errors**2))
    r2 = float(1.0 - residual / total) if total > 0 else float("nan")
    return {
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": mae,
        "r2": r2,
        "accuracy_within_tolerance": float(np.mean(np.abs(errors) <= tolerance)),
    }


@torch.no_grad()
def evaluate_split(
    model: KAN,
    features: torch.Tensor,
    targets: torch.Tensor,
    experiment: ExperimentConfig,
) -> dict[str, float]:
    prediction = predict_batched(model, features, experiment.training.eval_batch_size)
    targets_cpu = targets.detach().cpu()
    metrics = regression_metrics(
        y_true_scaled=targets_cpu,
        y_pred_scaled=prediction,
        target_scale=experiment.data.target_scale,
        tolerance=experiment.evaluation.rul_tolerance,
    )
    metrics["loss_scaled"] = float(compute_loss(prediction, targets_cpu, experiment.training).detach().cpu())
    return metrics


def _monitor_value(metrics_by_split: dict[str, dict[str, float]], monitor: str) -> float:
    split_name, metric_name = monitor.split("_", 1) if "_" in monitor else ("val", monitor)
    if split_name not in metrics_by_split or metric_name not in metrics_by_split[split_name]:
        raise ValueError(f"Monitor '{monitor}' was not found in metrics.")
    return float(metrics_by_split[split_name][metric_name])


def _monitor_is_better(current: float, best: float, monitor: str, min_delta: float) -> bool:
    maximize = monitor.endswith("r2") or monitor.endswith("accuracy_within_tolerance")
    if maximize:
        return current > best + min_delta
    return current < best - min_delta


def _initial_monitor_value(monitor: str) -> float:
    maximize = monitor.endswith("r2") or monitor.endswith("accuracy_within_tolerance")
    return -float("inf") if maximize else float("inf")


def train_model(
    model: KAN,
    dataset: NasaDataset,
    experiment: ExperimentConfig,
    device: torch.device,
) -> tuple[KAN, dict[str, Any], list[dict[str, float]], np.ndarray]:
    """Train a standard KAN regressor with minibatches and early stopping."""
    training = experiment.training
    seed = experiment.data.random_seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_train = dataset.X_train.to(device)
    y_train = dataset.y_train.to(device)
    X_val = dataset.X_val.to(device)
    y_val = dataset.y_val.to(device)
    X_test = dataset.X_test.to(device)
    y_test = dataset.y_test.to(device)

    optimizer = build_optimizer(model, training)
    rng = np.random.default_rng(seed)
    best_state = clone_state_dict(model)
    best_monitor = _initial_monitor_value(training.monitor)
    best_epoch = 0
    patience_counter = 0
    stopped_early = False
    history: list[dict[str, float]] = []

    for epoch in range(1, training.epochs + 1):
        model.train()
        epoch_losses: list[float] = []

        for batch_number, batch_indices in enumerate(batch_iterator(len(X_train), training.batch_size, rng), start=1):
            batch_X = X_train[batch_indices]
            batch_y = y_train[batch_indices]

            if epoch <= training.grid_update_epochs and batch_number % training.grid_update_every == 1:
                model.update_grid(batch_X)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_X)
            loss = compute_loss(prediction, batch_y, training)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        train_metrics = evaluate_split(model, X_train, y_train, experiment)
        val_metrics = evaluate_split(model, X_val, y_val, experiment)
        metrics_by_split = {"train": train_metrics, "val": val_metrics}
        monitor_value = _monitor_value(metrics_by_split, training.monitor)

        if _monitor_is_better(monitor_value, best_monitor, training.monitor, training.min_delta):
            best_monitor = monitor_value
            best_state = clone_state_dict(model)
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        row = {
            "epoch": float(epoch),
            "batch_loss_scaled": float(np.mean(epoch_losses)),
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "monitor_value": monitor_value,
        }
        history.append(row)

        if epoch == 1 or epoch % training.log_every == 0:
            print(
                f"Epoch {epoch:03d}"
                f" | loss={row['batch_loss_scaled']:.6f}"
                f" | train_mae={train_metrics['mae']:.3f}"
                f" | val_mae={val_metrics['mae']:.3f}"
                f" | val_rmse={val_metrics['rmse']:.3f}"
                f" | val_r2={val_metrics['r2']:.4f}"
                f" | best_{training.monitor}={best_monitor:.4f}"
            )

        if patience_counter >= training.patience:
            stopped_early = True
            print(f"Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    train_metrics = evaluate_split(model, X_train, y_train, experiment)
    val_metrics = evaluate_split(model, X_val, y_val, experiment)
    test_metrics = evaluate_split(model, X_test, y_test, experiment)
    test_predictions = (
        predict_batched(model, X_test, training.eval_batch_size)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
        * dataset.target_scale
    )

    metrics = {
        "mode": "standard",
        "config_path": str(experiment.config_path),
        "device": str(device),
        "input_dim": dataset.input_dim,
        "model_width": serialize_width(model.width),
        "target_scale": experiment.data.target_scale,
        "rul_tolerance": experiment.evaluation.rul_tolerance,
        "best_epoch": best_epoch,
        "trained_epochs": len(history),
        "monitor": training.monitor,
        "best_monitor_value": float(best_monitor),
        "stopped_early": stopped_early,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    return model, metrics, history, test_predictions


def save_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_predictions(path: Path, y_true_scaled: torch.Tensor, y_pred: np.ndarray, target_scale: float) -> None:
    y_true = y_true_scaled.detach().cpu().numpy().reshape(-1) * target_scale
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["actual_rul", "predicted_rul", "error"])
        writer.writeheader()
        for actual, predicted in zip(y_true, y_pred):
            writer.writerow(
                {
                    "actual_rul": float(actual),
                    "predicted_rul": float(predicted),
                    "error": float(predicted - actual),
                }
            )


def save_history_plot(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(epochs, [row["train_mae"] for row in history], label="train MAE")
    axes[0].plot(epochs, [row["val_mae"] for row in history], label="val MAE")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("RUL cycles")
    axes[0].grid(True, linewidth=0.4, alpha=0.35)
    axes[0].legend()

    axes[1].plot(epochs, [row["train_r2"] for row in history], label="train R2")
    axes[1].plot(epochs, [row["val_r2"] for row in history], label="val R2")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("R2")
    axes[1].grid(True, linewidth=0.4, alpha=0.35)
    axes[1].legend()

    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plain_width_from_model(model: KAN) -> list[int]:
    if not model.act_fun:
        raise ValueError("Cannot export a KAN without activation layers.")

    width = [int(model.act_fun[0].grid.shape[0])]
    for layer in model.act_fun:
        width.append(int(layer.coef.shape[1]))
    return width


def _infer_grid_range(model: KAN, degree: int) -> tuple[float, float]:
    left_edges: list[float] = []
    right_edges: list[float] = []

    for layer in model.act_fun:
        knots = layer.grid.detach().cpu()
        if knots.ndim != 2 or knots.shape[1] <= 2 * degree:
            raise ValueError(f"Invalid knot tensor shape for export: {tuple(knots.shape)}")
        left_edges.append(float(knots[:, degree].min()))
        right_edges.append(float(knots[:, -(degree + 1)].max()))

    return min(left_edges), max(right_edges)


def extract_export_payload(
    experiment: ExperimentConfig,
    dataset: NasaDataset,
    model: KAN,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Extract a layered KAN JSON export readable by src_inference/json_to_header.py."""
    width = _plain_width_from_model(model)
    degree = int(experiment.model.k)
    num_intervals = int(experiment.model.grid)
    num_control_points = num_intervals + degree
    num_knots = num_intervals + 1 + 2 * degree
    x_min, x_max = _infer_grid_range(model, degree)
    layers_payload: list[dict[str, Any]] = []

    if width[0] != dataset.input_dim:
        raise ValueError(f"Export width starts at {width[0]}, but dataset input_dim is {dataset.input_dim}.")

    for index, layer in enumerate(model.act_fun):
        knots = layer.grid.detach().cpu()
        control_points = layer.coef.detach().cpu()
        scale_base = layer.scale_base.detach().cpu()
        scale_sp = layer.scale_sp.detach().cpu()
        mask = layer.mask.detach().cpu()

        input_dim = width[index]
        output_dim = width[index + 1]
        expected_shapes = {
            "knots": (input_dim, num_knots),
            "control_points": (input_dim, output_dim, num_control_points),
            "scale_base": (input_dim, output_dim),
            "scale_sp": (input_dim, output_dim),
            "mask": (input_dim, output_dim),
        }
        actual_shapes = {
            "knots": tuple(knots.shape),
            "control_points": tuple(control_points.shape),
            "scale_base": tuple(scale_base.shape),
            "scale_sp": tuple(scale_sp.shape),
            "mask": tuple(mask.shape),
        }
        for name, expected_shape in expected_shapes.items():
            if actual_shapes[name] != expected_shape:
                raise ValueError(
                    f"Layer {index}: expected {name} shape {expected_shape}, "
                    f"found {actual_shapes[name]}."
                )

        layers_payload.append(
            {
                "layer_index": index,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "knots": [[float(value) for value in row] for row in knots.tolist()],
                "control_points": [
                    [[float(value) for value in edge_control_points] for edge_control_points in source_edges]
                    for source_edges in control_points.tolist()
                ],
                "scale_base": [
                    [float(value) for value in row]
                    for row in scale_base.tolist()
                ],
                "scale_sp": [
                    [float(value) for value in row]
                    for row in scale_sp.tolist()
                ],
                "mask": [
                    [float(value) for value in row]
                    for row in mask.tolist()
                ],
            }
        )

    return {
        "model_type": "kan_spline_stack",
        "dataset": "nasa_cmapss_rul",
        "config_path": str(experiment.config_path),
        "width": width,
        "input_dim": width[0],
        "output_dim": width[-1],
        "degree": degree,
        "num_control_points": num_control_points,
        "num_knots": num_knots,
        "num_intervals": num_intervals,
        "x_min": x_min,
        "x_max": x_max,
        "target_scale": float(experiment.data.target_scale),
        "feature_names": list(dataset.feature_names),
        "base_branch_enabled": True,
        "symbolic_branch_enabled": bool(experiment.model.symbolic_enabled),
        "metrics": metrics,
        "layers": layers_payload,
    }


def save_artifacts(
    experiment: ExperimentConfig,
    dataset: NasaDataset,
    model: KAN,
    metrics: dict[str, Any],
    history: list[dict[str, float]],
    test_predictions: np.ndarray,
) -> None:
    output_dir = experiment.output.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / experiment.output.model_filename
    export_path = output_dir / experiment.output.export_json_filename
    metrics_path = output_dir / experiment.output.metrics_filename
    history_path = output_dir / experiment.output.history_filename
    predictions_path = output_dir / experiment.output.predictions_filename
    config_snapshot_path = output_dir / experiment.output.config_snapshot_filename
    plot_path = output_dir / experiment.output.plot_filename
    export_payload = extract_export_payload(experiment, dataset, model, metrics)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_width": serialize_width(model.width),
            "grid": experiment.model.grid,
            "k": experiment.model.k,
            "target_scale": experiment.data.target_scale,
            "feature_names": dataset.feature_names,
            "metrics": metrics,
            "export": export_payload,
            "config": experiment.raw_config,
        },
        model_path,
    )
    write_json(export_path, export_payload)
    write_json(metrics_path, metrics)
    save_history(history_path, history)
    save_predictions(predictions_path, dataset.y_test, test_predictions, dataset.target_scale)
    save_history_plot(plot_path, history)
    shutil.copy2(experiment.config_path, config_snapshot_path)


def run_experiment(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    """Load config, train the NASA KAN regressor, and save artifacts."""
    experiment = load_config(config_path)
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

    print("\nNASA C-MAPSS RUL risultati finali")
    print(f"Config: {experiment.config_path}")
    print(f"Dataset source: {dataset.source}")
    print(f"Input dim: {dataset.input_dim}")
    print(f"Model width: {metrics['model_width']}")
    print(f"Train metrics: {json.dumps(metrics['train'], ensure_ascii=True)}")
    print(f"Validation metrics: {json.dumps(metrics['val'], ensure_ascii=True)}")
    print(f"Test metrics: {json.dumps(metrics['test'], ensure_ascii=True)}")
    print(f"Artefatti salvati in: {experiment.output.output_dir}")
    print(f"Export RISC-V JSON: {experiment.output.output_dir / experiment.output.export_json_filename}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the NASA C-MAPSS RUL KAN regressor.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to the TOML config file. Default: {DEFAULT_CONFIG_PATH}",
    )
    args = parser.parse_args(argv)
    run_experiment(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
