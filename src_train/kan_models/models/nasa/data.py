"""Data loading utilities for NASA C-MAPSS RUL regression."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kan_models.models.nasa.config import DataConfig


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
