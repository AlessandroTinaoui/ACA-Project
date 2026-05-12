"""Training and evaluation utilities for NASA C-MAPSS KAN regression."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from kan_models.common.kan_compat import KAN
from kan_models.common.shared import clone_state_dict, serialize_width
from kan_models.models.nasa.config import ExperimentConfig, ModelConfig, TrainingConfig
from kan_models.models.nasa.data import NasaDataset


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
    metrics_path = output_dir / experiment.output.metrics_filename
    history_path = output_dir / experiment.output.history_filename
    predictions_path = output_dir / experiment.output.predictions_filename
    config_snapshot_path = output_dir / experiment.output.config_snapshot_filename
    plot_path = output_dir / experiment.output.plot_filename

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_width": serialize_width(model.width),
            "grid": experiment.model.grid,
            "k": experiment.model.k,
            "target_scale": experiment.data.target_scale,
            "feature_names": dataset.feature_names,
            "metrics": metrics,
            "config": experiment.raw_config,
        },
        model_path,
    )
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    save_history(history_path, history)
    save_predictions(predictions_path, dataset.y_test, test_predictions, dataset.target_scale)
    save_history_plot(plot_path, history)
    shutil.copy2(experiment.config_path, config_snapshot_path)
