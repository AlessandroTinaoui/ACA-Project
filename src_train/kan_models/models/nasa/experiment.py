"""Entry point for NASA C-MAPSS KAN regression experiments."""

from __future__ import annotations

import json
from pathlib import Path

from kan_models.common.runtime import detect_device
from kan_models.models.nasa.config import DEFAULT_CONFIG_PATH, load_config
from kan_models.models.nasa.data import load_dataset
from kan_models.models.nasa.training import build_model, save_artifacts, train_model


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
