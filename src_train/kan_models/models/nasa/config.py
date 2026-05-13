"""Configuration models and TOML loader for NASA C-MAPSS RUL regression."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kan_models.common.paths import CONFIGS_DIR
from kan_models.common.shared import load_toml, resolve_path


DEFAULT_CONFIG_PATH = CONFIGS_DIR / "nasa" / "default.toml"


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
