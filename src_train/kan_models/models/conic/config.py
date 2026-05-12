"""Configuration models and TOML loaders for conic experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kan_models.common.paths import CONFIGS_DIR
from kan_models.common.shared import load_toml, resolve_path


DEFAULT_CONIC_CONFIG_PATH = CONFIGS_DIR / "conic" / "default.toml"
DEFAULT_CONTINUAL_CONFIG_PATH = DEFAULT_CONIC_CONFIG_PATH
CONTINUAL_CONFIG_VARIANTS = {
    "standard": "standard",
    "normal": "standard",
    "reversed": "reversed",
}
DEFAULT_CONFIG_PATH = DEFAULT_CONIC_CONFIG_PATH
DEFAULT_REVERSED_CONFIG_PATH = DEFAULT_CONIC_CONFIG_PATH


@dataclass
class ConicExperimentConfig:
    pruning: bool = False
    continual: bool = False
    variant: str = "standard"

    @property
    def mode(self) -> str:
        if self.continual and self.pruning:
            raise ValueError("Set only one of experiment.pruning or experiment.continual to true.")
        if self.continual:
            return "continual"
        if self.pruning:
            return "pruning"
        return "baseline"


@dataclass
class ConicDataConfig:
    csv_path: Path
    test_ratio: float
    seed: int
    target_column: str = "shape"


@dataclass
class ConicModelConfig:
    hidden: int
    grid: int
    spline_order: int
    auto_save: bool = False
    device: str = "auto"


@dataclass
class ConicOutputConfig:
    metrics_path: Path | None = None
    class_tests_path: Path | None = None
    high_loss_path: Path | None = None
    run_config_path: Path | None = None
    final_model_path: Path | None = None
    accuracy_config_path: Path | None = None
    plot_dir: Path | None = None


@dataclass
class ConicPlotConfig:
    clear_old_plots: bool = True
    prediction_examples: int = 2
    high_loss_examples: int = 12
    plot_edge_functions: bool = True
    enable_predictions: bool = True
    enable_loss: bool = True
    enable_error: bool = True
    enable_class_tests: bool = True
    enable_confusion_matrices: bool = True
    enable_edge_functions: bool = True


@dataclass
class BaselineTrainingConfig:
    optimizer: str
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    steps: int
    label_smoothing: float
    early_stopping_patience: int
    early_stopping_min_delta: float
    lr_scheduler_factor: float
    lr_scheduler_patience: int
    grid_update_every: int
    stop_grid_update_step: int
    lr_scheduler: str = "ReduceLROnPlateau"


@dataclass
class BaselineConfig:
    data: ConicDataConfig
    model: ConicModelConfig
    training: BaselineTrainingConfig
    standardize: bool
    plots: ConicPlotConfig
    output: ConicOutputConfig
    raw_config: dict[str, Any]
    config_path: Path


@dataclass
class PruningTrainingConfig:
    optimizer: str
    learning_rate: float
    standardize: bool
    probe_hidden: int
    probe_steps: int
    start_hidden: int
    keep_hidden_schedule: list[int]
    train_steps_per_model: int


@dataclass
class PruningConfig:
    data: ConicDataConfig
    model: ConicModelConfig
    training: PruningTrainingConfig
    plots: ConicPlotConfig
    output: ConicOutputConfig
    raw_config: dict[str, Any]
    config_path: Path


@dataclass
class ContinualTrainingConfig:
    epochs_per_task: list[int]
    batch_size: int
    optimizer: str
    learning_rate: float
    min_learning_rate: float
    mask_future_classes: bool
    label_smoothing: float
    use_balanced_sampler: bool
    use_lwf: bool
    initial_lambda_kd: float
    lambda_kd_decay: float
    distillation_temperature: float
    freeze_grid_after_first: bool
    enabled: bool
    check_every: int
    patience: int
    min_delta: float
    anti_forgetting_enabled: bool
    anti_forgetting_tolerance: float
    anti_forgetting_patience: int
    update_every: int
    stop_update_epoch: int


@dataclass
class ContinualConfig:
    data: ConicDataConfig
    model: ConicModelConfig
    training: ContinualTrainingConfig
    standardize: bool
    plots: ConicPlotConfig
    variant: str
    schedule: list[dict[str, int]]
    output: ConicOutputConfig
    raw_config: dict[str, Any]
    config_path: Path


def _mode_section(raw_config: dict[str, Any], mode: str) -> dict[str, Any]:
    section = raw_config.get(mode)
    if isinstance(section, dict):
        return section
    return raw_config


def load_experiment_config(path: str | Path) -> ConicExperimentConfig:
    _, raw_config = load_toml(path)
    section = raw_config.get("experiment", {})
    if not isinstance(section, dict):
        section = {}
    return ConicExperimentConfig(
        pruning=bool(section.get("pruning", False)),
        continual=bool(section.get("continual", False)),
        variant=str(section.get("variant", "standard")),
    )


def _optional_path(base_dir: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    return resolve_path(base_dir, value)


def _load_output_config(base_dir: Path, section: dict[str, Any]) -> ConicOutputConfig:
    return ConicOutputConfig(
        metrics_path=_optional_path(base_dir, section.get("metrics_path")),
        class_tests_path=_optional_path(base_dir, section.get("class_tests_path")),
        high_loss_path=_optional_path(base_dir, section.get("high_loss_path")),
        run_config_path=_optional_path(base_dir, section.get("run_config_path")),
        final_model_path=_optional_path(base_dir, section.get("final_model_path")),
        accuracy_config_path=_optional_path(base_dir, section.get("accuracy_config_path")),
        plot_dir=_optional_path(base_dir, section.get("plot_dir")),
    )


def _override_output_config(
    output: ConicOutputConfig,
    base_dir: Path,
    section: dict[str, Any],
) -> ConicOutputConfig:
    return ConicOutputConfig(
        metrics_path=_optional_path(base_dir, section.get("metrics_path")) if "metrics_path" in section else output.metrics_path,
        class_tests_path=_optional_path(base_dir, section.get("class_tests_path"))
        if "class_tests_path" in section
        else output.class_tests_path,
        high_loss_path=_optional_path(base_dir, section.get("high_loss_path")) if "high_loss_path" in section else output.high_loss_path,
        run_config_path=_optional_path(base_dir, section.get("run_config_path")) if "run_config_path" in section else output.run_config_path,
        final_model_path=_optional_path(base_dir, section.get("final_model_path"))
        if "final_model_path" in section
        else output.final_model_path,
        accuracy_config_path=_optional_path(base_dir, section.get("accuracy_config_path"))
        if "accuracy_config_path" in section
        else output.accuracy_config_path,
        plot_dir=_optional_path(base_dir, section.get("plot_dir")) if "plot_dir" in section else output.plot_dir,
    )


def _load_base_sections(raw_config: dict[str, Any], config_dir: Path) -> tuple[
    ConicDataConfig,
    ConicModelConfig,
    ConicOutputConfig,
]:
    data_section = raw_config["data"]
    split_section = raw_config["split"]
    model_section = raw_config["model"]
    output_section = raw_config["output"]

    data = ConicDataConfig(
        csv_path=resolve_path(config_dir, data_section["csv_path"]),
        target_column=data_section.get("target_column", "shape"),
        test_ratio=float(split_section["test_ratio"]),
        seed=int(split_section["seed"]),
    )
    model = ConicModelConfig(
        hidden=int(model_section["hidden"]),
        grid=int(model_section["grid"]),
        spline_order=int(model_section["spline_order"]),
        auto_save=bool(model_section.get("auto_save", False)),
        device=str(model_section.get("device", "auto")),
    )
    output = _load_output_config(config_dir, output_section)
    return data, model, output


def load_baseline_config(path: str | Path) -> BaselineConfig:
    config_path, raw_config = load_toml(path)
    config_dir = config_path.parent
    mode_config = _mode_section(raw_config, "baseline")
    data, model, output = _load_base_sections(
        {
            "data": raw_config["data"],
            "split": raw_config["split"],
            "model": mode_config["model"],
            "output": mode_config["output"],
        },
        config_dir,
    )
    training_section = dict(mode_config["training"])
    standardize = bool(mode_config.get("standardize", raw_config.get("standardize", training_section.pop("standardize", True))))
    training = BaselineTrainingConfig(**training_section)
    plots = ConicPlotConfig(**mode_config.get("plots", {}))
    return BaselineConfig(
        data=data,
        model=model,
        training=training,
        standardize=standardize,
        plots=plots,
        output=output,
        raw_config=raw_config,
        config_path=config_path,
    )


def load_pruning_config(path: str | Path) -> PruningConfig:
    config_path, raw_config = load_toml(path)
    config_dir = config_path.parent
    mode_config = _mode_section(raw_config, "pruning")
    data, model, output = _load_base_sections(
        {
            "data": raw_config["data"],
            "split": raw_config["split"],
            "model": mode_config["model"],
            "output": mode_config["output"],
        },
        config_dir,
    )
    training = PruningTrainingConfig(**mode_config["training"])
    plots = ConicPlotConfig(**mode_config.get("plots", {}))
    return PruningConfig(
        data=data,
        model=model,
        training=training,
        plots=plots,
        output=output,
        raw_config=raw_config,
        config_path=config_path,
    )


def load_continual_config(path: str | Path, variant: str | None = None) -> ContinualConfig:
    config_path, raw_config = load_toml(path)
    config_dir = config_path.parent
    mode_config = _mode_section(raw_config, "continual")
    data, model, output = _load_base_sections(
        {
            "data": raw_config["data"],
            "split": raw_config["split"],
            "model": mode_config["model"],
            "output": mode_config["output"],
        },
        config_dir,
    )
    training_section = dict(mode_config["training"])
    early_stopping_section = dict(mode_config["early_stopping"])
    grid_section = dict(mode_config["grid"])
    standardize = bool(mode_config.get("standardize", raw_config.get("standardize", grid_section.pop("standardize", True))))
    training = ContinualTrainingConfig(
        **training_section,
        **early_stopping_section,
        **grid_section,
    )
    plots = ConicPlotConfig(**mode_config.get("plots", {}))
    experiment_section = raw_config.get("experiment", {})
    configured_variant = str(experiment_section.get("variant", raw_config.get("active_variant", "standard")))
    requested_variant = configured_variant if variant is None else variant
    normalized_variant = CONTINUAL_CONFIG_VARIANTS.get(requested_variant, requested_variant)
    schedule_variants = mode_config.get("schedule_variants")
    if schedule_variants is None:
        if normalized_variant != "standard":
            raise ValueError(f"Variant '{requested_variant}' not available in {config_path}.")
        schedule = mode_config["schedule"]
    else:
        if normalized_variant not in schedule_variants:
            available = ", ".join(sorted(schedule_variants))
            raise ValueError(
                f"Variant '{requested_variant}' not available in {config_path}. Available: {available}"
            )
        schedule = schedule_variants[normalized_variant]

    variant_outputs = mode_config.get("variant_outputs", {})
    if normalized_variant in variant_outputs:
        output = _override_output_config(output, config_dir, variant_outputs[normalized_variant])

    return ContinualConfig(
        data=data,
        model=model,
        training=training,
        standardize=standardize,
        plots=plots,
        variant=normalized_variant,
        schedule=schedule,
        output=output,
        raw_config=raw_config,
        config_path=config_path,
    )
