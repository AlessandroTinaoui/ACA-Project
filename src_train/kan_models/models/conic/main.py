"""Unified entrypoint for conic experiments driven by one TOML file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from kan_models.common.runtime import configure_matplotlib


configure_matplotlib()

from kan_models.models.conic.baseline import run_baseline
from kan_models.models.conic.config import DEFAULT_CONIC_CONFIG_PATH, load_experiment_config
from kan_models.models.conic.continual.experiment import run_continual
from kan_models.models.conic.pruning import run_pruning


def run_experiment(config_path: str | Path = DEFAULT_CONIC_CONFIG_PATH) -> object:
    experiment = load_experiment_config(config_path)
    if experiment.mode == "baseline":
        return run_baseline(config_path)
    if experiment.mode == "pruning":
        return run_pruning(config_path)
    return run_continual(config_path, variant=experiment.variant)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a conic experiment from a unified TOML config.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONIC_CONFIG_PATH),
        help=f"Path to the TOML config file. Default: {DEFAULT_CONIC_CONFIG_PATH}",
    )
    args = parser.parse_args(argv)
    run_experiment(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
