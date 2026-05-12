"""Project-level paths used across experiments."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src_train"
DATASET_DIR = PROJECT_ROOT / "datasets"
CONFIGS_DIR = SRC_DIR / "configs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def project_path(*parts: str) -> Path:
    """Build an absolute path inside the repository root."""
    return PROJECT_ROOT.joinpath(*parts)
