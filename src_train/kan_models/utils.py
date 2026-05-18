"""Shared utilities for KAN training and export scripts."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

try:
    from kan import KAN  # type: ignore[attr-defined]
except (ImportError, AttributeError):
    from kan.MultKAN import MultKAN as KAN

if TYPE_CHECKING:
    from kan.MultKAN import MultKAN


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "src_train" / "configs"
QUANTKAN_ROOT = PROJECT_ROOT / "external" / "QuantKAN"


def configure_matplotlib(headless: bool = True) -> None:
    """Configure matplotlib before importing pyplot."""
    import matplotlib

    if headless:
        matplotlib.use("Agg")


def detect_device(preferred: str | None = None) -> torch.device:
    """Resolve the configured PyTorch device."""
    if preferred is not None and preferred != "auto":
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    """Resolve a potentially relative path against a base directory."""
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_toml(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Load a TOML file and return both its resolved path and parsed content."""
    resolved_path = Path(path).resolve()
    with resolved_path.open("rb") as handle:
        return resolved_path, tomllib.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with a stable, human-readable format."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def clone_state_dict(model: "MultKAN") -> dict[str, Any]:
    """Clone a model state for later restoration."""
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def serialize_width(width: list[Any]) -> list[Any]:
    """Convert pykan width metadata into JSON-friendly values."""
    serialized: list[Any] = []
    for layer in width:
        if isinstance(layer, (list, tuple)):
            serialized.append([int(value) for value in layer])
        else:
            serialized.append(int(layer))
    return serialized


def ensure_quantkan_path() -> Path:
    """Add the local QuantKAN checkout to sys.path."""
    if not QUANTKAN_ROOT.exists():
        raise FileNotFoundError(
            f"QuantKAN checkout not found: {QUANTKAN_ROOT}. "
            "Clone or add the submodule under external/QuantKAN first."
        )

    root_text = str(QUANTKAN_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return QUANTKAN_ROOT


def quantkan_commit() -> str | None:
    """Return the current QuantKAN commit hash when available."""
    ensure_quantkan_path()
    try:
        return subprocess.check_output(
            ["git", "-C", str(QUANTKAN_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_ptq_api() -> dict[str, Any]:
    """Import the QuantKAN PTQ helpers used by this project."""
    ensure_quantkan_path()

    from ptq.actquant import (  # type: ignore[import-not-found]
        collect_input_activation_scales,
        save_act_params,
        unwrap_input_quant,
        wrap_input_quant,
    )
    from ptq.uniform import quantize_weights_uniform  # type: ignore[import-not-found]

    return {
        "collect_input_activation_scales": collect_input_activation_scales,
        "quantize_weights_uniform": quantize_weights_uniform,
        "save_act_params": save_act_params,
        "unwrap_input_quant": unwrap_input_quant,
        "wrap_input_quant": wrap_input_quant,
    }


__all__ = [
    "CONFIGS_DIR",
    "KAN",
    "PROJECT_ROOT",
    "QUANTKAN_ROOT",
    "clone_state_dict",
    "configure_matplotlib",
    "detect_device",
    "ensure_quantkan_path",
    "load_ptq_api",
    "load_toml",
    "quantkan_commit",
    "resolve_path",
    "serialize_width",
    "write_json",
]
