"""Helpers for importing the local QuantKAN checkout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
QUANTKAN_ROOT = PROJECT_ROOT / "external" / "QuantKAN"


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
    """Import the PTQ helpers used by the local integration."""
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

