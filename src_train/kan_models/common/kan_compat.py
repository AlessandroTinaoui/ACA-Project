"""Compatibility import for pykan across package layouts."""

from __future__ import annotations

try:
    from kan import KAN  # type: ignore[attr-defined]
except (ImportError, AttributeError):
    from kan.MultKAN import MultKAN as KAN

__all__ = ["KAN"]
