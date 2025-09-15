# -*- coding: utf-8 -*-
"""
PyEGM package

Physics-inspired exemplar growth classifier that combines a prototype channel
and shell-based arrival centers. The public API re-exports the estimator and
its configuration dataclass.

Main entry points
-----------------
- PyEGM:            estimator with sklearn-like interface (fit/partial_fit/predict/score)
- ExplosionConfig:  configuration dataclass for the model
"""

from __future__ import annotations

from pathlib import Path

# Public API
from .pyegm import PyEGM, ExplosionConfig  # noqa: F401

__all__ = [
    "PyEGM",
    "ExplosionConfig",
    "data_dir",
    "__version__",
]


def data_dir() -> Path:
    """
    Return the path to packaged data assets inside the installed package.
    This folder may be empty depending on the distribution.
    """
    return Path(__file__).resolve().parent / "data"


# Best-effort package version (works in most install modes).
try:
    from importlib.metadata import version as _pkg_version  # type: ignore

    try:
        __version__ = _pkg_version("pyegm")
    except Exception:
        __version__ = "0+local"
except Exception:
    __version__ = "0+local"
