# filename: defaults.py
# -*- coding: utf-8 -*-
"""
Load default hyperparameters for PyEGM from an embedded YAML file.
If a key is missing in YAML, we fall back to safe hard-coded defaults.
"""

import os
from typing import Any, Dict

import yaml


# ----------------------------
# Hard fallback (safe defaults)
# ----------------------------
_HARD_DEFAULTS: Dict[str, Any] = {
    # Core physics / synthesis
    "num_points": 100,
    "total_energy": 1.0,
    "mass": 1.0,
    "explosion_time": 1.0,
    "noise_scale": 0.06,
    "noise_decay": 1.0,

    # Radius & energy scaling
    "adaptive_radius_mode": "local",  # {"local","global",None}
    "fixed_radius": None,
    "use_radius_clip": True,
    "dynamic_energy": True,
    "dirichlet_alpha": 1.0,

    # Memory & reproducibility
    "max_samples": 10_000,
    "random_state": 0,

    # Scheduling
    "points_schedule": "inv_size",
    "energy_schedule": "inv_size",
    "seed_mode": "per_sample",        # {"per_sample","centroid"}
    "num_iters": 2,
    "energy_decay": 0.7,
    "radius_growth": 1.1,

    # Local geometry
    "local_k": 5,
    "anisotropy": 0.2,

    # Rival deflection & momentum
    "deflect_rival": True,
    "deflect_strength": 0.2,
    "momentum_conserve": True,

    # Shell
    "shell_ratio": 0.3,
    "shell_jitter": 0.1,

    # Trend path voting
    "path_steps": 4,
    "path_step_size": 0.5,
    "path_k": 40,
    "path_gamma": 0.6,

    # Temporal weights
    "step_weight_mode": "early",      # {"none","early","late"}
    "step_gamma": 0.7,

    # Governor
    "governor": True,
}


def _yaml_path() -> str:
    here = os.path.dirname(__file__)
    return os.path.join(here, "data", "best_rules.yaml")


def load_yaml_defaults() -> Dict[str, Any]:
    """
    Load YAML defaults from pyegm/data/best_rules.yaml if present.
    Return empty dict if file missing.
    """
    path = _yaml_path()
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


# Loaded at import time
_YAML_DEFAULTS: Dict[str, Any] = load_yaml_defaults()


def get_default(key: str) -> Any:
    """
    Get a default value, prefer YAML; otherwise fall back to hard defaults.
    """
    if key in _YAML_DEFAULTS:
        return _YAML_DEFAULTS[key]
    return _HARD_DEFAULTS.get(key)


def all_defaults() -> Dict[str, Any]:
    """
    Merge YAML over hard defaults and return a complete dict.
    """
    merged = dict(_HARD_DEFAULTS)
    merged.update(_YAML_DEFAULTS)
    return merged


# For external imports
DEFAULTS = all_defaults()
