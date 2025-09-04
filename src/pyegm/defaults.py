# -*- coding: utf-8 -*-
"""
PyEGM Defaults Resolver (CPU-only)

Fit-time resolution order:
  1) user kwargs (non-None)
  2) YAML selected by `preset` (or env `PYEGM_DEFAULTS_FILE`)
  3) hard fallbacks (this file)

Preset:
- "auto": choose among balanced_kshot / imbalanced_kshot / extreme_lowshot via label stats.
- {"balanced_kshot","imbalanced_kshot","extreme_lowshot","fscil"}: use the recommended YAML.
- Any string ending with ".yaml": treated as a YAML file name or absolute path.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ----------------------- Hard fallbacks -----------------------
_HARD_DEFAULTS: Dict[str, Any] = {
    "num_points": 100,
    "total_energy": 1.0,
    "mass": 1.0,
    "explosion_time": 1.0,
    "noise_scale": 0.06,
    "noise_decay": 1.0,
    "adaptive_radius_mode": "local",  # or "global", or None + fixed_radius
    "fixed_radius": None,
    "use_radius_clip": True,
    "dynamic_energy": True,
    "dirichlet_alpha": 1.0,
    "max_samples": 10_000,
    "random_state": 0,
    "points_schedule": "inv_size",       # {"uniform","inv_size","sqrt_inv_size"}
    "energy_schedule": "inv_size",
    "seed_mode": "per_sample",           # or "centroid"
    "num_iters": 2,
    "energy_decay": 0.7,
    "radius_growth": 1.1,
    "local_k": 5,
    "anisotropy": 0.2,
    "deflect_rival": True,
    "deflect_strength": 0.2,
    "momentum_conserve": True,
    "shell_ratio": 0.3,
    "shell_jitter": 0.1,
    "path_steps": 4,
    "path_step_size": 0.5,
    "path_k": 40,
    "path_gamma": 0.6,
    "step_weight_mode": "early",         # {"none","early","late"}
    "step_gamma": 0.7,
    "governor": True,
}

# keys that indicate a "__rules.yaml"
_RULE_KEYS = {
    "a0","a1","a2","b0","c0","c1","d0","e0","f0","g0","g1","h0","h1",
    "j0","k0","m0","t0","nz0","s_mode"
}

# ------------------ preset → recommended YAML -----------------
_PRESET_TO_RECOMMENDED_YAML = {
    "balanced_kshot":   "balanced_kshot__rules.yaml",
    "balanced":         "balanced_kshot__rules.yaml",
    "imbalanced_kshot": "imbalanced_kshot__fixed.yaml",
    "imbalanced":       "imbalanced_kshot__fixed.yaml",
    "extreme_lowshot":  "extreme_lowshot__fixed.yaml",
    "extreme":          "extreme_lowshot__fixed.yaml",
    "fscil":            "fscil__rules.yaml",
}

def _canonical_preset(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    p = p.strip().lower()
    if p in ("balanced-kshot","balancedkshot"):
        return "balanced_kshot"
    if p in ("imbalanced-kshot","imbalancedkshot"):
        return "imbalanced_kshot"
    if p in ("extreme-lowshot","extremelowshot"):
        return "extreme_lowshot"
    return p

# ------------------------- YAML helpers -----------------------
def _data_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "data")

def _yaml_path(override_name: Optional[str] = None) -> str:
    name = override_name or os.environ.get("PYEGM_DEFAULTS_FILE", "balanced_kshot__rules.yaml")
    if os.path.isabs(name):
        return name
    return os.path.join(_data_dir(), name)

def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}

def _is_rules_yaml(d: Dict[str, Any]) -> bool:
    return any(k in d for k in _RULE_KEYS)

# ------------------ Train-set stats & rules map ----------------
def _clip_float(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def _clip_int(x: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(round(x)))))

def _compute_dataset_stats(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_samples, n_features = Xs.shape
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    min_per_class = int(np.min(counts)) if len(counts) else 0
    mean_per_class = float(np.mean(counts)) if len(counts) else 0.0
    imb_ratio = float(np.min(counts) / np.max(counts)) if n_classes > 1 else 1.0

    gc = np.mean(Xs, axis=0) if n_samples > 0 else np.zeros(n_features, dtype=float)
    global_radius = float(np.median(np.linalg.norm(Xs - gc, axis=1))) if n_samples > 0 else 1.0

    safe_max_comps = max(1, min(n_features, max(1, n_samples - 1), 16))
    try:
        pca = PCA(n_components=safe_max_comps, random_state=0)
        _ = pca.fit_transform(Xs)
        eig = pca.explained_variance_
        if eig.size == 0:
            ellipticity = 1.0
        else:
            eig_mean = float(np.mean(eig))
            ellipticity = float((np.max(eig) / (eig_mean + 1e-9))) if eig_mean > 0 else 1.0
    except Exception:
        ellipticity = 1.0

    if n_samples > 0:
        total_var = float(np.mean(np.sum((Xs - gc) ** 2, axis=1)))
    else:
        total_var = 0.0
    class_means_list = [np.mean(Xs[y == c], axis=0) for c in classes if np.any(y == c)]
    between_var = 0.0
    if len(class_means_list) > 0:
        class_means = np.stack(class_means_list, axis=0)
        between_var = float(np.mean(np.sum((class_means - gc) ** 2, axis=1)))
    sep_proxy = float(np.clip(between_var / (total_var + 1e-9), 0.0, 1.0)) if total_var > 0 else 0.0

    return dict(
        n_samples=int(n_samples),
        n_features=int(n_features),
        n_classes=int(n_classes),
        min_per_class=int(min_per_class),
        mean_per_class=float(mean_per_class),
        imbalance_ratio=float(imb_ratio),
        global_radius=float(global_radius),
        pca_ellipticity=float(ellipticity),
        sep_proxy=float(sep_proxy),
    )

def _rules_to_params(stats: Dict[str, float], coef: Dict[str, float]) -> Dict[str, Any]:
    import math
    N   = float(stats["n_samples"])
    imb = float(stats["imbalance_ratio"])
    rad = float(stats["global_radius"])
    ell = float(stats["pca_ellipticity"])
    sep = float(stats["sep_proxy"])
    logN = math.log(max(N, 3.0))

    num_points     = _clip_int(coef["a0"] + coef["a1"] * logN + coef["a2"] * math.log(stats["min_per_class"] + 1.0), 20, 200)
    total_energy   = _clip_float(coef["b0"] * rad, 0.2, 3.0)
    energy_decay   = _clip_float(coef["c0"] + coef["c1"] * (1.0 - imb), 0.5, 0.95)
    radius_growth  = _clip_float(1.0 + coef["d0"] * (1.0 - imb), 1.0, 1.3)

    anisotropy       = _clip_float(coef["e0"] * (ell - 1.0), 0.0, 0.5)
    deflect_strength = _clip_float(coef["f0"] * sep, 0.0, 0.5)

    path_k         = _clip_int(coef["g0"] + coef["g1"] * logN, 20, 60)
    path_steps     = _clip_int(coef["h0"] + coef["h1"] * sep, 3, 8)
    path_step_size = _clip_float(coef["j0"], 0.2, 0.8)
    path_gamma     = _clip_float(coef["k0"], 0.4, 0.9)

    step_gamma     = _clip_float(coef["m0"], 0.5, 0.9)
    sm             = coef["s_mode"]
    if sm < -0.33:
        step_weight_mode = "none"
    elif sm > 0.33:
        step_weight_mode = "early"
    else:
        step_weight_mode = "late"

    seed_mode = "centroid" if stats["mean_per_class"] < 12 else "per_sample"
    num_iters = _clip_int(coef["t0"], 1, 3)

    params = dict(_HARD_DEFAULTS)
    params.update(dict(
        governor=True,
        points_schedule="inv_size",
        dirichlet_alpha=1.0,
        local_k=5,
        momentum_conserve=True,
        deflect_rival=True,
        dynamic_energy=True,
        use_radius_clip=True,

        num_points=num_points,
        total_energy=total_energy,
        energy_decay=energy_decay,
        radius_growth=radius_growth,
        anisotropy=anisotropy,
        deflect_strength=deflect_strength,

        path_k=path_k,
        path_steps=path_steps,
        path_step_size=path_step_size,
        path_gamma=path_gamma,

        step_weight_mode=step_weight_mode,
        step_gamma=step_gamma,

        seed_mode=seed_mode,
        num_iters=num_iters,

        noise_scale=_clip_float(coef["nz0"], 0.005, 0.12),
    ))
    return params

# -------------------- Auto preset chooser ---------------------
def _choose_preset_from_stats(y: np.ndarray) -> str:
    if y is None or len(y) == 0:
        return "balanced_kshot"
    _, counts = np.unique(y, return_counts=True)
    min_c = int(np.min(counts))
    max_c = int(np.max(counts))
    if min_c <= 2:
        return "extreme_lowshot"
    if max_c / max(1, min_c) >= 3.0:
        return "imbalanced_kshot"
    return "balanced_kshot"

# ----------------- PUBLIC: resolve for fit() ------------------
def resolve_defaults_for_fit(
    X: np.ndarray,
    y: np.ndarray,
    user_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the final parameter dictionary for PyEGM at fit time.

    Resolution order:
      1) user_kwargs (non-None)
      2) YAML from `preset` (or env)
      3) hard defaults
    """
    final_params = dict(_HARD_DEFAULTS)

    preset_raw = (user_kwargs or {}).get("preset", "auto")
    preset = _canonical_preset(preset_raw if preset_raw is not None else "auto")

    # Decide YAML by preset/env/auto
    if isinstance(preset_raw, str) and preset_raw.strip().lower().endswith(".yaml"):
        yaml_name = preset_raw.strip()
    elif preset in (None, "", "auto"):
        auto_dir = _choose_preset_from_stats(y)
        yaml_name = _PRESET_TO_RECOMMENDED_YAML.get(auto_dir, "balanced_kshot__rules.yaml")
    elif preset in _PRESET_TO_RECOMMENDED_YAML:
        yaml_name = _PRESET_TO_RECOMMENDED_YAML[preset]
    else:
        yaml_name = os.environ.get("PYEGM_DEFAULTS_FILE", "balanced_kshot__rules.yaml")

    yaml_doc = _load_yaml(_yaml_path(yaml_name))

    # Merge from YAML
    if yaml_doc:
        if _is_rules_yaml(yaml_doc):
            missing = [k for k in _RULE_KEYS if k not in yaml_doc]
            if missing:
                raise KeyError(f"Rules YAML missing keys: {missing}")
            stats = _compute_dataset_stats(X, y)
            coef = {k: float(yaml_doc[k]) for k in _RULE_KEYS}
            final_params.update(_rules_to_params(stats, coef))
        else:
            final_params.update(yaml_doc)

    # Apply user overrides last
    for k, v in (user_kwargs or {}).items():
        if v is not None and k != "preset":
            final_params[k] = v

    return final_params

# ---------------- Convenience (no rules mapping) --------------
def get_default(key: str) -> Any:
    yaml_doc = _load_yaml(_yaml_path(None))
    if yaml_doc and (not _is_rules_yaml(yaml_doc)) and key in yaml_doc:
        return yaml_doc[key]
    return _HARD_DEFAULTS.get(key)

def all_defaults() -> Dict[str, Any]:
    merged = dict(_HARD_DEFAULTS)
    yaml_doc = _load_yaml(_yaml_path(None))
    if yaml_doc and (not _is_rules_yaml(yaml_doc)):
        merged.update(yaml_doc)
    return merged

DEFAULTS = all_defaults()
