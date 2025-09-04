# -*- coding: utf-8 -*-
"""
PyEGM — Physics-Inspired Exemplar Growth Model
==============================================

A sample-growth classifier inspired by explosion dynamics.

Unified backend policy:
- CPU  : HNSWLIB (approximate kNN in L2^2 space)  ← default for reproducibility & speed
- GPU  : PyTorch CUDA (exact kNN via blocked L2^2 distance + topk)
- AUTO : pick GPU if torch.cuda.is_available() else CPU

Features:
- Preset selection via `preset` (auto / scenario names / external YAML)
- Rules-based or fixed-YAML configuration resolution (see defaults.py)
- Incremental training (fit + partial_fit) with synthetic sample generation
- Trend-path voting for prediction
- Save/Load (arrays + ANN index), export fixed YAML, short resume, visualization

This file has no FAISS dependency by design to keep platform consistency.
"""

from __future__ import annotations

import os
from typing import Literal, Optional, Sequence, Tuple, Dict, List

import hnswlib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_X_y

from .defaults import resolve_defaults_for_fit


# ------------------------------- Torch probe ---------------------------------
try:
    import torch
    _HAVE_TORCH = True
    _HAVE_TORCH_CUDA = torch.cuda.is_available()
except Exception:  # torch not installed or import failed
    torch = None  # type: ignore
    _HAVE_TORCH = False
    _HAVE_TORCH_CUDA = False


Vector = np.ndarray


def _warn(msg: str):
    print(f"[WARN] {msg}")


class PyEGM(BaseEstimator, ClassifierMixin):
    """
    PyEGM (Physics-Inspired Exemplar Growth Model).

    Parameters are not fixed at init time. We keep user overrides in `_user_kwargs`
    and resolve final hyper-parameters at `fit(...)` by merging:
        user kwargs (non-None) > preset/YAML > code defaults.
    """

    _PRESET_TO_RULES_BASENAME: Dict[str, str] = {
        "balanced_kshot": "balanced_kshot__rules.yaml",
        "imbalanced_kshot": "imbalanced_kshot__rules.yaml",
        "extreme_lowshot": "extreme_lowshot__rules.yaml",
        "fscil": "fscil__rules.yaml",
    }

    def __init__(
        self,
        *,
        # Preset / rules source
        preset: Optional[str] = "auto",
        # Synthesis / physics
        num_points: Optional[int] = None,
        total_energy: Optional[float] = None,
        mass: Optional[float] = None,
        explosion_time: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_decay: Optional[float] = None,
        # Radius & energy scaling
        adaptive_radius_mode: Optional[Literal["local", "global", None]] = None,
        fixed_radius: Optional[float] = None,
        use_radius_clip: Optional[bool] = None,
        dynamic_energy: Optional[bool] = None,
        dirichlet_alpha: Optional[float] = None,
        # Memory & reproducibility
        max_samples: Optional[int] = None,
        random_state: Optional[int] = None,
        # Scheduling & seeding
        points_schedule: Optional[str] = None,
        energy_schedule: Optional[str] = None,
        seed_mode: Optional[Literal["per_sample", "centroid"]] = None,
        num_iters: Optional[int] = None,
        energy_decay: Optional[float] = None,
        radius_growth: Optional[float] = None,
        # Local geometry & dynamics
        local_k: Optional[int] = None,
        anisotropy: Optional[float] = None,
        deflect_rival: Optional[bool] = None,
        deflect_strength: Optional[float] = None,
        momentum_conserve: Optional[bool] = None,
        # Shell controls
        shell_ratio: Optional[float] = None,
        shell_jitter: Optional[float] = None,
        # Trend-path voting
        path_steps: Optional[int] = None,
        path_step_size: Optional[float] = None,
        path_k: Optional[int] = None,
        path_gamma: Optional[float] = None,
        # Temporal weights
        step_weight_mode: Optional[Literal["none", "early", "late"]] = None,
        step_gamma: Optional[float] = None,
        # Safeguards
        governor: Optional[bool] = None,
        # Backend selection (NEW)
        platform: Literal["auto", "cpu", "cuda"] = "auto",
        torch_q_block: int = 1024,   # query batch size on Torch backend
        torch_db_block: int = 20000, # database block size on Torch backend
    ) -> None:
        self._user_kwargs = {
            "preset": preset,
            "num_points": num_points,
            "total_energy": total_energy,
            "mass": mass,
            "explosion_time": explosion_time,
            "noise_scale": noise_scale,
            "noise_decay": noise_decay,
            "adaptive_radius_mode": adaptive_radius_mode,
            "fixed_radius": fixed_radius,
            "use_radius_clip": use_radius_clip,
            "dynamic_energy": dynamic_energy,
            "dirichlet_alpha": dirichlet_alpha,
            "max_samples": max_samples,
            "random_state": random_state,
            "points_schedule": points_schedule,
            "energy_schedule": energy_schedule,
            "seed_mode": seed_mode,
            "num_iters": num_iters,
            "energy_decay": energy_decay,
            "radius_growth": radius_growth,
            "local_k": local_k,
            "anisotropy": anisotropy,
            "deflect_rival": deflect_rival,
            "deflect_strength": deflect_strength,
            "momentum_conserve": momentum_conserve,
            "shell_ratio": shell_ratio,
            "shell_jitter": shell_jitter,
            "path_steps": path_steps,
            "path_step_size": path_step_size,
            "path_k": path_k,
            "path_gamma": path_gamma,
            "step_weight_mode": step_weight_mode,
            "step_gamma": step_gamma,
            "governor": governor,
            # backend
            "platform": platform,
            "torch_q_block": int(torch_q_block),
            "torch_db_block": int(torch_db_block),
        }
        self._backend: Optional[str] = None  # "hnsw" or "torch"
        self._device: str = "cpu"            # "cpu" or "cuda"

    # ---------------------------- helpers & backend ---------------------------
    @staticmethod
    def _data_dir() -> str:
        return os.path.join(os.path.dirname(__file__), "data")

    def _resolve_yaml_from_preset(self, preset: str, X: np.ndarray, y: np.ndarray) -> Optional[str]:
        if not isinstance(preset, str) or not preset:
            return None
        preset = preset.strip()

        if preset.lower().endswith(".yaml"):
            return preset if os.path.isabs(preset) else os.path.join(self._data_dir(), preset)

        key = preset.lower()
        if key == "auto":
            classes, counts = np.unique(y, return_counts=True)
            if len(counts) == 0:
                chosen = "balanced_kshot"
            else:
                minc = int(np.min(counts))
                meanc = float(np.mean(counts))
                if minc <= 1:
                    chosen = "extreme_lowshot"
                elif meanc > 0 and (minc / meanc) < 0.6:
                    chosen = "imbalanced_kshot"
                else:
                    chosen = "balanced_kshot"
            return os.path.join(self._data_dir(), self._PRESET_TO_RULES_BASENAME[chosen])

        if key in self._PRESET_TO_RULES_BASENAME:
            return os.path.join(self._data_dir(), self._PRESET_TO_RULES_BASENAME[key])

        return None

    def _apply_resolved(self, params: dict) -> None:
        for k, v in params.items():
            setattr(self, k, v)

    def _select_backend(self):
        plat = (self._user_kwargs.get("platform") or "auto").lower()
        if plat == "cuda":
            if not _HAVE_TORCH:
                _warn("Torch not installed; falling back to CPU/HNSW.")
                self._device = "cpu"
            else:
                self._device = "cuda" if _HAVE_TORCH_CUDA else "cpu"
                if self._device == "cpu":
                    _warn("CUDA not available; falling back to CPU/HNSW.")
        elif plat == "cpu":
            self._device = "cpu"
        else:  # auto
            self._device = "cuda" if _HAVE_TORCH_CUDA else "cpu"

        self._backend = "torch" if (self._device == "cuda") else "hnsw"

    # ------------------------------- fit / pfit -------------------------------
    def fit(self, X: Vector, y: Sequence[int]):
        X, y = check_X_y(X, y)
        X = X.astype(np.float32, copy=False)

        # Resolve presets / defaults
        _prev_yaml = os.environ.get("PYEGM_DEFAULTS_FILE", None)
        _tmp_set = False
        try:
            preset = self._user_kwargs.get("preset", None)
            if isinstance(preset, str) and preset.strip():
                sel = self._resolve_yaml_from_preset(preset, X, np.asarray(y))
                if sel:
                    os.environ["PYEGM_DEFAULTS_FILE"] = sel
                    _tmp_set = True
            resolved = resolve_defaults_for_fit(X, np.asarray(y), self._user_kwargs)
        finally:
            if _tmp_set:
                if _prev_yaml is None:
                    os.environ.pop("PYEGM_DEFAULTS_FILE", None)
                else:
                    os.environ["PYEGM_DEFAULTS_FILE"] = _prev_yaml

        if self._user_kwargs.get("random_state") is not None:
            resolved["random_state"] = self._user_kwargs["random_state"]
        self._apply_resolved(resolved)

        # Cache original set
        self._rng = np.random.default_rng(self.random_state)
        self._classes_, indices = np.unique(y, return_inverse=True)
        self._orig_X_ = X.copy()
        self._orig_y_ = np.asarray(y).copy()

        # Global geometry scalars
        self._radius_ = self._adaptive_radius(X, indices)
        self._energy_ = self._rescale_total_energy(self.total_energy, self._radius_)

        # Synthesize
        synth_X, synth_y, synth_E, synth_V, synth_S = self._multi_iter_generate(X, y)

        # Merge original + synthetic
        Xw = np.vstack([X, synth_X]) if len(synth_X) else X
        yw = np.concatenate([y, synth_y]) if len(synth_y) else np.asarray(y)
        is_synth = (np.concatenate([np.zeros(len(X), dtype=bool), np.ones(len(synth_X), dtype=bool)])
                    if len(synth_X) else np.zeros(len(X), dtype=bool))
        point_energy = (np.concatenate([np.full(len(X), 1e-9, dtype=np.float32), synth_E])
                        if len(synth_X) else np.full(len(X), 1e-9, dtype=np.float32))
        zero_V = np.zeros((len(X), X.shape[1]), dtype=np.float32)
        point_vec = np.vstack([zero_V, synth_V]) if len(synth_X) else zero_V
        point_step = (np.concatenate([np.zeros(len(X), dtype=np.int32), synth_S])
                      if len(synth_X) else np.zeros(len(X), dtype=np.int32))

        self._X_, self._y_ = Xw.astype(np.float32, copy=False), yw
        self._is_synth_ = is_synth
        self._point_energy_ = point_energy.astype(np.float32, copy=False)
        self._point_vec_ = point_vec.astype(np.float32, copy=False)
        self._point_step_ = point_step.astype(np.int32, copy=False)

        # Backend
        self._select_backend()
        if self._backend == "hnsw":
            self._build_nn_index(self._X_.shape[1])
        else:
            self._prepare_torch_bank()
        return self

    def partial_fit(self, X: Vector, y: Sequence[int], classes: Optional[Sequence[int]] = None):
        if not hasattr(self, "_X_"):
            return self.fit(X, y)

        X, y = check_X_y(X, y)
        X = X.astype(np.float32, copy=False)
        if not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(self.random_state)

        if hasattr(self, "_classes_"):
            if classes is not None:
                self._classes_ = np.unique(np.concatenate([self._classes_, np.asarray(classes)]))
            else:
                self._classes_ = np.unique(np.concatenate([self._classes_, np.unique(y)]))
        else:
            self._classes_ = np.unique(y)

        self._orig_X_ = np.vstack([self._orig_X_, X])
        self._orig_y_ = np.concatenate([self._orig_y_, y])

        comb_X = np.vstack([self._orig_X_, X])
        comb_idx = np.searchsorted(self._classes_, np.concatenate([self._orig_y_, y]))
        self._radius_ = self._adaptive_radius(comb_X, comb_idx)
        self._energy_ = self._rescale_total_energy(self.total_energy, self._radius_)

        synth_X, synth_y, synth_E, synth_V, synth_S = self._multi_iter_generate(X, y)

        self._X_ = np.vstack([self._X_, X, synth_X]).astype(np.float32, copy=False)
        self._y_ = np.concatenate([self._y_, y, synth_y])
        self._is_synth_ = np.concatenate(
            [self._is_synth_, np.zeros(len(X), dtype=bool), np.ones(len(synth_X), dtype=bool)]
        )
        self._point_energy_ = np.concatenate(
            [self._point_energy_, np.full(len(X), 1e-9, dtype=np.float32), synth_E.astype(np.float32, copy=False)]
        )
        zero_V = np.zeros((len(X), self._X_.shape[1]), dtype=np.float32)
        self._point_vec_ = np.vstack([self._point_vec_, zero_V, synth_V.astype(np.float32, copy=False)])
        self._point_step_ = np.concatenate([self._point_step_, np.zeros(len(X), dtype=np.int32), synth_S])

        if len(self._y_) > self.max_samples:
            excess = len(self._y_) - self.max_samples
            synth_idx = np.nonzero(self._is_synth_)[0]
            to_remove = synth_idx[:excess]
            remaining = excess - len(to_remove)
            if remaining > 0:
                survivors = np.ones(len(self._y_), dtype=bool)
                survivors[to_remove] = False
                orig_candidates = np.where(survivors & (~self._is_synth_))[0]
                to_remove = np.concatenate([to_remove, orig_candidates[:remaining]])
            keep = np.ones(len(self._y_), dtype=bool)
            keep[to_remove] = False
            self._X_ = self._X_[keep]
            self._y_ = self._y_[keep]
            self._is_synth_ = self._is_synth_[keep]
            self._point_energy_ = self._point_energy_[keep]
            self._point_vec_ = self._point_vec_[keep]
            self._point_step_ = self._point_step_[keep]

        self._select_backend()
        if self._backend == "hnsw":
            self._build_nn_index(self._X_.shape[1])
        else:
            self._prepare_torch_bank()
        return self

    # --------------------------- predict / score ------------------------------
    def predict(self, X: Vector) -> np.ndarray:
        X = check_array(X).astype(np.float32, copy=False)
        if not hasattr(self, "_X_"):
            raise RuntimeError("Model has not been fitted.")

        if self._backend == "hnsw":
            if not hasattr(self, "_ann_index"):
                self._build_nn_index(X.shape[1])
            return self._predict_trend_path_cpu(X)
        else:
            if not hasattr(self, "_tx"):
                self._prepare_torch_bank()
            return self._predict_trend_path_torch(X)

    def score(self, X: Vector, y: Sequence[int], sample_weight: Optional[Sequence[float]] = None) -> float:
        return float(np.average(self.predict(X) == y, weights=sample_weight))

    # ------------------- geometry / synthesis primitives ---------------------
    def _adaptive_radius(self, X: Vector, idx: np.ndarray) -> float:
        if self.adaptive_radius_mode == "local":
            radii = []
            for c in range(len(self._classes_)):
                pts = X[idx == c]
                if len(pts) <= 1:
                    continue
                D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
                np.fill_diagonal(D, np.inf)
                k_eff = min(max(1, self.local_k), max(1, len(pts) - 1))
                k_idx = k_eff - 1
                kth = np.partition(D, k_idx, axis=1)[:, k_idx]
                radii.append(np.median(kth))
            return float(np.median(radii)) if radii else 1.0

        if self.adaptive_radius_mode == "global":
            gc = X.mean(axis=0)
            return float(np.median(np.linalg.norm(X - gc, axis=1)))

        if self.adaptive_radius_mode is None and self.fixed_radius is not None:
            return float(self.fixed_radius)

        raise ValueError("Invalid adaptive_radius_mode/fixed_radius combination")

    def _rescale_total_energy(self, energy: float, radius: float) -> float:
        return energy * radius if self.dynamic_energy else energy

    def _class_stats(self, X: Vector, y: np.ndarray) -> np.ndarray:
        centers = []
        for c in self._classes_:
            pts = X[y == c]
            centers.append(pts.mean(axis=0) if len(pts) else np.zeros(X.shape[1], dtype=np.float32))
        return np.stack(centers, axis=0).astype(np.float32, copy=False)

    def _rk_median(self, class_pts: np.ndarray, k: int) -> float:
        n = len(class_pts)
        if n <= 1:
            return float(getattr(self, "_radius_", 1.0))
        D = np.linalg.norm(class_pts[:, None, :] - class_pts[None, :, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        k_eff = min(max(1, k), n - 1)
        k_idx = k_eff - 1
        kth = np.partition(D, k_idx, axis=1)[:, k_idx]
        return float(np.median(kth))

    def _local_center_cov(self, class_pts: Vector, seed: Vector, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(class_pts) == 0:
            return seed.astype(np.float32, copy=False), np.eye(seed.shape[0], dtype=np.float32)
        dists = np.linalg.norm(class_pts - seed, axis=1)
        k_eff = min(max(1, k), len(class_pts))
        idx = np.argpartition(dists, k_eff - 1)[:k_eff]
        nbrs = class_pts[idx]
        center = nbrs.mean(axis=0).astype(np.float32, copy=False)
        Xc = nbrs - center
        if len(nbrs) <= 1:
            cov = np.eye(seed.shape[0], dtype=np.float32)
        else:
            cov = (Xc.T @ Xc) / max(1, len(nbrs) - 1)
            cov = cov.astype(np.float32, copy=False)
        return center, cov

    def _multi_iter_generate(
        self, X: Vector, y: Sequence[int]
    ) -> Tuple[Vector, np.ndarray, np.ndarray, Vector, np.ndarray]:
        y = np.asarray(y)
        dim = X.shape[1]
        classes = self._classes_

        centers_all = self._class_stats(X, y)
        synth_X: List[np.ndarray] = []
        synth_y: List[int] = []
        synth_E: List[np.ndarray] = []
        synth_V: List[np.ndarray] = []
        synth_S: List[np.ndarray] = []

        per_class_counts = np.array([np.sum(y == c) for c in classes], dtype=np.int32)
        per_class_weights = np.ones_like(per_class_counts, dtype=np.float32)
        if self.points_schedule == "inv_size":
            per_class_weights = 1.0 / np.maximum(per_class_counts, 1)
        elif self.points_schedule == "sqrt_inv_size":
            per_class_weights = 1.0 / np.maximum(np.sqrt(per_class_counts), 1.0)
        per_class_weights = per_class_weights / per_class_weights.sum()

        base_energy = self._energy_
        seeds_X = X
        seeds_y = y
        noise0 = self.noise_scale

        for t in range(max(1, self.num_iters)):
            radius_t = self._radius_ * (self.radius_growth ** t)
            energy_t = base_energy * (self.energy_decay ** t)
            noise_t = noise0 * (self.noise_decay ** t)

            class_centers = self._class_stats(seeds_X, seeds_y) if self.seed_mode == "per_sample" else centers_all

            for ci, c in enumerate(classes):
                class_pts = seeds_X[seeds_y == c] if self.seed_mode == "per_sample" else X[y == c]
                if len(class_pts) == 0:
                    continue

                seeds_for_class = class_pts if self.seed_mode == "per_sample" else class_centers[ci][None, :]
                n_synth_c = max(1, int(round(self.num_points * per_class_weights[ci])))
                energies = self._rng.dirichlet([self.dirichlet_alpha] * n_synth_c).astype(np.float32)
                energies *= energy_t

                rival_centers = (
                    np.stack([centers_all[j] for j, cc in enumerate(classes) if cc != c], axis=0)
                    if len(classes) > 1
                    else None
                )
                rival_vec = None

                if self.governor and len(class_pts) >= 2:
                    _ = self._rk_median(class_pts.astype(np.float32, copy=False), self.local_k)

                e_ptr = 0
                while e_ptr < n_synth_c:
                    seed_id = self._rng.integers(0, len(seeds_for_class))
                    seed = seeds_for_class[seed_id].astype(np.float32, copy=False)

                    center_local, cov_local = self._local_center_cov(
                        class_pts.astype(np.float32, copy=False), seed, self.local_k
                    )

                    u = seed - center_local
                    nu = np.linalg.norm(u)
                    u = (u / nu).astype(np.float32) if nu >= 1e-9 else self._rng.normal(size=dim).astype(np.float32)

                    if rival_centers is not None and self.deflect_rival:
                        rv = rival_centers - seed[None, :]
                        d = np.linalg.norm(rv, axis=1)
                        if len(d):
                            rid = np.argmin(d)
                            rdir = rv[rid] / (np.linalg.norm(rv[rid]) + 1e-9)
                            rival_vec = rdir.astype(np.float32, copy=False)

                    # anisotropy
                    if self.anisotropy > 0.0:
                        w, V = np.linalg.eigh(cov_local + 1e-6 * np.eye(cov_local.shape[0], dtype=np.float32))
                        w = np.maximum(w, 1e-6)
                        scale = (w ** (-0.5 * self.anisotropy)).astype(np.float32, copy=False)
                        M = V @ np.diag(scale) @ V.T
                        u = M @ u
                    # deflection
                    if self.deflect_rival and rival_vec is not None:
                        proj = np.dot(u, rival_vec) * rival_vec
                        u = u - self.deflect_strength * proj
                    u = u / (np.linalg.norm(u) + 1e-9)

                    remaining = n_synth_c - e_ptr
                    pair = self.momentum_conserve and remaining >= 2
                    take = 2 if pair else 1
                    Es = energies[e_ptr: e_ptr + take]

                    for k in range(take):
                        e = float(Es[k])
                        v_mag = np.sqrt(2.0 * e / self.mass) * self.explosion_time
                        direction = u if k == 0 else (-u)
                        disp = v_mag * direction

                        dnorm = np.linalg.norm(disp)
                        if self.use_radius_clip and dnorm > radius_t:
                            disp = disp * (radius_t / (dnorm + 1e-9))

                        shell = self._rng.normal(scale=self.shell_jitter, size=1).astype(np.float32)[0]
                        disp = disp * (1.0 - self.shell_ratio + (self.shell_ratio + shell) * self._rng.random())
                        point = seed + disp + self._rng.normal(scale=noise_t, size=dim).astype(np.float32)

                        synth_X.append(point.astype(np.float32, copy=False))
                        synth_y.append(int(c))
                        synth_E.append(np.float32(e))
                        synth_V.append(disp.astype(np.float32, copy=False))
                        synth_S.append(np.int32(t + 1))

                    e_ptr += take

            seeds_X = np.asarray(synth_X, dtype=np.float32)
            seeds_y = np.asarray(synth_y, dtype=classes.dtype) if len(synth_y) else np.empty((0,), dtype=classes.dtype)
            if len(seeds_X) == 0:
                break

        if len(synth_X) == 0:
            return (np.empty((0, dim), dtype=np.float32),
                    np.empty((0,), dtype=classes.dtype),
                    np.empty((0,), dtype=np.float32),
                    np.empty((0, dim), dtype=np.float32),
                    np.empty((0,), dtype=np.int32))

        return (np.asarray(synth_X, dtype=np.float32),
                np.asarray(synth_y, dtype=classes.dtype),
                np.asarray(synth_E, dtype=np.float32),
                np.asarray(synth_V, dtype=np.float32),
                np.asarray(synth_S, dtype=np.int32))

    # ----------------------------- CPU (HNSWLIB) ------------------------------
    def _build_nn_index(self, dim: int) -> None:
        self._rng = np.random.default_rng(self.random_state)
        self._ann_index = hnswlib.Index(space="l2", dim=dim)
        self._ann_index.init_index(
            max_elements=len(self._X_), M=16, ef_construction=200, random_seed=self.random_state or 0
        )
        self._ann_index.add_items(self._X_, np.arange(len(self._X_)))
        self._ann_index.set_ef(50)

    def _step_weight_cpu(self, steps: np.ndarray) -> np.ndarray:
        if self.step_weight_mode == "early":
            return (self.step_gamma ** steps).astype(np.float32)
        if self.step_weight_mode == "late":
            return (steps + 1).astype(np.float32)
        return np.ones_like(steps, dtype=np.float32)

    def _predict_trend_path_cpu(self, X: Vector) -> np.ndarray:
        n = len(X)
        out = np.empty(n, dtype=self._classes_.dtype)
        K = int(max(1, self.path_k))
        T = int(max(1, self.path_steps))
        step_size = float(self.path_step_size)
        radius = float(getattr(self, "_radius_", 1.0))
        sigma = float(self.path_gamma) * radius
        denom = 2.0 * (sigma * sigma) + 1e-12

        for i in range(n):
            x0 = X[i].astype(np.float32, copy=False)
            best_cls, best_score = None, -np.inf
            any_success = False

            for c in self._classes_:
                x = x0.copy()
                score = 0.0
                success = True

                for _ in range(T):
                    idxs, dists = self._ann_index.knn_query(x[None, :], k=K)
                    ids = idxs[0]
                    ds = dists[0]  # squared distances

                    mask = self._y_[ids] == c
                    if not np.any(mask):
                        success = False
                        break

                    ids = ids[mask]
                    ds = ds[mask]

                    v = self._point_vec_[ids]
                    en = self._point_energy_[ids]
                    sp = self._point_step_[ids].astype(np.int32)

                    en = np.where(en > 0, en, 1e-9)
                    w_step = self._step_weight_cpu(sp)
                    w = (en * w_step * np.exp(-ds / denom)).astype(np.float32, copy=False)

                    w_sum = float(np.sum(w))
                    if w_sum <= 1e-12:
                        success = False
                        break

                    v_hat = (w[:, None] * v).sum(axis=0) / w_sum
                    x = x + step_size * v_hat.astype(np.float32, copy=False)
                    score += w_sum

                if success:
                    any_success = True
                    if score > best_score:
                        best_score, best_cls = score, c

            if any_success:
                out[i] = best_cls  # type: ignore
            else:
                idxs, _ = self._ann_index.knn_query(x0[None, :], k=max(1, K))
                ids = idxs[0]
                cls, cnts = np.unique(self._y_[ids], return_counts=True)
                out[i] = cls[np.argmax(cnts)]
        return out

    # ----------------------------- GPU (Torch) --------------------------------
    def _prepare_torch_bank(self):
        assert _HAVE_TORCH, "PyTorch is required for GPU backend."
        dev = "cuda" if (_HAVE_TORCH_CUDA and self._device == "cuda") else "cpu"
        self._torch_dev = torch.device(dev)
        self._tx = torch.from_numpy(self._X_.astype(np.float32, copy=False)).to(self._torch_dev, non_blocking=True)
        self._ty = torch.from_numpy(self._y_.astype(np.int64, copy=False)).to(self._torch_dev, non_blocking=True)
        self._tvec = torch.from_numpy(self._point_vec_.astype(np.float32, copy=False)).to(self._torch_dev, non_blocking=True)
        self._teng = torch.from_numpy(self._point_energy_.astype(np.float32, copy=False)).to(self._torch_dev, non_blocking=True)
        self._tstep = torch.from_numpy(self._point_step_.astype(np.int32, copy=False)).to(self._torch_dev, non_blocking=True)
        self._classes_t = torch.from_numpy(self._classes_.astype(np.int64, copy=False)).to(self._torch_dev)
        self._tx_norm2 = (self._tx * self._tx).sum(dim=1, keepdim=True)

    def _torch_knn_topk(self, q: "torch.Tensor", k: int):
        assert _HAVE_TORCH, "PyTorch backend not available."
        with torch.no_grad():
            B, _ = q.shape
            N = self._tx.shape[0]
            q_norm2 = (q * q).sum(dim=1, keepdim=True)
            topk_dist = torch.full((B, k), float("inf"), device=self._torch_dev)
            topk_idx = torch.full((B, k), -1, dtype=torch.int64, device=self._torch_dev)
            block = int(max(1, int(self._user_kwargs.get("torch_db_block", 20000))))
            s = 0
            while s < N:
                e = min(s + block, N)
                d2 = q_norm2 + self._tx_norm2[s:e].T - 2.0 * (q @ self._tx[s:e].T)
                cand = torch.cat([topk_dist, d2], dim=1)
                cand_idx = torch.cat([topk_idx, torch.arange(s, e, device=self._torch_dev)[None, :].expand(B, -1)], dim=1)
                new_dist, new_pos = torch.topk(cand, k, dim=1, largest=False, sorted=True)
                new_idx = torch.gather(cand_idx, 1, new_pos)
                topk_dist, topk_idx = new_dist, new_idx
                s = e
            return topk_idx, topk_dist

    def _predict_trend_path_torch(self, Xn: np.ndarray) -> np.ndarray:
        assert _HAVE_TORCH, "PyTorch backend not available."
        with torch.no_grad():
            dev = self._torch_dev
            K = int(max(1, self.path_k))
            T = int(max(1, self.path_steps))
            step_size = float(self.path_step_size)
            radius = float(getattr(self, "_radius_", 1.0))
            sigma = float(self.path_gamma) * radius
            denom = 2.0 * (sigma * sigma) + 1e-12
            q_block = int(max(1, int(self._user_kwargs.get("torch_q_block", 1024))))

            xt = torch.from_numpy(Xn.astype(np.float32, copy=False)).to(dev, non_blocking=True)
            out = torch.empty((xt.shape[0],), dtype=torch.long, device=dev)

            for s in range(0, xt.shape[0], q_block):
                e = min(s + q_block, xt.shape[0])
                q = xt[s:e].contiguous()
                best_cls = torch.empty((q.shape[0],), dtype=torch.long, device=dev)
                best_score = torch.full((q.shape[0],), -1e30, device=dev)
                any_succ = torch.zeros((q.shape[0],), dtype=torch.bool, device=dev)

                for c in self._classes_t:
                    x = q.clone()
                    score = torch.zeros((q.shape[0],), device=dev)
                    success = torch.ones((q.shape[0],), dtype=torch.bool, device=dev)

                    for _ in range(T):
                        idxs, dists = self._torch_knn_topk(x, K)
                        cls_mask = (self._ty[idxs] == c)
                        have = cls_mask.any(dim=1)
                        success &= have
                        if not have.any():
                            break

                        # keep only class-c entries
                        ds = torch.where(cls_mask, dists, torch.full_like(dists, float("inf")))
                        good = ds.isfinite()
                        if not good.any():
                            success &= False
                            break

                        # pack valid columns to variable-length rows
                        ids_list, ds_list = [], []
                        for i in range(ds.shape[0]):
                            gi = good[i]
                            if gi.any():
                                ids_list.append(idxs[i][gi])
                                ds_list.append(ds[i][gi])
                            else:
                                ids_list.append(torch.empty((0,), dtype=torch.long, device=dev))
                                ds_list.append(torch.empty((0,), dtype=ds.dtype, device=dev))

                        from torch.nn.utils.rnn import pad_sequence
                        ids = pad_sequence(ids_list, batch_first=True, padding_value=-1)
                        ds = pad_sequence(ds_list, batch_first=True, padding_value=float("inf"))
                        mask = ids.ge(0)
                        if not mask.any():
                            success &= False
                            break

                        v = self._tvec[ids.clamp_min(0)]
                        en = self._teng[ids.clamp_min(0)]
                        sp = self._tstep[ids.clamp_min(0)].to(torch.int32)

                        en = torch.clamp(en, min=1e-9)
                        if self.step_weight_mode == "early":
                            w_step = (self.step_gamma ** sp).float()
                        elif self.step_weight_mode == "late":
                            w_step = (sp + 1).float()
                        else:
                            w_step = torch.ones_like(en)

                        w = (en * w_step * torch.exp(-ds / denom)).float()
                        w = torch.where(mask, w, torch.zeros_like(w))
                        w_sum = w.sum(dim=1)
                        ok = w_sum > 1e-12
                        success &= ok
                        if not ok.any():
                            break

                        v = torch.where(mask.unsqueeze(-1), v, torch.zeros_like(v))
                        v_hat = (w.unsqueeze(-1) * v).sum(dim=1) / w_sum.unsqueeze(-1)
                        x = x + step_size * v_hat
                        score = score + w_sum

                    upd = success & (score > best_score)
                    best_score = torch.where(upd, score, best_score)
                    best_cls = torch.where(upd, c, best_cls)
                    any_succ |= success

                fail = ~any_succ
                if fail.any():
                    idxs, _ = self._torch_knn_topk(q[fail], max(1, K))
                    maj = torch.mode(self._ty[idxs], dim=1).values
                    best_cls[fail] = maj

                out[s:e] = best_cls

            return out.detach().cpu().numpy()

    # ------------------------------ persistence -------------------------------
    def save(self, dir_path: str) -> None:
        import json
        from pathlib import Path

        if not hasattr(self, "_X_"):
            raise RuntimeError("Model has not been fitted; nothing to save.")

        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)

        param_keys = [
            "num_points","total_energy","mass","explosion_time","noise_scale","noise_decay",
            "adaptive_radius_mode","fixed_radius","use_radius_clip","dynamic_energy","dirichlet_alpha",
            "max_samples","random_state","points_schedule","energy_schedule","seed_mode","num_iters",
            "energy_decay","radius_growth","local_k","anisotropy","deflect_rival","deflect_strength",
            "momentum_conserve","shell_ratio","shell_jitter","path_steps","path_step_size","path_k",
            "path_gamma","step_weight_mode","step_gamma","governor",
            # backend params
            "platform","torch_q_block","torch_db_block",
        ]
        params = {k: getattr(self, k) for k in param_keys if hasattr(self, k)}
        cfg = {
            "version": "1.4-unified-backend",
            "dim": int(self._X_.shape[1]),
            "params": params,
            "backend": self._backend,
            "device": self._device,
            "rng_state": getattr(getattr(self, "_rng", None), "bit_generator", None).state
            if hasattr(getattr(self, "_rng", None), "bit_generator")
            else None,
        }
        with open(p / "config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        np.savez_compressed(
            p / "arrays.npz",
            X=self._X_, y=self._y_, classes=self._classes_, is_synth=self._is_synth_,
            point_energy=self._point_energy_, point_vec=self._point_vec_, point_step=self._point_step_,
            orig_X=getattr(self, "_orig_X_", None), orig_y=getattr(self, "_orig_y_", None),
            radius=np.float32(getattr(self, "_radius_", 1.0)), energy=np.float32(getattr(self, "_energy_", 1.0)),
        )
        if hasattr(self, "_backend") and self._backend == "hnsw" and hasattr(self, "_ann_index"):
            self._ann_index.save_index(str(p / "hnsw.bin"))
        # Torch backend has no separate index file (recomputed on load)

    @classmethod
    def load(cls, dir_path: str) -> "PyEGM":
        import json
        from pathlib import Path
        p = Path(dir_path)
        with open(p / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        data = np.load(p / "arrays.npz", allow_pickle=True)
        inst = cls(**cfg.get("params", {}))
        for k, v in cfg.get("params", {}).items():
            setattr(inst, k, v)

        inst._X_ = data["X"].astype(np.float32, copy=False)
        inst._y_ = data["y"]
        inst._classes_ = data["classes"]
        inst._is_synth_ = data["is_synth"]
        inst._point_energy_ = data["point_energy"].astype(np.float32, copy=False)
        inst._point_vec_ = data["point_vec"].astype(np.float32, copy=False)
        inst._point_step_ = data["point_step"].astype(np.int32, copy=False)
        inst._orig_X_ = data["orig_X"]
        inst._orig_y_ = data["orig_y"]
        inst._radius_ = float(data["radius"])
        inst._energy_ = float(data["energy"])

        inst._rng = np.random.default_rng()
        if cfg.get("rng_state") is not None:
            try:
                inst._rng.bit_generator.state = cfg["rng_state"]
            except Exception:
                pass

        # Recreate backend
        inst._backend = cfg.get("backend", None)
        inst._device = cfg.get("device", "cpu")
        if inst._backend == "hnsw":
            inst._build_nn_index(inst._X_.shape[1])
        else:
            if _HAVE_TORCH and (_HAVE_TORCH_CUDA or inst._device == "cpu"):
                inst._prepare_torch_bank()
            else:
                _warn("Torch backend not available on load; switching to CPU/HNSW.")
                inst._build_nn_index(inst._X_.shape[1])
                inst._backend = "hnsw"
                inst._device = "cpu"
        return inst

    # ---------------------- export / introspection / resume -------------------
    def export_fixed_yaml(self, path: str, include_meta: bool = True) -> str:
        if not hasattr(self, "_X_"):
            raise RuntimeError("Model not fitted. Call fit() before exporting YAML.")
        import yaml, time

        param_keys = [
            "num_points","total_energy","mass","explosion_time","noise_scale","noise_decay",
            "adaptive_radius_mode","fixed_radius","use_radius_clip","dynamic_energy","dirichlet_alpha",
            "max_samples","random_state","points_schedule","energy_schedule","seed_mode","num_iters",
            "energy_decay","radius_growth","local_k","anisotropy","deflect_rival","deflect_strength",
            "momentum_conserve","shell_ratio","shell_jitter","path_steps","path_step_size","path_k",
            "path_gamma","step_weight_mode","step_gamma","governor",
            "platform","torch_q_block","torch_db_block",
        ]
        data = {k: getattr(self, k) for k in param_keys if hasattr(self, k)}

        if include_meta:
            stats = None
            if hasattr(self, "_orig_y_") and self._orig_y_ is not None:
                y = self._orig_y_
                cls, cnt = np.unique(y, return_counts=True)
                stats = dict(
                    n_samples=int(len(y)),
                    n_classes=int(len(cls)),
                    min_per_class=int(cnt.min()) if len(cnt) else 0,
                    max_per_class=int(cnt.max()) if len(cnt) else 0,
                    imbalance_ratio=float(cnt.max() / max(1, cnt.min())) if len(cnt) else 1.0,
                )
            data["_meta"] = dict(
                note="Fixed snapshot of final hyperparameters after fit().",
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                preset_input=self._user_kwargs.get("preset", None),
                env_yaml=os.environ.get("PYEGM_DEFAULTS_FILE", None),
                dataset_stats=stats,
                backend=self._backend,
                device=self._device,
            )

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        return path

    def get_fitted_params(self) -> dict:
        if not hasattr(self, "_X_"):
            raise RuntimeError("Model not fitted.")
        keys = [
            "num_points","total_energy","mass","explosion_time","noise_scale","noise_decay",
            "adaptive_radius_mode","fixed_radius","use_radius_clip","dynamic_energy","dirichlet_alpha",
            "max_samples","random_state","points_schedule","energy_schedule","seed_mode","num_iters",
            "energy_decay","radius_growth","local_k","anisotropy","deflect_rival","deflect_strength",
            "momentum_conserve","shell_ratio","shell_jitter","path_steps","path_step_size","path_k",
            "path_gamma","step_weight_mode","step_gamma","governor",
            "platform","torch_q_block","torch_db_block",
        ]
        return {k: getattr(self, k) for k in keys if hasattr(self, k)}

    def continue_fit(self, X=None, y=None, extra_iters: int = 1, reseed: int | None = None):
        if not hasattr(self, "_X_"):
            raise RuntimeError("Model not fitted. Call fit() first.")
        if X is None or y is None:
            if not hasattr(self, "_orig_X_") or self._orig_X_ is None:
                raise RuntimeError("No cached training data available for resume.")
            X, y = self._orig_X_, self._orig_y_

        old_iters = getattr(self, "num_iters", 1)
        try:
            if extra_iters is None or extra_iters <= 0:
                return self
            self.num_iters = int(extra_iters)
            if reseed is not None:
                self.random_state = int(reseed)
            return self.partial_fit(X, y)
        finally:
            self.num_iters = old_iters

    # ----------------------------- visualization ------------------------------
    def visualize_explosion(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        *,
        title: str = "PyEGM • Live Synthesis Viewer",
        save_path: Optional[str] = None,  # ".mp4" / ".gif" / folder for PNG frames
        fps: int = 12,
        dpi: int = 120,
        figsize=(7.6, 5.8),
        point_size: int = 36,
        alpha_orig: float = 0.90,
        alpha_synth: float = 0.85,
        show_vectors: bool = True,
        enable_legend: bool = True,
        enable_slider: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
            import matplotlib as mpl
            from sklearn.decomposition import PCA
            from matplotlib.widgets import Slider, Button
        except Exception as e:
            raise RuntimeError(
                "Visualization requires matplotlib (and optionally ffmpeg/Pillow for export). "
                "Please install matplotlib first."
            ) from e

        if X is None or y is None:
            if not hasattr(self, "_orig_X_") or self._orig_X_ is None:
                raise RuntimeError("No training data available. Provide X,y or call fit() first.")
            X, y = self._orig_X_, self._orig_y_
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        viz_rng = np.random.default_rng(self.random_state if random_state is None else random_state)

        classes = np.unique(y)
        self._classes_ = classes
        self._radius_ = self._adaptive_radius(X, np.searchsorted(classes, y))
        self._energy_ = self._rescale_total_energy(self.total_energy, self._radius_)

        centers_all = self._class_stats(X, y)
        per_class_counts = np.array([np.sum(y == c) for c in classes], dtype=np.int32)
        per_class_weights = np.ones_like(per_class_counts, dtype=np.float32)
        if self.points_schedule == "inv_size":
            per_class_weights = 1.0 / np.maximum(per_class_counts, 1)
        elif self.points_schedule == "sqrt_inv_size":
            per_class_weights = 1.0 / np.maximum(np.sqrt(per_class_counts), 1.0)
        per_class_weights = per_class_weights / per_class_weights.sum()

        base_energy = self._energy_
        seeds_X = X
        seeds_y = y
        noise0 = self.noise_scale

        step_frames: list[dict] = []
        for t in range(max(1, self.num_iters)):
            radius_t = self._radius_ * (self.radius_growth ** t)
            energy_t = base_energy * (self.energy_decay ** t)
            noise_t = noise0 * (self.noise_decay ** t)

            class_centers = self._class_stats(seeds_X, seeds_y) if self.seed_mode == "per_sample" else centers_all

            step_X, step_y, step_V = [], [], []
            for ci, c in enumerate(classes):
                class_pts = seeds_X[seeds_y == c] if self.seed_mode == "per_sample" else X[y == c]
                if len(class_pts) == 0:
                    continue

                seeds_for_class = class_pts if self.seed_mode == "per_sample" else class_centers[ci][None, :]
                n_synth_c = max(1, int(round(self.num_points * per_class_weights[ci])))
                energies = viz_rng.dirichlet([self.dirichlet_alpha] * n_synth_c).astype(np.float32)
                energies *= energy_t

                rival_centers = (
                    np.stack([centers_all[j] for j, cc in enumerate(classes) if cc != c], axis=0)
                    if len(classes) > 1
                    else None
                )
                rival_vec = None

                e_ptr = 0
                while e_ptr < n_synth_c:
                    seed_id = viz_rng.integers(0, len(seeds_for_class))
                    seed = seeds_for_class[seed_id].astype(np.float32, copy=False)

                    center_local, cov_local = self._local_center_cov(
                        class_pts.astype(np.float32, copy=False), seed, self.local_k
                    )

                    u = seed - center_local
                    nu = np.linalg.norm(u)
                    u = (u / nu).astype(np.float32) if nu >= 1e-9 else viz_rng.normal(size=X.shape[1]).astype(np.float32)

                    if rival_centers is not None and self.deflect_rival:
                        rv = rival_centers - seed[None, :]
                        d = np.linalg.norm(rv, axis=1)
                        if len(d):
                            rid = np.argmin(d)
                            rdir = rv[rid] / (np.linalg.norm(rv[rid]) + 1e-9)
                            rival_vec = rdir.astype(np.float32, copy=False)

                    if self.anisotropy > 0.0:
                        w, V = np.linalg.eigh(cov_local + 1e-6 * np.eye(cov_local.shape[0], dtype=np.float32))
                        w = np.maximum(w, 1e-6)
                        scale = (w ** (-0.5 * self.anisotropy)).astype(np.float32, copy=False)
                        M = V @ np.diag(scale) @ V.T
                        u = M @ u
                    if self.deflect_rival and rival_vec is not None:
                        proj = np.dot(u, rival_vec) * rival_vec
                        u = u - self.deflect_strength * proj
                    u = u / (np.linalg.norm(u) + 1e-9)

                    remaining = n_synth_c - e_ptr
                    pair = self.momentum_conserve and remaining >= 2
                    take = 2 if pair else 1
                    Es = energies[e_ptr: e_ptr + take]

                    for k in range(take):
                        e = float(Es[k])
                        v_mag = np.sqrt(2.0 * e / self.mass) * self.explosion_time
                        direction = u if k == 0 else (-u)
                        disp = v_mag * direction
                        dnorm = np.linalg.norm(disp)
                        if self.use_radius_clip and dnorm > radius_t:
                            disp = disp * (radius_t / (dnorm + 1e-9))
                        shell = viz_rng.normal(scale=self.shell_jitter)
                        disp = disp * (1.0 - self.shell_ratio + (self.shell_ratio + shell) * viz_rng.random())
                        point = seed + disp + viz_rng.normal(scale=noise_t, size=X.shape[1]).astype(np.float32)

                        step_X.append(point.astype(np.float32, copy=False))
                        step_y.append(int(c))
                        step_V.append(disp.astype(np.float32, copy=False))

                    e_ptr += take

            if len(step_X):
                step_X = np.asarray(step_X, dtype=np.float32)
                step_y = np.asarray(step_y, dtype=classes.dtype)
                step_V = np.asarray(step_V, dtype=np.float32)
            else:
                step_X = np.empty((0, X.shape[1]), dtype=np.float32)
                step_y = np.empty((0,), dtype=classes.dtype)
                step_V = np.empty((0, X.shape[1]), dtype=np.float32)

            step_frames.append({"X": step_X, "y": step_y, "V": step_V})
            seeds_X, seeds_y = step_X, step_y
            if len(seeds_X) == 0:
                break

        total_steps = len(step_frames)

        all_concat = [X] + [sf["X"] for sf in step_frames if sf["X"].size > 0]
        X_all = np.vstack(all_concat) if all_concat else X
        pca = PCA(n_components=2, random_state=self.random_state or 0)
        pca.fit(X_all)

        X2 = pca.transform(X) if X.shape[1] >= 2 else np.pad(X, ((0, 0), (0, 2 - X.shape[1])))

        frames2 = []
        for sf in step_frames:
            Xi = sf["X"]
            Vi = sf["V"]
            if len(Xi):
                Xi2 = pca.transform(Xi)
                Vi2 = Vi @ pca.components_.T
            else:
                Xi2 = np.empty((0, 2), dtype=np.float32)
                Vi2 = np.empty((0, 2), dtype=np.float32)
            frames2.append({"X2": Xi2, "y": sf["y"], "V2": Vi2})

        cmap = mpl.cm.get_cmap("tab10", max(10, len(classes)))
        class_to_color = {int(c): cmap(i % 10) for i, c in enumerate(classes)}

        fig, ax = plt.subplots(figsize=figsize)
        try:
            fig.canvas.manager.set_window_title(title)
        except Exception:
            pass
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.15)

        ax.scatter(
            X2[:, 0], X2[:, 1],
            s=point_size, c=[class_to_color[int(c)] for c in y],
            alpha=alpha_orig, label="Original"
        )

        sc_synth = ax.scatter([], [], s=point_size, c=[], alpha=alpha_synth, label="Synthetic ≤ step")
        quiv = ax.quiver([], [], [], [], angles='xy', scale_units='xy', scale=1.0, alpha=0.45, color="gray")

        if enable_legend:
            import matplotlib.lines as mlines
            legend_handles = []
            for c in classes[:10]:
                patch = mlines.Line2D([], [], linestyle='none', marker='o', markersize=6,
                                      markerfacecolor=class_to_color[int(c)], markeredgecolor='none',
                                      label=f"class {int(c)}")
                legend_handles.append(patch)
            legend_handles.append(mlines.Line2D([], [], linestyle='none', marker='o', markersize=6,
                                               markerfacecolor='k', markeredgecolor='none', alpha=0.3, label="Original"))
            legend_handles.append(mlines.Line2D([], [], linestyle='none', marker='o', markersize=6,
                                               markerfacecolor='k', markeredgecolor='none', alpha=0.7, label="Synthetic"))
            legend_handles.append(mlines.Line2D([], [], color='gray', lw=2, label="Vectors"))
            ax.legend(handles=legend_handles, loc="upper right", frameon=True)

        if enable_slider:
            plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.15)
            ax_slider = plt.axes([0.10, 0.06, 0.70, 0.04])
            ax_button = plt.axes([0.82, 0.055, 0.10, 0.05])
            slider = Slider(ax_slider, "Step", 0, max(1, total_steps), valinit=0, valstep=1, color="#5B8FF9")
            btn = Button(ax_button, "Play / Pause")
            playing = {"run": True}

            def on_changed(val):
                set_step(int(val))
            slider.on_changed(on_changed)

            def on_click(event):
                playing["run"] = not playing["run"]
            btn.on_clicked(on_click)

            def on_key(event):
                if event.key == " ":
                    playing["run"] = not playing["run"]
                elif event.key == "left" and slider.val > 0:
                    slider.set_val(slider.val - 1)
                elif event.key == "right" and slider.val < max(1, total_steps):
                    slider.set_val(slider.val + 1)
            fig.canvas.mpl_connect("key_press_event", on_key)
        else:
            slider = None
            playing = {"run": True}

        def collect_upto(k: int):
            if k <= 0 or total_steps == 0:
                return np.empty((0, 2), dtype=np.float32), np.array([], dtype=classes.dtype), np.empty((0, 2))
            Xc, yc, Vc = [], [], []
            for i in range(min(k, total_steps)):
                fr = frames2[i]
                if fr["X2"].size:
                    Xc.append(fr["X2"])
                    yc.append(fr["y"])
                    Vc.append(fr["V2"])
            Xc = np.vstack(Xc) if Xc else np.empty((0, 2), dtype=np.float32)
            yc = np.concatenate(yc) if yc else np.array([], dtype=classes.dtype)
            Vc = np.vstack(Vc) if Vc else np.empty((0, 2), dtype=np.float32)
            return Xc, yc, Vc

        def set_step(k: int):
            Xc, yc, Vc = collect_upto(k)
            if len(Xc):
                sc_synth.set_offsets(Xc)
                sc_synth.set_color([class_to_color[int(c)] for c in yc])
            else:
                sc_synth.set_offsets(np.empty((0, 2)))
                sc_synth.set_color([])
            if 0 < k <= total_steps and show_vectors:
                fr = frames2[k - 1]
                Xi = fr["X2"]
                Vi = fr["V2"]
                quiv.set_offsets(Xi)
                if Vi.size:
                    quiv.set_UVC(Vi[:, 0], Vi[:, 1])
                else:
                    quiv.set_UVC([], [])
            else:
                quiv.set_offsets([])
                quiv.set_UVC([], [])
            ax.set_title(f"{title}  |  step {k}/{total_steps}")
            fig.canvas.draw_idle()

        set_step(0)

        def _update(frame):
            if not playing["run"]:
                return sc_synth
            if enable_slider:
                k = int(slider.val) + 1
                if k > max(1, total_steps):
                    k = 0
                slider.set_val(k)
            else:
                k = (frame % (max(1, total_steps) + 1))
                set_step(k)
            return sc_synth

        anim = FuncAnimation(fig, _update, interval=int(1000 / max(1, fps)), blit=False)

        if save_path:
            try:
                ext = os.path.splitext(save_path)[1].lower()
                if ext == ".mp4":
                    writer = FFMpegWriter(fps=fps, bitrate=1800)
                    anim.save(save_path, writer=writer, dpi=dpi)
                elif ext == ".gif":
                    writer = PillowWriter(fps=fps)
                    anim.save(save_path, writer=writer, dpi=dpi)
                else:
                    outdir = save_path
                    os.makedirs(outdir, exist_ok=True)
                    for k in range(0, max(1, total_steps) + 1):
                        set_step(k)
                        fig.savefig(os.path.join(outdir, f"frame_{k:03d}.png"), dpi=dpi)
            except Exception as e:
                print(f"[WARN] Failed to export animation ({e}). Showing window only.")

        import matplotlib.pyplot as plt  # re-import for clarity
        plt.show()

    def watch_explosion(self, *args, **kwargs):
        return self.visualize_explosion(*args, **kwargs)
