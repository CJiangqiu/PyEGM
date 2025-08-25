# filename: pyegm.py
# -*- coding: utf-8 -*-
from typing import Literal, Optional, Sequence, Tuple
import numpy as np
import hnswlib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_X_y

from .defaults import get_default, DEFAULTS

Vector = np.ndarray


class PyEGM(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        *,
        # --------- all hyperparameters (None -> load from config) ----------
        num_points: Optional[int] = None,
        total_energy: Optional[float] = None,
        mass: Optional[float] = None,
        explosion_time: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_decay: Optional[float] = None,
        adaptive_radius_mode: Optional[Literal["local", "global", None]] = None,
        fixed_radius: Optional[float] = None,
        use_radius_clip: Optional[bool] = None,
        dynamic_energy: Optional[bool] = None,
        dirichlet_alpha: Optional[float] = None,
        max_samples: Optional[int] = None,
        random_state: Optional[int] = None,
        points_schedule: Optional[str] = None,
        energy_schedule: Optional[str] = None,
        seed_mode: Optional[Literal["per_sample", "centroid"]] = None,
        num_iters: Optional[int] = None,
        energy_decay: Optional[float] = None,
        radius_growth: Optional[float] = None,
        local_k: Optional[int] = None,
        anisotropy: Optional[float] = None,
        deflect_rival: Optional[bool] = None,
        deflect_strength: Optional[float] = None,
        momentum_conserve: Optional[bool] = None,
        shell_ratio: Optional[float] = None,
        shell_jitter: Optional[float] = None,
        path_steps: Optional[int] = None,
        path_step_size: Optional[float] = None,
        path_k: Optional[int] = None,
        path_gamma: Optional[float] = None,
        step_weight_mode: Optional[Literal["none", "early", "late"]] = None,
        step_gamma: Optional[float] = None,
        governor: Optional[bool] = None,
    ) -> None:
        """
        Explosive Generative Model (PyEGM)

        Default values are loaded from an embedded YAML file (pyegm/data/best_rules.yaml).
        If a parameter is provided explicitly, it overrides the config.
        If a key is missing in YAML, we fall back to safe hard-coded defaults.

        See pyegm.defaults for details.
        """
        # Helper to resolve a param: user arg > YAML > hard default
        def _resolve(name: str, val):
            return val if val is not None else get_default(name)

        # Resolve everything
        self.num_points = int(_resolve("num_points", num_points))
        self.total_energy = float(_resolve("total_energy", total_energy))
        self.mass = float(_resolve("mass", mass))
        self.explosion_time = float(_resolve("explosion_time", explosion_time))
        self.noise_scale = float(_resolve("noise_scale", noise_scale))
        self.noise_decay = float(_resolve("noise_decay", noise_decay))

        self.adaptive_radius_mode = _resolve("adaptive_radius_mode", adaptive_radius_mode)
        self.fixed_radius = _resolve("fixed_radius", fixed_radius)
        self.use_radius_clip = bool(_resolve("use_radius_clip", use_radius_clip))
        self.dynamic_energy = bool(_resolve("dynamic_energy", dynamic_energy))
        self.dirichlet_alpha = float(_resolve("dirichlet_alpha", dirichlet_alpha))

        self.max_samples = int(_resolve("max_samples", max_samples))
        self.random_state = _resolve("random_state", random_state)

        self.points_schedule = _resolve("points_schedule", points_schedule)
        self.energy_schedule = _resolve("energy_schedule", energy_schedule)
        self.seed_mode = _resolve("seed_mode", seed_mode)
        self.num_iters = int(_resolve("num_iters", num_iters))

        self.energy_decay = float(_resolve("energy_decay", energy_decay))
        self.radius_growth = float(_resolve("radius_growth", radius_growth))
        self.local_k = int(_resolve("local_k", local_k))
        self.anisotropy = float(_resolve("anisotropy", anisotropy))

        self.deflect_rival = bool(_resolve("deflect_rival", deflect_rival))
        self.deflect_strength = float(_resolve("deflect_strength", deflect_strength))
        self.momentum_conserve = bool(_resolve("momentum_conserve", momentum_conserve))

        self.shell_ratio = float(_resolve("shell_ratio", shell_ratio))
        self.shell_jitter = float(_resolve("shell_jitter", shell_jitter))

        self.path_steps = int(_resolve("path_steps", path_steps))
        self.path_step_size = float(_resolve("path_step_size", path_step_size))
        self.path_k = int(_resolve("path_k", path_k))
        self.path_gamma = float(_resolve("path_gamma", path_gamma))

        self.step_weight_mode = _resolve("step_weight_mode", step_weight_mode)
        self.step_gamma = float(_resolve("step_gamma", step_gamma))

        self.governor = bool(_resolve("governor", governor))

    # ----------------------------------------------------------------------
    # Fit / partial_fit / predict  (unchanged functional logic)
    # ----------------------------------------------------------------------
    def fit(self, X: Vector, y: Sequence[int]):
        X, y = check_X_y(X, y)
        X = X.astype(np.float32, copy=False)
        self._rng = np.random.default_rng(self.random_state)
        self._classes_, indices = np.unique(y, return_inverse=True)
        self._orig_X_ = X.copy()
        self._orig_y_ = np.asarray(y).copy()
        self._radius_ = self._adaptive_radius(X, indices)
        self._energy_ = self._rescale_total_energy(self.total_energy, self._radius_)

        synth_X, synth_y, synth_E, synth_V, synth_S = self._multi_iter_generate(X, y)
        Xw = np.vstack([X, synth_X]) if len(synth_X) else X
        yw = np.concatenate([y, synth_y]) if len(synth_y) else np.asarray(y)
        is_synth = (
            np.concatenate([np.zeros(len(X), dtype=bool), np.ones(len(synth_X), dtype=bool)])
            if len(synth_X)
            else np.zeros(len(X), dtype=bool)
        )
        point_energy = (
            np.concatenate([np.full(len(X), 1e-9, dtype=np.float32), synth_E])
            if len(synth_X)
            else np.full(len(X), 1e-9, dtype=np.float32)
        )
        zero_V = np.zeros((len(X), X.shape[1]), dtype=np.float32)
        point_vec = np.vstack([zero_V, synth_V]) if len(synth_X) else zero_V
        point_step = (
            np.concatenate([np.zeros(len(X), dtype=np.int32), synth_S])
            if len(synth_X)
            else np.zeros(len(X), dtype=np.int32)
        )

        self._X_, self._y_ = Xw, yw
        self._is_synth_ = is_synth
        self._point_energy_ = point_energy
        self._point_vec_ = point_vec
        self._point_step_ = point_step

        self._build_nn_index(self._X_.shape[1])
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

        self._X_ = np.vstack([self._X_, X, synth_X])
        self._y_ = np.concatenate([self._y_, y, synth_y])
        self._is_synth_ = np.concatenate(
            [self._is_synth_, np.zeros(len(X), dtype=bool), np.ones(len(synth_X), dtype=bool)]
        )
        self._point_energy_ = np.concatenate(
            [self._point_energy_, np.full(len(X), 1e-9, dtype=np.float32), synth_E]
        )
        zero_V = np.zeros((len(X), self._X_.shape[1]), dtype=np.float32)
        self._point_vec_ = np.vstack([self._point_vec_, zero_V, synth_V])
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

        self._build_nn_index(self._X_.shape[1])
        return self

    def predict(self, X: Vector) -> np.ndarray:
        X = check_array(X).astype(np.float32, copy=False)
        if not hasattr(self, "_ann_index"):
            raise RuntimeError("Model has not been fitted.")
        return self._predict_trend_path(X)

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------
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

    def _class_stats(self, X: Vector, y: np.ndarray):
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

    def _local_center_cov(self, class_pts: Vector, seed: Vector, k: int):
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

    def _shape_direction(
        self,
        u: Vector,
        cov: Vector,
        seed: Vector,
        class_center: Vector,
        rival_dir: Optional[Vector],
    ):
        if self.anisotropy > 0.0:
            w, V = np.linalg.eigh(cov + 1e-6 * np.eye(cov.shape[0], dtype=np.float32))
            w = np.maximum(w, 1e-6)
            scale = (w ** (-0.5 * self.anisotropy)).astype(np.float32, copy=False)
            M = V @ np.diag(scale) @ V.T
            u = M @ u
        if self.deflect_rival and rival_dir is not None:
            proj = np.dot(u, rival_dir) * rival_dir
            u = u - self.deflect_strength * proj
        nrm = np.linalg.norm(u) + 1e-9
        return (u / nrm).astype(np.float32, copy=False)

    def _multi_iter_generate(
        self, X: Vector, y: Sequence[int]
    ) -> Tuple[Vector, np.ndarray, np.ndarray, Vector, np.ndarray]:
        y = np.asarray(y)
        dim = X.shape[1]
        classes = self._classes_
        centers_all = self._class_stats(X, y)
        synth_X, synth_y, synth_E, synth_V, synth_S = [], [], [], [], []

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

                if self.seed_mode == "per_sample":
                    seeds_for_class = class_pts
                else:
                    seeds_for_class = class_centers[ci][None, :]

                n_synth_c = max(1, int(round(self.num_points * per_class_weights[ci])))
                energies = self._rng.dirichlet([self.dirichlet_alpha] * n_synth_c).astype(np.float32)
                energies *= energy_t

                rival_centers = np.stack([centers_all[j] for j, cc in enumerate(classes) if cc != c], axis=0) \
                    if len(classes) > 1 else None
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
                    if nu < 1e-9:
                        u = self._rng.normal(size=dim).astype(np.float32)
                    else:
                        u = (u / nu).astype(np.float32)

                    if rival_centers is not None and self.deflect_rival:
                        rv = rival_centers - seed[None, :]
                        d = np.linalg.norm(rv, axis=1)
                        if len(d):
                            rid = np.argmin(d)
                            rdir = rv[rid] / (np.linalg.norm(rv[rid]) + 1e-9)
                            rival_vec = rdir.astype(np.float32, copy=False)

                    u = self._shape_direction(u, cov_local, seed, center_local, rival_vec)

                    remaining = n_synth_c - e_ptr
                    pair = self.momentum_conserve and remaining >= 2
                    take = 2 if pair else 1
                    Es = energies[e_ptr : e_ptr + take]

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
                        synth_y.append(c)
                        synth_E.append(np.float32(e))
                        synth_V.append(disp.astype(np.float32, copy=False))
                        synth_S.append(np.int32(t + 1))

                    e_ptr += take

            seeds_X = np.asarray(synth_X, dtype=np.float32)
            seeds_y = np.asarray(synth_y, dtype=classes.dtype)
            if len(seeds_X) == 0:
                break

        if len(synth_X) == 0:
            return (
                np.empty((0, dim), dtype=np.float32),
                np.empty((0,), dtype=classes.dtype),
                np.empty((0,), dtype=np.float32),
                np.empty((0, dim), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        return (
            np.asarray(synth_X, dtype=np.float32),
            np.asarray(synth_y, dtype=classes.dtype),
            np.asarray(synth_E, dtype=np.float32),
            np.asarray(synth_V, dtype=np.float32),
            np.asarray(synth_S, dtype=np.int32),
        )

    def _build_nn_index(self, dim: int) -> None:
        self._rng = np.random.default_rng(self.random_state)
        self._ann_index = hnswlib.Index(space="l2", dim=dim)
        self._ann_index.init_index(
            max_elements=len(self._X_), M=16, ef_construction=200, random_seed=self.random_state or 0
        )
        self._ann_index.add_items(self._X_, np.arange(len(self._X_)))
        self._ann_index.set_ef(50)

    def _step_weight(self, steps: np.ndarray) -> np.ndarray:
        if self.step_weight_mode == "early":
            return (self.step_gamma ** steps).astype(np.float32)
        if self.step_weight_mode == "late":
            return (steps + 1).astype(np.float32)
        return np.ones_like(steps, dtype=np.float32)

    def _predict_trend_path(self, X: Vector) -> np.ndarray:
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
                    idxs, dists = self._ann_index.knn_query(x[None, :], k=K, filter=None)
                    ids = idxs[0]
                    ds = dists[0]
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
                    w_step = self._step_weight(sp)
                    w = (en * w_step * np.exp(-(ds * ds) / denom)).astype(np.float32, copy=False)
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
                out[i] = best_cls
            else:
                idxs, _ = self._ann_index.knn_query(x0[None, :], k=max(1, K), filter=None)
                ids = idxs[0]
                cls, cnts = np.unique(self._y_[ids], return_counts=True)
                out[i] = cls[np.argmax(cnts)]

        return out

    def score(
        self, X: Vector, y: Sequence[int], sample_weight: Optional[Sequence[float]] = None
    ) -> float:
        return float(np.average(self.predict(X) == y, weights=sample_weight))
