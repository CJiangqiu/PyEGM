# -*- coding: utf-8 -*-
"""
PyEGM — Physics-Inspired Exemplar Growth Model (GPU-Optional)

A lightweight classifier for fixed-feature protocols. Each class is represented
by a prototype (class mean) and a set of arrival centers placed on concentric
shells along deterministic rays. Prediction blends a prototype kernel and an
arrival-center kernel.

Optional acceleration: if PyTorch is installed and CUDA is available, the
matrix computations inside scoring are executed on GPU. Otherwise, the model
falls back to NumPy/CPU with identical results.

Scikit-learn–style interface:
- fit(X, y)
- partial_fit(X, y, classes=None)
- predict(X)
- score(X, y, sample_weight=None)
- save(dir_path) / load(dir_path)
- get_fitted_params()

Notation
--------
Score_c(x) = β0 · κ(x, μ_c; τ0) + (1 − β0) · Σ_s w_s · (1/M) Σ_m κ(x, z_{c,s,m}; τ0)

μ_c: class prototype. z_{c,s,m}: arrival center for class c, shell s, ray m.
κ: cosine (temperature) or RBF over L2 distance. w_s: nonnegative shell weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import json
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

# Optional torch for GPU inference
try:
    import torch
except Exception:
    torch = None  # type: ignore

# ============================= Configuration ==============================

@dataclass
class ExplosionConfig:
    """
    Hyperparameters for the model.

    Parameters
    ----------
    metric : {"cos","l2"}, default="cos"
        Scoring kernel type. "cos" uses temperature-scaled cosine similarity.
        "l2" uses an RBF over squared L2 distances.

    normalize : bool, default=True
        Apply row-wise L2 normalization to inputs and class means when
        metric="cos".

    num_shells : int, default=1
        Number of concentric shells (S) used to place arrival centers.

    num_rays : int, default=9
        Number of rays per shell (M). Rays are deterministic given the seed.

    alpha : float, default=1.5005390509141003
        Base radial scale before shell growth.

    gamma : float, default=2.13404351595625
        Shell growth factor. The s-th shell radius scales as
        r_s = alpha * scale * gamma^s.

    eta : float, default=1.3577951967090702
        Radial gain applied after anisotropic whitening by per-dimension std.

    tau0 : float, default=0.1250617147506319
        Temperature for the cosine kernel.

    beta0 : float in [0,1], default=0.7357649153648804
        Mixing weight between the prototype channel and arrival-center channel.
        1.0 reduces to a prototype-only classifier (NCM).

    l2_sigma_scale : float, default=1.0
        Width scale for the RBF when metric="l2". The per-class σ is derived
        from class statistics and multiplied by this scale.

    cache_centers : bool, default=True
        If True, precompute and cache arrival centers for all shells.

    random_state : int, default=4858
        Seed for deterministic ray directions.

    platform : {"auto","cpu","cuda"}, default="auto"
        Device preference for scoring. "auto" selects "cuda" when PyTorch is
        available and CUDA is detected; otherwise uses "cpu".
    """

    # Defaults replaced with a searched configuration
    metric: Literal["cos", "l2"] = "cos"
    normalize: bool = True
    num_shells: int = 1
    num_rays: int = 9
    alpha: float = 1.5005390509141003
    gamma: float = 2.13404351595625
    eta: float = 1.3577951967090702
    tau0: float = 0.1250617147506319
    beta0: float = 0.7357649153648804
    l2_sigma_scale: float = 1.0
    cache_centers: bool = True
    random_state: int = 4858
    platform: Literal["auto", "cpu", "cuda"] = "auto"

    def __post_init__(self):
        self.metric = str(self.metric).lower()
        if self.metric not in ("cos", "l2"):
            raise ValueError("metric must be 'cos' or 'l2'.")
        self.num_shells = int(max(0, self.num_shells))
        self.num_rays = int(max(0, self.num_rays))
        self.alpha = float(max(0.0, self.alpha))
        self.gamma = float(max(1.0, self.gamma))
        self.eta = float(max(0.0, self.eta))
        self.tau0 = float(max(1e-6, self.tau0))
        self.beta0 = float(min(1.0, max(0.0, self.beta0)))
        self.l2_sigma_scale = float(max(1e-6, self.l2_sigma_scale))
        self.platform = str(self.platform).lower()
        if self.platform not in ("auto", "cpu", "cuda"):
            self.platform = "auto"

# ================================ Utilities ===============================

def _normalize_rows(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Return row-wise L2-normalized array."""
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / n

def _ensure_dir(p: Path) -> None:
    """Create directory if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)

# ================================== Model =================================

class PyEGM(BaseEstimator, ClassifierMixin):
    """
    Classifier with a prototype channel and an arrival-center channel.

    Construction
    ------------
    PyEGM(config: Optional[ExplosionConfig] = None, random_state: Optional[int] = None)

    Parameters
    ----------
    config : ExplosionConfig or None
        Configuration object. When None, defaults are used.

    random_state : int or None
        Optional seed that overrides the seed in `config` for reproducibility.

    Notes
    -----
    - The model operates on fixed features; no backbone is trained here.
    - If PyTorch with CUDA is available and `config.platform` allows GPU,
      scoring uses GPU linear algebra. Otherwise, NumPy/CPU is used.
    """

    # ---- construction & state ----
    def __init__(self, config: Optional[ExplosionConfig] = None, random_state: Optional[int] = None):
        self.cfg = config if config is not None else ExplosionConfig()
        if random_state is not None:
            self.cfg.random_state = int(random_state)

        # Fitted state
        self._classes: Optional[np.ndarray] = None      # (C,)
        self._n: Optional[np.ndarray] = None            # (C,)
        self._mean: Optional[np.ndarray] = None         # (C,D)
        self._M2: Optional[np.ndarray] = None           # (C,D)
        self._std: Optional[np.ndarray] = None          # (C,D)
        self._scale: Optional[np.ndarray] = None        # (C,)
        self._centers_flat: Optional[List[Dict[str, np.ndarray]]] = None  # per shell
        self._dim: Optional[int] = None

        # Runtime device and torch caches (built when CUDA is used)
        self._device: Literal["cpu", "cuda"] = self._pick_device(self.cfg.platform)
        self._mean_t: Optional["torch.Tensor"] = None
        self._sigma_t: Optional["torch.Tensor"] = None           # per-class sigma for L2 path
        self._centers_flat_t: Optional[List[Dict[str, "torch.Tensor"]]] = None

    # ------------------------ device selection helpers ------------------------

    @staticmethod
    def _pick_device(platform: Literal["auto", "cpu", "cuda"]) -> Literal["cpu", "cuda"]:
        """Return 'cuda' if allowed and available, otherwise 'cpu'."""
        if platform == "cuda":
            return "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        if platform == "auto":
            return "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        return "cpu"

    def _maybe_build_torch_cache(self) -> None:
        """
        Prepare Torch tensors for fast scoring on GPU.

        This mirrors key NumPy buffers on the selected CUDA device. When torch
        is not available or the device is CPU, the caches are cleared.
        """
        if torch is None or self._device != "cuda":
            self._mean_t = None
            self._sigma_t = None
            self._centers_flat_t = None
            return
        if self._mean is None:
            return

        # Prototypes
        self._mean_t = torch.tensor(self._mean, device="cuda", dtype=torch.float32)

        # Per-class sigma for L2 kernel
        if self._scale is not None:
            sigma = (self._scale * float(self.cfg.l2_sigma_scale)).astype(np.float32)  # (C,)
            self._sigma_t = torch.tensor(sigma, device="cuda", dtype=torch.float32)
        else:
            self._sigma_t = None

        # Arrival centers per shell
        self._centers_flat_t = None
        if self._centers_flat is not None and len(self._centers_flat) > 0:
            out: List[Dict[str, "torch.Tensor"]] = []
            for rec in self._centers_flat:
                Z = torch.tensor(rec["Z"], device="cuda", dtype=torch.float32)          # (C*M, D)
                Z_norm = torch.tensor(rec["Z_norm"], device="cuda", dtype=torch.float32)
                item: Dict[str, "torch.Tensor"] = {"Z": Z, "Z_norm": Z_norm}
                if self.cfg.metric == "l2":
                    item["Z_sq"] = torch.sum(Z * Z, dim=1)                              # (C*M,)
                out.append(item)
            self._centers_flat_t = out

    # ------------------------------ helpers for stats ------------------------------

    def _prepare_space(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize or pass through the input space depending on the metric.

        Parameters
        ----------
        X : ndarray of shape (N, D)
            Input feature matrix.

        Returns
        -------
        ndarray of shape (N, D)
            Preprocessed features ready for scoring.
        """
        X = np.asarray(X, dtype=np.float32)
        if self.cfg.metric == "cos" and self.cfg.normalize:
            return _normalize_rows(X)
        return X

    @staticmethod
    def _batch_stats(X: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Compute per-class batch statistics.

        Parameters
        ----------
        X : ndarray of shape (n_i, D)
            Samples of a single class.

        Returns
        -------
        n : int
            Number of samples.
        mu : ndarray of shape (D,)
            Mean vector.
        M2 : ndarray of shape (D,)
            Sum of squared deviations for variance computation.
        """
        n = X.shape[0]
        if n == 0:
            raise ValueError("Empty class batch.")
        mu = X.mean(axis=0, keepdims=False).astype(np.float32)
        diff = X - mu
        M2 = (diff * diff).sum(axis=0).astype(np.float32)
        return n, mu, M2

    @staticmethod
    def _combine_stats(
        n0: int, mu0: np.ndarray, M20: np.ndarray,
        n1: int, mu1: np.ndarray, M21: np.ndarray
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Combine two sets of Welford statistics.

        Parameters
        ----------
        n0, mu0, M20 : current counts, mean, and M2
        n1, mu1, M21 : incoming counts, mean, and M2

        Returns
        -------
        n, mu, M2 : updated statistics after aggregation.
        """
        if n0 == 0:
            return n1, mu1, M21
        if n1 == 0:
            return n0, mu0, M20
        n = n0 + n1
        delta = mu1 - mu0
        mu = mu0 + delta * (n1 / max(1.0, n))
        M2 = M20 + M21 + (delta * delta) * (n0 * n1 / max(1.0, n))
        return int(n), mu.astype(np.float32), M2.astype(np.float32)

    def _derive_std_scale(self) -> None:
        """
        Derive per-class std vectors and scalar scales.

        std[c, d] is the standard deviation per dimension for class c.
        scale[c] is a scalar summary (mean std) used by the radial schedule.
        """
        assert self._n is not None and self._M2 is not None and self._mean is not None
        C, D = self._mean.shape
        std = np.zeros((C, D), dtype=np.float32)
        for i in range(C):
            ni = int(self._n[i])
            if ni <= 1:
                std[i, :] = 1e-6
            else:
                var = self._M2[i] / float(max(1, ni - 1))
                std[i, :] = np.sqrt(np.maximum(var, 1e-12))
        scale = np.sqrt(np.maximum((std * std).mean(axis=1), 1e-12)).astype(np.float32)
        self._std, self._scale = std, scale

    # --------------------------------- training ---------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the classifier from scratch.

        Parameters
        ----------
        X : ndarray of shape (N, D)
            Training features.

        y : ndarray of shape (N,)
            Integer labels aligned with X.

        Returns
        -------
        self : PyEGM
            Fitted instance.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("X must be (N, D) and y must be (N,) with matching N.")

        Xp = self._prepare_space(X)
        classes = np.unique(y)
        C, D = len(classes), Xp.shape[1]

        n = np.zeros((C,), dtype=np.int64)
        mean = np.zeros((C, D), dtype=np.float32)
        M2 = np.zeros((C, D), dtype=np.float32)

        for i, c in enumerate(classes):
            Xi = Xp[y == c]
            ni, mui, M2i = self._batch_stats(Xi)
            n[i] = ni
            mean[i] = mui
            M2[i] = M2i

        if self.cfg.metric == "cos" and self.cfg.normalize:
            mean = _normalize_rows(mean)

        self._classes = classes.astype(np.int64, copy=False)
        self._n, self._mean, self._M2 = n, mean, M2
        self._dim = int(D)
        self._derive_std_scale()

        # Reset arrival centers; rebuilt if requested
        self._centers_flat = None
        if self.cfg.cache_centers and self.cfg.num_shells > 0 and self.cfg.num_rays > 0:
            self._build_centers_flat()

        # Rebuild device caches based on availability
        self._device = self._pick_device(self.cfg.platform)
        self._maybe_build_torch_cache()
        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: Optional[np.ndarray] = None):
        """
        Update the classifier with additional data.

        Parameters
        ----------
        X : ndarray of shape (N, D)
            New batch of features.

        y : ndarray of shape (N,)
            Integer labels for the batch.

        classes : ndarray of shape (C_total,), optional
            Ignored for this implementation; new labels are automatically
            added when observed.

        Returns
        -------
        self : PyEGM
            Updated instance.
        """
        if self._classes is None:
            return self.fit(X, y)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        Xp = self._prepare_space(X)

        # Expand label set if new classes appear
        new_labels = np.setdiff1d(np.unique(y), self._classes)
        if new_labels.size > 0:
            C_old, D = self._mean.shape
            C_new = C_old + new_labels.size
            order = np.concatenate([self._classes, new_labels]).astype(np.int64)
            n = np.zeros((C_new,), dtype=np.int64); n[:C_old] = self._n
            mean = np.zeros((C_new, D), dtype=np.float32); mean[:C_old] = self._mean
            M2 = np.zeros((C_new, D), dtype=np.float32); M2[:C_old] = self._M2
            self._classes, self._n, self._mean, self._M2 = order, n, mean, M2

        # Aggregate Welford statistics
        idx_map = {int(c): i for i, c in enumerate(self._classes.tolist())}
        for c in np.unique(y):
            Xi = Xp[y == c]
            ni, mui, M2i = self._batch_stats(Xi)
            i = idx_map[int(c)]
            n0, mu0, M20 = int(self._n[i]), self._mean[i], self._M2[i]
            n, mu, M2 = self._combine_stats(n0, mu0, M20, ni, mui, M2i)
            self._n[i], self._mean[i], self._M2[i] = n, mu, M2

        if self.cfg.metric == "cos" and self.cfg.normalize:
            self._mean = _normalize_rows(self._mean)

        self._derive_std_scale()

        # Rebuild centers if needed
        self._centers_flat = None
        if self.cfg.cache_centers and self.cfg.num_shells > 0 and self.cfg.num_rays > 0:
            self._build_centers_flat()

        # Refresh device caches
        self._device = self._pick_device(self.cfg.platform)
        self._maybe_build_torch_cache()
        return self

    # --------------------------- arrival center generation ---------------------------

    def _ray_directions(self, D: int, M: int, seed: int) -> np.ndarray:
        """
        Generate M deterministic unit vectors in R^D.

        Parameters
        ----------
        D : int
            Dimensionality of the feature space.

        M : int
            Number of rays to generate.

        seed : int
            Random seed for reproducibility.

        Returns
        -------
        ndarray of shape (M, D)
            Row-wise unit vectors.
        """
        rng = np.random.default_rng(seed)
        U = rng.normal(size=(M, D)).astype(np.float32)
        U /= (np.linalg.norm(U, axis=1, keepdims=True) + 1e-9)
        return U

    def _centers_for_class(self, c_idx: int) -> List[np.ndarray]:
        """
        Create arrival centers for one class across shells.

        Parameters
        ----------
        c_idx : int
            Index into the internal class ordering.

        Returns
        -------
        list of ndarray
            For each shell s ∈ {0..S−1}, an array of shape (M, D).
        """
        assert self._mean is not None and self._std is not None and self._scale is not None
        S, M = int(self.cfg.num_shells), int(self.cfg.num_rays)
        mu = self._mean[c_idx]           # (D,)
        std = self._std[c_idx]           # (D,)
        scl = float(self._scale[c_idx])  # scalar

        shells: List[np.ndarray] = []
        D = mu.shape[0]
        for s in range(S):
            r_s = float(self.cfg.alpha) * scl * (self.cfg.gamma ** s)
            seed = (self.cfg.random_state * 73856093 + int(self._classes[c_idx]) * 19349663 + (s + 1) * 83492791) & 0xFFFFFFFF  # type: ignore
            U = self._ray_directions(D, M, seed=seed)     # (M,D)
            V = std[None, :] * U                          # anisotropic stretch
            V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
            Z = mu[None, :] + (r_s * float(self.cfg.eta)) * V
            shells.append(Z.astype(np.float32, copy=False))
        return shells

    def _build_centers_flat(self) -> None:
        """
        Precompute flattened per-shell matrices for fast matmul.

        For each shell s:
          - Z          : (C*M, D) stacked per class
          - Z_norm     : normalized rows for cosine path (or Z if metric!="cos")
          - Z_sq       : row-wise squared norms for L2 path
        """
        assert self._classes is not None and self._mean is not None and self._std is not None
        C, D = self._mean.shape
        S, M = int(self.cfg.num_shells), int(self.cfg.num_rays)
        out: List[Dict[str, np.ndarray]] = []

        per_class_shells: List[List[np.ndarray]] = [self._centers_for_class(i) for i in range(C)]

        for s in range(S):
            blocks = [per_class_shells[i][s] for i in range(C)]
            Z_flat = np.vstack(blocks).astype(np.float32, copy=False)  # (C*M, D)

            rec: Dict[str, np.ndarray] = {"Z": Z_flat}
            if self.cfg.metric == "cos" and self.cfg.normalize:
                rec["Z_norm"] = _normalize_rows(Z_flat)
            else:
                rec["Z_norm"] = Z_flat

            if self.cfg.metric == "l2":
                rec["Z_sq"] = (Z_flat * Z_flat).sum(axis=1).astype(np.float32)
            out.append(rec)

        self._centers_flat = out

    # ------------------------------- scoring kernels -------------------------------

    def _proto_scores(self, Xp: np.ndarray) -> np.ndarray:
        """
        Prototype-channel scores.

        Parameters
        ----------
        Xp : ndarray of shape (N, D)
            Preprocessed inputs (normalized if cosine metric).

        Returns
        -------
        ndarray of shape (N, C)
            Prototype kernel scores.
        """
        assert self._classes is not None and self._mean is not None and self._scale is not None

        # GPU path
        if self._device == "cuda" and torch is not None and self._mean_t is not None:
            X_t = torch.tensor(Xp, device="cuda", dtype=torch.float32)
            if self.cfg.metric == "cos":
                S = X_t @ self._mean_t.T
                out = torch.exp(S / float(self.cfg.tau0))
                return out.detach().cpu().numpy().astype(np.float32)
            else:
                C = self._mean_t.shape[0]
                X_sq = torch.sum(X_t * X_t, dim=1, keepdim=True)               # (N,1)
                Mu_sq = torch.sum(self._mean_t * self._mean_t, dim=1)[None, :] # (1,C)
                D2 = X_sq + Mu_sq - 2.0 * (X_t @ self._mean_t.T)               # (N,C)
                D2 = torch.clamp(D2, min=0.0)
                sigma = self._sigma_t if self._sigma_t is not None else torch.ones(C, device="cuda")
                denom = 2.0 * (sigma[None, :] ** 2) + 1e-12
                out = torch.exp(-D2 / denom)
                return out.detach().cpu().numpy().astype(np.float32)

        # CPU path
        if self.cfg.metric == "cos":
            S = Xp @ self._mean.T
            return np.exp(S / float(self.cfg.tau0)).astype(np.float32)

        sigma = (self._scale * float(self.cfg.l2_sigma_scale)).astype(np.float32)  # (C,)
        X_sq = (Xp * Xp).sum(axis=1, keepdims=True)             # (N,1)
        Mu_sq = (self._mean * self._mean).sum(axis=1)[None, :]  # (1,C)
        D2 = X_sq + Mu_sq - 2.0 * (Xp @ self._mean.T)           # (N,C)
        np.maximum(D2, 0.0, out=D2)
        denom = 2.0 * (sigma[None, :] ** 2) + 1e-12
        return np.exp(-D2 / denom).astype(np.float32)

    def _arrival_scores(self, Xp: np.ndarray) -> np.ndarray:
        """
        Arrival-center-channel scores.

        Parameters
        ----------
        Xp : ndarray of shape (N, D)
            Preprocessed inputs (normalized if cosine metric).

        Returns
        -------
        ndarray of shape (N, C)
            Arrival-center kernel scores aggregated over shells and rays.
        """
        assert self._classes is not None and self._mean is not None and self._scale is not None

        S, M = int(self.cfg.num_shells), int(self.cfg.num_rays)
        if S == 0 or M == 0:
            return np.zeros((Xp.shape[0], self._classes.shape[0]), dtype=np.float32)

        if self._centers_flat is None:
            self._build_centers_flat()
        # Ensure torch cache exists if GPU is in use and centers were just built
        if self._device == "cuda" and (self._centers_flat_t is None):
            self._maybe_build_torch_cache()

        # GPU path
        if self._device == "cuda" and torch is not None and self._centers_flat_t is not None:
            C = self._classes.shape[0]
            X_t = torch.tensor(Xp, device="cuda", dtype=torch.float32)
            out = torch.zeros((X_t.shape[0], C), device="cuda", dtype=torch.float32)
            raw = torch.tensor([1.0 / (self.cfg.gamma ** s) for s in range(S)],
                               device="cuda", dtype=torch.float32)
            w = raw / (torch.sum(raw) + 1e-12)

            if self.cfg.metric == "cos":
                for s in range(S):
                    Z = self._centers_flat_t[s]["Z_norm"]      # (C*M, D)
                    K = torch.exp((X_t @ Z.T) / float(self.cfg.tau0))  # (N, C*M)
                    K = K.reshape(X_t.shape[0], C, M).mean(dim=2)      # (N, C)
                    out = out + w[s] * K
                return out.detach().cpu().numpy().astype(np.float32)

            # L2 path
            sigma = self._sigma_t if self._sigma_t is not None else torch.ones(C, device="cuda")
            for s in range(S):
                rec = self._centers_flat_t[s]
                Z = rec["Z"]                                   # (C*M, D)
                Z_sq = rec["Z_sq"]                             # (C*M,)
                X_sq = torch.sum(X_t * X_t, dim=1, keepdim=True)    # (N,1)
                D2 = X_sq + Z_sq[None, :] - 2.0 * (X_t @ Z.T)        # (N, C*M)
                D2 = torch.clamp(D2, min=0.0)
                denom = 2.0 * (torch.repeat_interleave(sigma, repeats=M)[None, :] ** 2) + 1e-12
                K = torch.exp(-D2 / denom)                           # (N, C*M)
                K = K.reshape(X_t.shape[0], C, M).mean(dim=2)        # (N, C)
                out = out + w[s] * K
            return out.detach().cpu().numpy().astype(np.float32)

        # CPU path
        C = self._classes.shape[0]
        out = np.zeros((Xp.shape[0], C), dtype=np.float32)

        # Shell weights: heavier near the center when gamma > 1
        raw = np.array([1.0 / (self.cfg.gamma ** s) for s in range(S)], dtype=np.float32)
        w = raw / (raw.sum() + 1e-9)

        if self.cfg.metric == "cos":
            for s in range(S):
                Z = self._centers_flat[s]["Z_norm"]  # (C*M, D)
                K = np.exp((Xp @ Z.T) / float(self.cfg.tau0)).astype(np.float32)  # (N, C*M)
                K = K.reshape(Xp.shape[0], C, M).mean(axis=2)  # (N, C)
                out += w[s] * K
            return out

        sigma = (self._scale * float(self.cfg.l2_sigma_scale)).astype(np.float32)  # (C,)
        for s in range(S):
            Z = self._centers_flat[s]["Z"]                 # (C*M, D)
            Z_sq = self._centers_flat[s]["Z_sq"]           # (C*M,)
            X_sq = (Xp * Xp).sum(axis=1, keepdims=True)    # (N,1)
            D2 = X_sq + Z_sq[None, :] - 2.0 * (Xp @ Z.T)   # (N, C*M)
            np.maximum(D2, 0.0, out=D2)
            denom = 2.0 * (np.repeat(sigma, M)[None, :] ** 2) + 1e-12
            K = np.exp(-D2 / denom).astype(np.float32)     # (N, C*M)
            K = K.reshape(Xp.shape[0], C, M).mean(axis=2)  # (N, C)
            out += w[s] * K
        return out

    def _scores(self, X: np.ndarray) -> np.ndarray:
        """
        Total blended scores.

        Parameters
        ----------
        X : ndarray of shape (N, D)
            Raw inputs before preprocessing.

        Returns
        -------
        ndarray of shape (N, C)
            Combined prototype and arrival-center scores.
        """
        assert self._classes is not None
        Xp = self._prepare_space(np.asarray(X, dtype=np.float32))
        proto = self._proto_scores(Xp)
        if self.cfg.beta0 >= 1.0 or self.cfg.num_shells == 0 or self.cfg.num_rays == 0:
            return proto
        arr = self._arrival_scores(Xp)
        return float(self.cfg.beta0) * proto + float(1.0 - self.cfg.beta0) * arr

    # --------------------------------- inference ---------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict integer labels.

        Parameters
        ----------
        X : ndarray of shape (N, D)
            Input features.

        Returns
        -------
        ndarray of shape (N,)
            Predicted labels from argmax over class scores.
        """
        if self._classes is None:
            raise RuntimeError("Model is not fitted.")
        S = self._scores(X)
        return self._classes[np.argmax(S, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Mean accuracy.

        Parameters
        ----------
        X : ndarray of shape (N, D)
            Input features.

        y : ndarray of shape (N,)
            Ground-truth integer labels.

        sample_weight : ndarray of shape (N,), optional
            Optional sample weights.

        Returns
        -------
        float
            Accuracy computed with scikit-learn's signature.
        """
        y = np.asarray(y, dtype=np.int64)
        yp = self.predict(X)
        return float(accuracy_score(y, yp, sample_weight=sample_weight))

    # ============================== Persistence ============================

    def save(self, dir_path: str) -> None:
        """
        Save configuration and running statistics.

        Parameters
        ----------
        dir_path : str
            Directory path to create or reuse. Files written:
            - config.json
            - classes.npy
            - n.npy
            - mean.npy
            - M2.npy

        Notes
        -----
        Arrival centers are recomputed on demand after loading.
        """
        if self._classes is None or self._mean is None or self._M2 is None or self._n is None:
            raise RuntimeError("Model must be fitted before save().")

        p = Path(dir_path)
        _ensure_dir(p)

        meta = {"version": "pce-0.1", "dim": int(self._dim) if self._dim is not None else None}
        with open(p / "config.json", "w", encoding="utf-8") as f:
            json.dump({"cfg": self.cfg.__dict__, "meta": meta}, f, indent=2, ensure_ascii=False)

        np.save(p / "classes.npy", self._classes)
        np.save(p / "n.npy", self._n)
        np.save(p / "mean.npy", self._mean)
        np.save(p / "M2.npy", self._M2)

    @classmethod
    def load(cls, dir_path: str) -> "PyEGM":
        """
        Load a model directory produced by `save(...)`.

        Parameters
        ----------
        dir_path : str
            Directory containing saved arrays and config.

        Returns
        -------
        PyEGM
            Restored instance ready for scoring after centers are regenerated.
        """
        p = Path(dir_path)
        with open(p / "config.json", "r", encoding="utf-8") as f:
            obj = json.load(f)
        cfg = ExplosionConfig(**obj["cfg"])
        inst = cls(cfg)

        inst._classes = np.load(p / "classes.npy")
        inst._n = np.load(p / "n.npy")
        inst._mean = np.load(p / "mean.npy")
        inst._M2 = np.load(p / "M2.npy")
        inst._dim = int(inst._mean.shape[1])
        inst._derive_std_scale()

        inst._centers_flat = None
        # Prepare device caches according to current environment
        inst._device = inst._pick_device(inst.cfg.platform)
        inst._maybe_build_torch_cache()
        return inst

    # ============================== Introspection ==========================

    def get_fitted_params(self) -> Dict[str, Any]:
        """
        Return configuration and runtime metadata.

        Returns
        -------
        dict
            Dictionary containing the configuration and a 'runtime' section
            with num_classes, feature_dim, counts, and device.
        """
        if self._classes is None or self._mean is None:
            raise RuntimeError("Model must be fitted first.")
        return {
            "config": dict(self.cfg.__dict__),
            "runtime": {
                "num_classes": int(self._classes.shape[0]),
                "feature_dim": int(self._mean.shape[1]),
                "counts": self._n.astype(int).tolist() if self._n is not None else None,
                "device": self._device,
            },
        }

