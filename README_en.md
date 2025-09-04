# PyEGM: A Sample Growth Model Inspired by Physical Explosion Phenomena

> Grow class-wise samples via an energy–direction analogy and decide with trend‑path voting. Designed for few‑shot class‑incremental and long‑tailed regimes.  
> ⚙️ Backend switchable: `platform="auto"|"cpu"|"cuda"` (uses CUDA when available, otherwise falls back to CPU).

---

## 🧠 Model Inspiration

### Why “physical explosion phenomena”?
In few‑shot and long‑tailed settings, training data rarely covers the critical regions near decision boundaries. Pure interpolation or random augmentations often extrapolate poorly. We need a *direction‑aware, magnitude‑controlled, stage‑wise* mechanism for **within‑class growth**: push limited samples toward plausible boundaries while avoiding intrusions into rival classes. Explosion phenomena exhibit **instantaneous energy release → outward expansion along directions → anisotropy and shell‑like structures under constraints → front propagation in stages**. These traits align with our methodological goals, so we use them as a model inspiration rather than solving real dynamics equations.

### Our approach
- **Growth‑style generation**: Allocate an “energy budget” per class and expand samples along local principal directions. Parameters such as `total_energy`, `mass`, `explosion_time`, and `noise_*` control intensity and pacing.
- **Anisotropy and rival suppression**: Shape with local covariance (`anisotropy`) and deflect components toward rival class centers (`deflect_rival`, `deflect_strength`), echoing directional preference and shielding effects.
- **Shells and radius control**: Limit extrapolation using `shell_ratio`, `shell_jitter`, and `adaptive_radius_mode`, keeping synthetic “shells” close to plausible decision fronts.
- **Trend‑path voting**: At inference, move step‑by‑step along each candidate class direction (`path_steps`, `path_step_size`), aggregate neighbors with weights (`path_k`, `path_gamma`, `step_weight_mode`), and fall back to majority neighbors when a path is unreliable.

---

## ⚙️ API Reference

### Constructor — key arguments
- **Presets & config**: `preset` (`auto|fscil|balanced_kshot|imbalanced_kshot|extreme_lowshot` or a `*.yaml` path)  
- **Generation (analogy)**: `num_points`, `total_energy`, `mass`, `explosion_time`, `noise_scale`, `noise_decay`, `dirichlet_alpha`, `dynamic_energy`  
- **Radius / shells**: `adaptive_radius_mode`, `fixed_radius`, `use_radius_clip`, `shell_ratio`, `shell_jitter`  
- **Geometry / rivalry**: `local_k`, `anisotropy`, `deflect_rival`, `deflect_strength`, `momentum_conserve`  
- **Trend‑path**: `path_steps`, `path_step_size`, `path_k`, `path_gamma`, `step_weight_mode`, `step_gamma`  
- **Scheduling / stability**: `points_schedule`, `energy_schedule`, `num_iters`, `energy_decay`, `radius_growth`, `governor`  
- **Backend & env**: `platform` (`auto/cpu/cuda`), `torch_q_block`, `torch_db_block`, `random_state`

### Methods — quick list
- `fit(X, y)`: Train the base session, parse preset/config, synthesize growth samples, and build the retrieval backend.  
- `partial_fit(X, y, classes=None)`: Incremental update for few‑shot or newly introduced classes.  
- `continue_fit(extra_iters=..., reseed=None)`: Add a small number of extra iterations on the current state (short resuming).  
- `predict(X)`: Trend‑path‑based prediction.  
- `score(X, y)`: Convenience evaluation.  
- `export_fixed_yaml(path, include_meta=True)`: Export the final effective hyperparameter snapshot.  
- `get_fitted_params()`: Return the final effective hyperparameter dictionary.  
- `save(dir_path)` / `load(dir_path)`: Checkpoint and restore the model (config, tensors/arrays, and backend index).  
- `visualize_explosion(...)`: 2‑D projection visualization of growth and paths; supports static images and animations.

---

## 🧩 Features

### 🎬 Visualization
- **What it shows**: real samples, synthetic growth samples, energy vectors, and trend paths; supports PCA / t‑SNE / UMAP projections.  
- **Common arguments**: `projection`, `n_samples`, `path_steps`, `show_synthetic`, `show_energy_vectors`, `save_path` (file suffix decides `png/gif/mp4`).  
- **Dependencies**: `matplotlib`; for animations use `imageio`/`imageio-ffmpeg` or `moviepy`; UMAP requires `umap-learn`.  
- **Example**:
  ```python
  clf.visualize_explosion(
      projection="pca", n_samples=800, path_steps=6,
      show_synthetic=True, show_energy_vectors=True,
      save_path="explosion.png", show=False
  )
  ```

### 💾 Checkpointing & 🔁 Resuming
- **Save & restore**: `save(dir)` writes config and internal arrays, plus the needed index/cache; `PyEGM.load(dir)` restores a usable model for inference or further training.  
- **Short resuming**: `continue_fit(extra_iters=..., reseed=...)` adds a small number of iterations on the existing configuration and data to reduce retraining overhead.  
- **Incremental sessions**: `partial_fit(X, y, classes=None)` merges new samples and updates indexing; combine with save/load for staged training and evaluation.  
- **Backend choice**: `platform="auto"` uses GPU when CUDA is available, otherwise CPU. When restoring across environments, the backend is chosen by availability.

---

## 🚀 Minimal Example
```python
import numpy as np
from pyegm import PyEGM
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Data
X, y = make_classification(n_samples=2000, n_features=64, n_classes=5, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize & train
clf = PyEGM(preset="auto", platform="auto", random_state=42)
clf.fit(Xtr.astype(np.float32), ytr)
print("Accuracy:", clf.score(Xte.astype(np.float32), yte))

# Save & resume
clf.export_fixed_yaml("pyegm_fixed.yaml")
clf.save("checkpoints/egm_run")

clf2 = PyEGM.load("checkpoints/egm_run")
clf2.continue_fit(extra_iters=3, reseed=123)
clf2.partial_fit(Xtr[:40].astype(np.float32), ytr[:40])
clf2.save("checkpoints/egm_run_after")

# Visualization (export a static image)
clf2.visualize_explosion(
    projection="pca", n_samples=800, path_steps=6,
    show_synthetic=True, show_energy_vectors=True,
    save_path="explosion.png", show=False
)
```

---

### ✅ At‑a‑Glance
- 🧩 Growth‑style generation + trend‑path voting for few‑shot and long‑tailed scenarios  
- ⚙️ Switchable CPU/GPU backends with automatic detection  
- 💾 Checkpoint/restore/resume and hyperparameter snapshot export  
- 🎬 2‑D visualization and animation export
