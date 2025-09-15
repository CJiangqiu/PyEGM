[English](#english) | [中文说明](#chinese)

<a id="english"></a>

# PyEGM — Physics-Inspired Exemplar Growth Model (GPU-Optional)

PyEGM is a lightweight classifier for **fixed-feature** protocols. Each class is represented by a **prototype** (class mean) and a small set of **arrival centers** placed on concentric shells along deterministic rays. Prediction blends a prototype kernel and an arrival-center kernel with a single mixing weight.

- **Core**: NumPy + scikit-learn  
- **Optional (acceleration)**: **User-installed** PyTorch + CUDA for GPU scoring (falls back to CPU)

**API**: `fit`, `partial_fit`, `predict`, `score`, `save`, `load`, 

---

## Installation

```bash
pip install pyegm
```
Core requirements: `numpy>=1.21`, `scikit-learn>=1.0`  

### Enabling GPU Acceleration (Optional)
PyEGM does **not** depend on PyTorch by default. If you want GPU scoring:
1. Install a **compatible CUDA toolkit / driver** on your machine.
2. Install **PyTorch** from the official website and select a wheel that matches your OS and CUDA version: https://pytorch.org/get-started/locally/  
3. Keep `ExplosionConfig(platform="auto")` (default) or set `"cuda"`. When PyTorch+CUDA is detected, scoring runs on GPU; otherwise it falls back to CPU with identical results.

> Scope: GPU is used for matrix computations in scoring (prototype/arrival kernels). Training remains CPU-only.

---

## Quick Start

```python
import numpy as np
from pyegm import PyEGM, ExplosionConfig

# toy data: 6 classes, 128-dim embeddings
X = np.random.randn(600, 128).astype("float32")
y = np.repeat(np.arange(6), repeats=100).astype("int64")

cfg = ExplosionConfig(
    metric="cos",        # "cos" (temperature-scaled cosine) or "l2" (RBF)
    normalize=True,      # row-wise L2 normalization when metric="cos"
    num_shells=2,        # shells (S)
    num_rays=8,          # rays per shell (M)
    alpha=1.0,           # base radius scale
    gamma=1.6,           # shell growth factor
    eta=1.4,             # radial gain after whitening
    tau0=0.12,           # temperature for cosine kernel
    beta0=0.60,          # mix between prototype and arrival channels
    l2_sigma_scale=1.0,  # width scale for RBF when metric="l2"
    cache_centers=True,  # precompute arrival centers
    random_state=0,
    platform="auto",     # "auto" | "cpu" | "cuda"
)

model = PyEGM(config=cfg).fit(X, y)
print("device:", model.get_fitted_params()["runtime"]["device"])  # "cuda" or "cpu"
print("pred:", model.predict(X[:8]))
print("acc :", model.score(X, y))
```

> PyEGM operates on **fixed features** (e.g., precomputed embeddings). No backbone is trained here.

---

## Incremental Updates

```python
# simulate a new batch with new and/or old classes
X_new = np.random.randn(100, 128).astype("float32")
y_new = np.random.choice(np.arange(8), size=100).astype("int64")

# update running statistics; unseen labels are added automatically
model.partial_fit(X_new, y_new)

# evaluate again
print("acc after update:", model.score(X, y))
```

---

## Optional Acceleration (Details)

- **Activation**: Install PyTorch from the official site (see link above) and ensure your local CUDA toolkit/driver is compatible. No extra in `pip install pyegm[...]` is required.  
- **Detection**: With `platform="auto"` (default), the model uses GPU when `torch.cuda.is_available()` is true; otherwise it uses CPU.  
- **Determinism**: GPU and CPU scoring produce **identical** outputs for the same inputs and parameters.

---

## API Reference

### `ExplosionConfig`
Configuration dataclass.

- `metric`: `"cos"` (temperature-scaled cosine) or `"l2"` (RBF over squared L2).
- `normalize`: apply row-wise normalization when `metric="cos"`.
- `num_shells` (`S`): number of shells.
- `num_rays` (`M`): rays per shell.
- `alpha`: base radius scale.
- `gamma`: shell growth factor (`radius_s = alpha * scale * gamma^s`).
- `eta`: radial gain after anisotropic whitening.
- `tau0`: temperature for cosine kernel.
- `beta0 ∈ [0,1]`: blend between prototype and arrival channels (`1.0` reduces to pure NCM).
- `l2_sigma_scale`: width scale for RBF when `metric="l2"`.
- `cache_centers`: precompute and cache arrival centers.
- `random_state`: seed for deterministic ray directions.
- `platform`: `{"auto","cpu","cuda"}` — device preference for scoring.

### `PyEGM`

- `fit(X, y)` → self  
  Compute per-class mean/variance statistics; optionally generate arrival centers.
- `partial_fit(X, y, classes=None)` → self  
  Update running statistics via Welford aggregation; new labels are added as needed.
- `predict(X)` → `ndarray[int]`  
  Class labels from blended kernel scores.
- `score(X, y, sample_weight=None)` → `float`  
  Mean accuracy (scikit-learn signature).
- `save(dir_path)` / `load(dir_path)`  
  Persist/restore configuration and statistics. Arrival centers are regenerated on demand.
- `get_fitted_params()` → `dict`  
  Return configuration and runtime metadata, including current `device`.
- `visualize_explosion(...)`  
  Plot prototypes and arrival centers; optionally overlay a subsample of data points.

---

## Notes
- No ANN/HNSW dependency is required.  

---

[中文说明](#chinese)

<a id="chinese"></a>

# PyEGM — 受物理“爆炸”启发的样本增长模型（可选 GPU 加速）

PyEGM 面向**固定特征**协议。每个类别由**原型**（类均值）与放置在同心外壳、沿确定性射线的**到达中心**共同表征。预测阶段将“原型通道”和“到达通道”的核分数进行加权融合。

- **核心依赖**：NumPy + scikit-learn  
- **可选（加速）**：**用户自行安装** PyTorch + CUDA 后，可在打分阶段使用 GPU（不可用时自动回退 CPU）

**API**：`fit`, `partial_fit`, `predict`, `score`, `save`, `load`, 

---

## 安装

```bash
pip install pyegm                 # 核心
```
核心依赖：`numpy>=1.21`, `scikit-learn>=1.0`  

### 启用 GPU 加速（可选）
PyEGM **不**默认依赖 PyTorch。若需 GPU 打分：  
1. 在本机安装**兼容的 CUDA 工具链/驱动**；  
2. 前往 PyTorch 官网选择与你系统与 CUDA 版本匹配的安装命令：https://pytorch.org/get-started/locally/  
3. 保持 `ExplosionConfig(platform="auto")`（默认）或设为 `"cuda"`。当检测到 PyTorch+CUDA 时，打分在 GPU 上执行；否则自动回退 CPU，结果一致。

> 范围：GPU 仅用于打分阶段的矩阵计算；训练仍在 CPU 上完成。

---

## 快速上手

```python
import numpy as np
from pyegm import PyEGM, ExplosionConfig

X = np.random.randn(600, 128).astype("float32")
y = np.repeat(np.arange(6), repeats=100).astype("int64")

cfg = ExplosionConfig(
    metric="cos",
    normalize=True,
    num_shells=2,
    num_rays=8,
    alpha=1.0,
    gamma=1.6,
    eta=1.4,
    tau0=0.12,
    beta0=0.60,
    l2_sigma_scale=1.0,
    cache_centers=True,
    random_state=0,
    platform="auto",   # "auto" | "cpu" | "cuda"
)

model = PyEGM(cfg).fit(X, y)
print("device:", model.get_fitted_params()["runtime"]["device"])  # "cuda" 或 "cpu"
print("acc:", model.score(X, y))
```

---

## 增量更新

```python
X_new = np.random.randn(100, 128).astype("float32")
y_new = np.random.choice(np.arange(8), size=100).astype("int64")

model.partial_fit(X_new, y_new)
print("acc after update:", model.score(X, y))
```

---

## 可选加速（详细）

- **启用方式**：从 PyTorch 官网安装，并确保本机 CUDA 工具链/驱动与所选 wheel 匹配；无需通过 `pip install pyegm[...]` 安装任何 “gpu” 额外选项。  
- **自动检测**：`platform="auto"`（默认）时，当 `torch.cuda.is_available()` 为真使用 GPU，否则使用 CPU。  
- **一致性**：同一输入与参数下，GPU 与 CPU 的打分结果保持一致。

---

## 说明
- 不依赖 ANN/HNSW。  
