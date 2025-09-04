# PyEGM：以物理爆炸现象为灵感的样本生长模型

> 通过“能量—方向”的类比在类内进行生长式生成，并以趋势路径投票完成判别；适用于小样本增量与不平衡场景。  
> ⚙️ 后端可切换：`platform="auto"|"cpu"|"cuda"`（检测到 CUDA 时启用 GPU，否则回退 CPU）。

---

## 🧠 模型灵感

### 为什么选择“物理爆炸现象”作为灵感
在小样本与长尾条件下，真实数据难以覆盖决策边界附近的关键区域，单纯的插值或随机数据增强容易导致边界外推失真。我们期望一种**面向边界方向、幅度可控、阶段递进**的类内扩张机制：既能将有限样本向潜在边界推进，又避免越界到对手类别的势域。物理爆炸现象呈现了**能量瞬时释放 → 物质沿特定方向外扩 → 受环境约束而产生各向异性与壳层结构 → 前沿分阶段推进**的过程，这与我们的方法论目标高度契合，因此选择其作为模型灵感来源（而非求解真实动力学方程）。

### 我们的做法
- **生长式生成**：为每个类别分配“能量预算”，驱动样本沿局部几何主轴向外扩张；`total_energy、mass、explosion_time、noise_*` 等参数控制生成强度与节奏。
- **各向异性与对手抑制**：基于局部协方差塑形（`anisotropy`），并对朝向对手类中心的分量进行偏折/抑制（`deflect_rival, deflect_strength`）。
- **壳层与半径控制**：通过 `shell_ratio, shell_jitter, adaptive_radius_mode` 等限制外推空间，使生成壳层贴近潜在决策边界。
- **趋势路径投票**：推理时沿候选类别方向多步推进（`path_steps, path_step_size`），每步聚合近邻并加权（`path_k, path_gamma, step_weight_mode`）；路径不可靠时回退多数近邻表决。

---

## ⚙️ API 说明

### 构造函数关键参数
- **预设与配置**：`preset`（`auto|fscil|balanced_kshot|imbalanced_kshot|extreme_lowshot` 或 `*.yaml`）  
- **生成（物理类比）**：`num_points, total_energy, mass, explosion_time, noise_scale, noise_decay, dirichlet_alpha, dynamic_energy`  
- **半径/壳层**：`adaptive_radius_mode, fixed_radius, use_radius_clip, shell_ratio, shell_jitter`  
- **几何/对抗**：`local_k, anisotropy, deflect_rival, deflect_strength, momentum_conserve`  
- **趋势路径**：`path_steps, path_step_size, path_k, path_gamma, step_weight_mode, step_gamma`  
- **调度/稳定**：`points_schedule, energy_schedule, num_iters, energy_decay, radius_growth, governor`  
- **后端与环境**：`platform`（`auto/cpu/cuda`），`torch_q_block`, `torch_db_block`, `random_state`

### 方法一览
- `fit(X, y)`：训练基座；解析预设并生成合成样本，建立检索后端。  
- `partial_fit(X, y, classes=None)`：增量训练（few-shot / 新类并入）。  
- `continue_fit(extra_iters=..., reseed=None)`：在当前状态上追加少量迭代（短点续训）。  
- `predict(X)`：趋势路径投票预测。  
- `score(X, y)`：便捷评估。  
- `export_fixed_yaml(path, include_meta=True)`：导出最终生效超参快照。  
- `get_fitted_params()`：返回最终生效超参字典。  
- `save(dir_path)` / `load(dir_path)`：保存与恢复模型（包含配置、内部数组与后端索引）。  
- `visualize_explosion(...)`：二维投影下的生长过程与趋势路径可视化，支持导出静图/动画。

---

## 🧩 功能介绍

### 🎬 可视化
- **展示内容**：真实样本、合成样本、能量向量、趋势路径；支持 PCA / t-SNE / UMAP 投影。  
- **常用参数**：`projection`、`n_samples`、`path_steps`、`show_synthetic`、`show_energy_vectors`、`save_path`（按后缀导出 `png/gif/mp4`）。  
- **依赖**：`matplotlib`；动画导出可选 `imageio`/`imageio-ffmpeg` 或 `moviepy`；UMAP 需 `umap-learn`。  
- **示例**：
  ```python
  clf.visualize_explosion(
      projection="pca", n_samples=800, path_steps=6,
      show_synthetic=True, show_energy_vectors=True,
      save_path="explosion.png", show=False
  )
  ```

### 💾 保存与 🔁 续训
- **保存与恢复**：`save(dir)` 写出配置与内部数组，并保存所需索引/缓存；`PyEGM.load(dir)` 可在目标环境中恢复并直接推理或继续训练。  
- **短点续训**：`continue_fit(extra_iters=..., reseed=...)` 在既有配置与数据上追加少量迭代，以降低再次训练开销。  
- **增量会话**：`partial_fit(X, y, classes=None)` 并入新增样本并更新索引；可与保存/加载结合，用于分阶段训练与评估。  
- **后端选择**：`platform="auto"` 在检测到 CUDA 时使用 GPU，否则回退 CPU；跨环境恢复时按可用性自动选择。

---

## 🚀 最小使用示例
```python
import numpy as np
from pyegm import PyEGM
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 数据
X, y = make_classification(n_samples=2000, n_features=64, n_classes=5, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

# 初始化与训练
clf = PyEGM(preset="auto", platform="auto", random_state=42)
clf.fit(Xtr.astype(np.float32), ytr)
print("Accuracy:", clf.score(Xte.astype(np.float32), yte))

# 保存与续训
clf.export_fixed_yaml("pyegm_fixed.yaml")
clf.save("checkpoints/egm_run")

clf2 = PyEGM.load("checkpoints/egm_run")
clf2.continue_fit(extra_iters=3, reseed=123)
clf2.partial_fit(Xtr[:40].astype(np.float32), ytr[:40])
clf2.save("checkpoints/egm_run_after")

# 可视化（导出静图）
clf2.visualize_explosion(
    projection="pca", n_samples=800, path_steps=6,
    show_synthetic=True, show_energy_vectors=True,
    save_path="explosion.png", show=False
)
```

---

### ✅ 特性速览
- 🧩 面向少样本与长尾的**生长式生成**与**趋势路径投票**  
- ⚙️ **CPU/GPU** 后端可切换（自动检测）  
- 💾 支持**保存/恢复/续训**与**超参快照导出**  
- 🎬 提供**二维可视化**与**动画导出**能力
