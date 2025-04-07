# PyEGM: Explosive Generative Model for Classification

PyEGM is an explosive generative model (EGM) classifier designed for incremental learning. It generates new training points during each iteration and adapts its geometry based on the input data. This allows it to improve over time and make accurate predictions even as the data evolves.

## 🌟 Key Features

- **Dual Generation Modes**
  - 🎯 **Hypersphere Generation**: Creates samples on dimensional spheres
  - 🌌 **Gaussian Generation**: Produces samples through probabilistic dispersion

- **Dynamic Adaptation**
  - 🔍 Local/Global radius adjustment strategies
  - ⚖️ Automatic sample pruning with decay mechanism
  - 📈 Adaptive density-aware scaling

- **Practical Advantages**
  - 🚀 Built-in incremental learning capability
  - 🛡️ Robust to class imbalance through proportional generation
  - ⚡ Efficient neighbor search with NN indexing

## Installation
To install PyEGM, you can simply use pip:
```python
pip install pyegm
```
