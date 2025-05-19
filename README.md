# PyEGM: Physically-Inspired Explosive Generative Model for Classification

PyEGM is a physically-inspired explosive generative model (EGM) classifier designed for incremental learning. It generates new training points by simulating a physical explosion process and dynamically adjusts its generation strategy based on the input data distribution. This allows it to improve over time and make accurate predictions even as the data evolves.

## 🌟 Key Features
- **Physics-Inspired Explosive Generation**
  - 💥 **Energy Allocation**: Dirichlet-distributed energy partitioning
  - 🚀 **Kinematic Simulation**: Velocity derived from kinetic energy formula (v = √(2E/m))
  - 🌪️ **Directional Dispersion**: Random unit vectors with controlled noise

- **Dynamic Generation**
  - 🔄 **Incremental Learning**: Seamless updates via `partial_fit`
  - 🧩 **Class-Conditional Generation**: Per-class energy allocation
  - 🎛️ **Noise Control**: Local or global radius adjustment strategies

- **Practical Advantages**
  - 🚀 High-performance neighbor search with HNSW indexing
  - 🛡️ Robust to class imbalance through proportional generation
  - ⚡ Efficient memory management with sample limits

## Installation
To install PyEGM, you can simply use pip:
```python
pip install pyegm
```
# Usage
If you have PyEGM in your project, you can import it directly:
```python
from pyegm import PyEGM
```
## Model Parameters:
- num_points: Number of new points generated per class.
- total_energy: Total energy allocated to each class for the explosion process.
- mass: Mass of each generated "particle" in the explosion.
- explosion_time: Duration of the explosion influencing the displacement.
- noise_scale: Scale of the noise added to each new point.
- dirichlet_alpha: Concentration parameter for Dirichlet distribution splitting total_energy.
- max_samples: Maximum number of samples to retain.
- radius_adjustment: Strategy for adjusting radius ('local' or 'global').
- decay_factor: Coefficient for sample weight decay in incremental learning.
- new_data_weight: Initial weight assigned to new data in partial_fit.
- generate_in_partial_fit: Whether to generate new points in partial_fit.

## Model Methods:
- fit(X, y): Trains the model and generates new points based on the explosion process.
- partial_fit(X, y): Allows for incremental training, adding new data while generating additional samples.
- predict(X): Predicts class labels for the provided input data.
- score(X, y, sample_weight=None): Evaluates the model's accuracy on test data.
                                                                                 
# Example: 
```python
import numpy as np
from pyegm import PyEGM
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, 
                          n_classes=3, n_informative=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Initialize and train the PyEGM model
model = PyEGM(num_points=50, total_energy=1.2, noise_scale=0.05)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")

# Incremental learning with new data
X_new, y_new = make_classification(n_samples=200, n_features=10,
                                  n_classes=3, n_informative=5)
model.partial_fit(X_new, y_new)

# Re-evaluate after incremental learning
new_accuracy = model.score(X_test, y_test)
print(f"Accuracy after partial_fit: {new_accuracy:.4f}")
```
# Advanced Usage:
For imbalanced datasets, you can adjust the energy allocation to generate more samples for minority classes:
```python
# For highly imbalanced data
imbalanced_model = PyEGM(
    num_points=100,         # Generate more points
    total_energy=2.0,       # Higher energy for more dispersion
    dirichlet_alpha=0.5,    # More variance in energy allocation
    noise_scale=0.2         # Add more noise for diversity
)
```

