from typing import Literal, Tuple

import hnswlib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y


class PyEGM(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        num_points: int = 100,          # Number of new points to generate per class
        total_energy: float = 1.0,      # Total energy allocated to each class for the explosion process
        mass: float = 1.0,              # Mass of each generated "particle"
        explosion_time: float = 1.0,    # Duration of the explosion influencing the displacement
        noise_scale: float = 0.1,       # Scale of the noise added to each new point
        dirichlet_alpha: float = 1.0,   # Concentration parameter for Dirichlet distribution splitting total_energy
        max_samples: int = 1000,        # Maximum number of samples to retain
        radius_adjustment: Literal['local', 'global'] = 'local',  # Radius adjustment strategy
        decay_factor: float = 0.9,      # Decay factor for sample weight decay in incremental learning
        new_data_weight: float = 0.5,   # Initial weight assigned to new data in partial_fit
        generate_in_partial_fit: bool = True  # Generate new points in partial_fit
    ):
        """
        PyEGM: A Physically-Inspired Explosive Generative Model for Classification

        This model generates new samples by simulating a physical explosion process.
        It also supports incremental learning.
        And dynamically adjusts sample generation based on input data characteristics.

        Parameters:
        - num_points: Number of new points generated per class.
        - total_energy: Total energy allocated to each class for the explosion process.
        - mass: Mass of each generated "particle".
        - explosion_time: Duration of the explosion influencing the displacement.
        - noise_scale: Standard deviation of noise added to generated points.
        - dirichlet_alpha: Concentration parameter for Dirichlet distribution.
        - max_samples: Maximum number of samples to retain.
        - radius_adjustment: Strategy for adjusting radius ('local' or 'global').
        - decay_factor: Coefficient for sample weight decay in incremental learning.
        - new_data_weight: Weight assigned to new data in partial_fit.
        - generate_in_partial_fit: Whether to generate new points in partial_fit.
        """
        self.num_points = num_points
        self.total_energy = total_energy
        self.mass = mass
        self.explosion_time = explosion_time
        self.noise_scale = noise_scale
        self.dirichlet_alpha = dirichlet_alpha
        self.max_samples = max_samples
        self.radius_adjustment = radius_adjustment
        self.decay_factor = decay_factor
        self.new_data_weight = new_data_weight
        self.generate_in_partial_fit = generate_in_partial_fit

        # State variables
        self.trained_points_ = None
        self.trained_labels_ = None
        self.sample_weights_ = None
        self.radius_ = None
        self.dim_ = None
        self.classes_ = None
        self.nn_index_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PyEGM':
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.dim_ = X.shape[1]

        # Initialization
        self.trained_points_ = X
        self.trained_labels_ = y
        self.sample_weights_ = np.ones(len(X))
        self.radius_ = self._adaptive_radius(self.trained_points_)

        # Generate explosive points based on selected method
        new_points, new_labels = self._generate_explosive_points()

        # Merge new points with training data
        self.trained_points_ = np.vstack([self.trained_points_, new_points])
        self.trained_labels_ = np.concatenate([self.trained_labels_, new_labels])
        new_weights = np.ones(len(new_points))
        self.sample_weights_ = np.concatenate([self.sample_weights_, new_weights])

        # Build HNSW-based nearest neighbor index
        self._build_nn_index()

        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> 'PyEGM':
        if self.trained_points_ is None:
            return self.fit(X, y)

        # Append new data
        self.trained_points_ = np.vstack([self.trained_points_, X])
        self.trained_labels_ = np.concatenate([self.trained_labels_, y])
        self.classes_ = np.unique(self.trained_labels_)
        self.dim_ = self.trained_points_.shape[1]

        # Generate explosive points based on selected method
        new_points, new_labels = self._generate_explosive_points()

        # Merge new points with training data
        self.trained_points_ = np.vstack([self.trained_points_, new_points])
        self.trained_labels_ = np.concatenate([self.trained_labels_, new_labels])

        return self

    def _generate_explosive_points(self) -> Tuple[np.ndarray, np.ndarray]:
        new_points = []
        new_labels = []

        for class_label in self.classes_:
            class_mask = (self.trained_labels_ == class_label)
            class_points = self.trained_points_[class_mask]

            if len(class_points) == 0:
                continue

            # Find class center and split energy
            center = np.mean(class_points, axis=0)
            energies = np.random.dirichlet([self.dirichlet_alpha] * self.num_points)
            energies *= self.total_energy  # Scale to total energy

            # Generate new points based on explosion dynamics
            for e_i in energies:
                velocity_magnitude = np.sqrt(2.0 * e_i / self.mass)
                direction = np.random.normal(size=self.dim_)
                norm = np.linalg.norm(direction)
                if norm < 1e-12:  # Avoid divide by zero
                    direction = np.zeros(self.dim_)
                    direction[0] = 1.0
                    norm = 1.0
                direction /= norm

                displacement = velocity_magnitude * self.explosion_time * direction
                noise = np.random.normal(0, self.noise_scale, self.dim_)
                new_point = center + displacement + noise
                new_points.append(new_point)
                new_labels.append(class_label)

        return np.array(new_points, dtype=np.float32), np.array(new_labels, dtype=self.trained_labels_.dtype)

    def _adaptive_radius(self, points: np.ndarray) -> float:
        if len(points) <= 1:
            return 1.0
        if self.radius_adjustment == 'local':
            # Estimate local radius
            n_neighbors = min(5, len(points) - 1)
            nbrs = hnswlib.Index(space='l2', dim=points.shape[1])
            nbrs.init_index(max_elements=len(points), ef_construction=100, M=16)
            nbrs.add_items(points.astype(np.float32))
            labels, distances = nbrs.knn_query(points.astype(np.float32), k=n_neighbors,filter=None)
            return np.median(distances[:, -1])
        else:  # global
            centroid = np.mean(points, axis=0)
            return np.median(np.linalg.norm(points - centroid, axis=1))

    def _build_nn_index(self, max_neighbors: int = 50):
        if self.trained_points_ is None:
            return
        num_elements = self.trained_points_.shape[0]
        index = hnswlib.Index(space='l2', dim=self.dim_)
        index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        index.add_items(self.trained_points_.astype(np.float32))
        index.set_ef(max_neighbors)
        self.nn_index_ = index

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.nn_index_ is None:
            return np.full(X.shape[0], self.classes_[0])
        labels, _ = self.nn_index_.knn_query(X.astype(np.float32), k=1)
        return self.trained_labels_[labels.flatten()]

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Evaluate the model using accuracy score. This is the implementation of the score method
        from sklearn's ClassifierMixin.

        Arguments:
        - X: Test data of shape (n_samples, n_features).
        - y: True labels of shape (n_samples,).
        - sample_weight: Optional sample weights, default is None.

        Returns:
        - Accuracy score of the model on the test data.
        """
        y_pred = self.predict(X)
        if sample_weight is not None:
            # Explicitly cast to float to ensure the return type is float
            return float(np.average(y_pred == y, weights=sample_weight))
        else:
            # Explicitly cast to float to ensure the return type is float
            return float(np.mean(y_pred == y))

