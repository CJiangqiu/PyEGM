import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_X_y
from typing import Literal, Tuple
from scipy.stats import kurtosis


class PyEGM(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 num_points: int = 100,
                 max_samples: int = 1000,
                 explosion_factor: float = 0.5,
                 radius_adjustment: Literal['local', 'global'] = 'local',
                 generation_method: Literal['auto', 'hypersphere', 'gaussian'] = 'auto',
                 center_pull: float = 0.5,
                 decay_factor: float = 0.9,
                 new_data_weight: float = 0.5,
                 generate_in_partial_fit: bool = True):
        """
        PyEGM: Explosive Generative Model for Classification

        This classifier implements an explosive generative model for incremental learning.
        It generates new training points during each iteration and (optionally) during partial_fit,
        and dynamically adjusts its generation strategy based on the input data distribution.

        Key Parameters:
        - num_points: Number of new points generated per iteration.
        - max_samples: Maximum number of samples to retain.
        - explosion_factor: Coefficient controlling the explosion magnitude.
        - radius_adjustment: Strategy for adjusting the radius ('local' or 'global').
        - generation_method: Generation strategy:
            - 'auto': Automatically choose the strategy based on data characteristics.
            - 'hypersphere': Generate samples uniformly on a hypersphere.
            - 'gaussian': Generate samples using a Gaussian distribution with center bias.
        - center_pull: (For Gaussian generation) Degree to pull generated points towards the center (0.0 to 1.0).
        - decay_factor: Coefficient for sample weight decay in incremental learning.
        - new_data_weight: Initial weight assigned to newly arrived data in partial_fit.
        - generate_in_partial_fit: Whether to generate new points in each call to partial_fit.
        """
        self.num_points = num_points
        self.max_samples = max_samples
        self.explosion_factor = explosion_factor
        self.radius_adjustment = radius_adjustment
        self.generation_method = generation_method
        self.center_pull = center_pull
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

        if self.generation_method == 'auto':
            chosen_method = self._auto_choose_generation_method()
        else:
            chosen_method = self.generation_method

        if chosen_method == 'hypersphere':
            new_points, new_labels = self._generate_hypersphere_points()
        elif chosen_method == 'gaussian':
            new_points, new_labels = self._generate_gaussian_points()
        else:
            raise ValueError(f"Unsupported generation method: {chosen_method}")

        # Merge new points with training data
        self.trained_points_ = np.vstack([self.trained_points_, new_points])
        self.trained_labels_ = np.concatenate([self.trained_labels_, new_labels])


        new_weights = np.ones(len(new_points))
        self.sample_weights_ = np.concatenate([self.sample_weights_, new_weights])

        self._build_nn_index()

        return self

    def _auto_choose_generation_method(self) -> str:
        """
        Automatically choose a generation method based on data distribution.

        Enhanced heuristic:
          1) For each class, compute the covariance eigenvalue ratio (max/min).
             If the ratio is near 1 -> roughly spherical, otherwise elongated.
          2) Additionally measure kurtosis to see if data is more "peaky" vs. heavy-tailed.
          3) Combine ratio & kurtosis heuristics:
             - If ratio < 2 and avg_kurtosis < 3 => 'hypersphere'
             - Else => 'gaussian'
        """
        ratios = []
        kurts = []
        for class_label in self.classes_:
            class_mask = self.trained_labels_ == class_label
            class_points = self.trained_points_[class_mask]
            if len(class_points) < 2:
                continue

            # Covariance ratio
            cov = np.cov(class_points, rowvar=False)
            eigvals = np.linalg.eigvalsh(cov)
            eps = 1e-6
            ratio = np.max(eigvals) / (np.min(eigvals) + eps)
            ratios.append(ratio)

            # Kurtosis
            kurt_vals = []
            for dim_i in range(class_points.shape[1]):
                k = kurtosis(class_points[:, dim_i], fisher=False)
                kurt_vals.append(k)
            kurts.append(np.mean(kurt_vals))

        if len(ratios) == 0:
            return 'hypersphere'

        avg_ratio = np.mean(ratios)
        avg_kurt = np.mean(kurts) if len(kurts) else 3.0

        if avg_ratio < 2 and avg_kurt < 3:
            return 'hypersphere'
        else:
            return 'gaussian'

    def _adaptive_radius(self, points: np.ndarray) -> float:
        if len(points) <= 1:
            return 1.0

        if self.radius_adjustment == 'local':
            n_neighbors = min(5, len(points) - 1)
            nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
            distances, _ = nbrs.kneighbors(points)
            base_radius = np.median(distances[:, -1])
        else:  # global
            centroid = np.mean(points, axis=0)
            base_radius = np.median(np.linalg.norm(points - centroid, axis=1))

        dim_penalty = np.sqrt(self.dim_) if self.dim_ > 10 else 1.0
        return base_radius * self.explosion_factor / dim_penalty

    def _build_nn_index(self, max_neighbors: int = 50):
        if self.trained_points_ is None:
            return
        n_neighbors = min(max_neighbors, len(self.trained_points_))
        self.nn_index_ = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(self.trained_points_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.nn_index_ is None:
            return np.full(X.shape[0], self.classes_[0])
        _, indices = self.nn_index_.kneighbors(X, n_neighbors=1)
        return self.trained_labels_[indices.flatten()]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.trained_points_ is None:
            return np.zeros((len(X), len(self.classes_)))
        n_neighbors = min(50, len(self.trained_points_))
        distances, indices = self.nn_index_.kneighbors(X, n_neighbors=n_neighbors)
        proba = []
        for i in range(len(X)):
            in_radius = distances[i] <= self.radius_
            if np.any(in_radius):
                weights = self.sample_weights_[indices[i][in_radius]]
                counts = np.bincount(self.trained_labels_[indices[i][in_radius]],
                                     weights=weights,
                                     minlength=len(self.classes_))
            else:
                closest = indices[i][0]
                counts = np.zeros(len(self.classes_))
                counts[self.trained_labels_[closest]] = 1.0
            proba.append(counts / (counts.sum() + 1e-12))
        return np.array(proba)

    def _generate_hypersphere_points(self) -> Tuple[np.ndarray, np.ndarray]:
        new_points = []
        new_labels = []
        for class_label in self.classes_:
            class_mask = self.trained_labels_ == class_label
            class_points = self.trained_points_[class_mask]
            class_weights = self.sample_weights_[class_mask]
            if len(class_points) == 0:
                continue
            n_points = max(1, int(self.num_points * np.sqrt(class_mask.mean())))
            center_indices = np.random.choice(
                len(class_points),
                size=min(n_points, len(class_points)),
                p=class_weights / class_weights.sum()
            )
            for center in class_points[center_indices]:
                direction = np.random.normal(size=self.dim_)
                direction /= (np.linalg.norm(direction) + 1e-12)
                radius = self._get_effective_radius()
                new_points.append(center + radius * direction)
                new_labels.append(class_label)
        if not new_points:
            return np.empty((0, self.dim_)), np.array([])
        return np.array(new_points), np.array(new_labels, dtype=self.trained_labels_.dtype)

    def _generate_gaussian_points(self) -> Tuple[np.ndarray, np.ndarray]:
        new_points = []
        new_labels = []
        for class_label in self.classes_:
            class_mask = self.trained_labels_ == class_label
            class_points = self.trained_points_[class_mask]
            class_weights = self.sample_weights_[class_mask]
            if len(class_points) == 0:
                continue
            n_points = max(1, int(self.num_points * np.sqrt(class_mask.mean())))
            center_indices = np.random.choice(
                len(class_points),
                size=min(n_points, len(class_points)),
                p=class_weights / class_weights.sum()
            )
            for center in class_points[center_indices]:
                direction = np.random.normal(size=self.dim_)
                direction /= (np.linalg.norm(direction) + 1e-12)
                sigma = self._get_effective_radius()
                distance = np.abs(np.random.normal(loc=0, scale=sigma))
                adjusted_distance = distance * (1 - self.center_pull) + self.center_pull * sigma
                new_points.append(center + adjusted_distance * direction)
                new_labels.append(class_label)
        if not new_points:
            return np.empty((0, self.dim_)), np.array([])
        return np.array(new_points), np.array(new_labels, dtype=self.trained_labels_.dtype)

    def _get_effective_radius(self) -> float:
        if len(self.trained_points_) > 1:
            distances = np.linalg.norm(self.trained_points_ - np.mean(self.trained_points_, axis=0), axis=1)
            density = np.median(distances)
            density_factor = 1.0 / (1.0 + density)
        else:
            density_factor = 1.0
        return self.radius_ * density_factor

    def _prune_samples(self):
        if len(self.trained_points_) <= self.max_samples:
            return
        keep_idx = np.argsort(self.sample_weights_)[-self.max_samples:]
        self.trained_points_ = self.trained_points_[keep_idx]
        self.trained_labels_ = self.trained_labels_[keep_idx]
        self.sample_weights_ = self.sample_weights_[keep_idx]
        self._build_nn_index()

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> 'PyEGM':
        """
        Incrementally fit the model with new data.
        Optionally generate new points based on the newly arrived samples.
        """
        if self.trained_points_ is None:
            return self.fit(X, y)

        X, y = check_X_y(X, y)

        self.sample_weights_ *= self.decay_factor

        current_new_weights = np.full(len(X), self.new_data_weight)

        self.trained_points_ = np.vstack([self.trained_points_, X])
        self.trained_labels_ = np.concatenate([self.trained_labels_, y])
        self.sample_weights_ = np.concatenate([self.sample_weights_, current_new_weights])

        if self.generate_in_partial_fit and len(X) > 0:
            self.radius_ = self._adaptive_radius(self.trained_points_)
            if self.generation_method == 'auto':
                chosen_method = self._auto_choose_generation_method()
            else:
                chosen_method = self.generation_method

            if chosen_method == 'hypersphere':
                new_points, new_labels = self._generate_hypersphere_points()
            elif chosen_method == 'gaussian':
                new_points, new_labels = self._generate_gaussian_points()
            else:
                raise ValueError(f"Unsupported generation method: {chosen_method}")

            self.trained_points_ = np.vstack([self.trained_points_, new_points])
            self.trained_labels_ = np.concatenate([self.trained_labels_, new_labels])
            synth_weights = np.full(len(new_points), self.new_data_weight)
            self.sample_weights_ = np.concatenate([self.sample_weights_, synth_weights])

        self._prune_samples()
        self.radius_ = self._adaptive_radius(self.trained_points_)
        self._build_nn_index()

        return self
