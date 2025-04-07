import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_X_y
from typing import Literal

class PyEGM(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 num_points: int = 100,
                 max_samples: int = 1000,
                 explosion_factor: float = 0.5,
                 radius_adjustment: Literal['local', 'global'] = 'local',
                 generation_method: Literal['hypersphere', 'gaussian'] = 'hypersphere',
                 decay_factor: float = 0.9):
        """
        A fully functional PyEGM classifier.

        完整可运行的PyEGM分类器

        Parameters:
        - num_points: The number of new points to generate per iteration.
        - max_samples: The maximum number of samples to retain.
        - explosion_factor: The explosion factor coefficient.
        - radius_adjustment: The strategy for adjusting the radius ('local' or 'global').
        - generation_method: The point generation method ('hypersphere' or 'gaussian').
        - decay_factor: The sample decay coefficient.
        """
        self.num_points = num_points
        self.max_samples = max_samples
        self.explosion_factor = explosion_factor
        self.radius_adjustment = radius_adjustment
        self.generation_method = generation_method
        self.decay_factor = decay_factor

        # State variables
        # 状态变量
        self.trained_points_ = None
        self.trained_labels_ = None
        self.sample_weights_ = None
        self.radius_ = None
        self.dim_ = None
        self.classes_ = None
        self.nn_index_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PyEGM':
        """
        Fit the model to the data.

        训练模型

        Parameters:
        - X: Input data.
        - y: Labels.

        Returns:
        - self: The trained model.
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.dim_ = X.shape[1]

        # Initialization
        # 初始化
        self.trained_points_ = X
        self.trained_labels_ = y
        self.sample_weights_ = np.ones(len(X))
        self.radius_ = self._adaptive_radius(self.trained_points_)

        # Generate new points based on the chosen generation method
        # 根据选择的生成方法生成新点
        if self.generation_method == 'hypersphere':
            new_points = self._generate_hypersphere_points()
        elif self.generation_method == 'gaussian':
            new_points = self._generate_gaussian_points()
        else:
            raise ValueError(f"Unsupported generation method: {self.generation_method}")

        # Merge new points with training data
        # 将生成的新点与训练数据合并
        self.trained_points_ = np.vstack([self.trained_points_, new_points])
        self.trained_labels_ = np.concatenate([self.trained_labels_, np.zeros(len(new_points))])  # New points' labels can be set as needed


        self._build_nn_index()

        return self

    def _adaptive_radius(self, points: np.ndarray) -> float:
        """
        Calculate the adaptive radius.

        计算自适应半径

        Parameters:
        - points: The training points.

        Returns:
        - float: The calculated radius.
        """
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

        # Dimensional adjustment
        # 维度调整
        dim_penalty = np.sqrt(self.dim_) if self.dim_ > 10 else 1.0
        return base_radius * self.explosion_factor / dim_penalty

    def _build_nn_index(self, max_neighbors: int = 50):
        """
        Build the nearest neighbors index.

        构建最近邻索引

        Parameters:
        - max_neighbors: The maximum number of neighbors to consider.
        """
        if self.trained_points_ is None:
            return

        n_neighbors = min(max_neighbors, len(self.trained_points_))
        self.nn_index_ = NearestNeighbors(
            n_neighbors=n_neighbors,
            algorithm='auto'
        ).fit(self.trained_points_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the given data.

        预测给定数据的类别标签

        Parameters:
        - X: Input data to predict.

        Returns:
        - np.ndarray: Predicted class labels.
        """
        if self.nn_index_ is None:
            return np.full(X.shape[0], self.classes_[0])

        _, indices = self.nn_index_.kneighbors(X, n_neighbors=1)
        return self.trained_labels_[indices.flatten()]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class probabilities for the given data.

        预测给定数据的类别概率

        Parameters:
        - X: Input data to predict.

        Returns:
        - np.ndarray: Predicted probabilities.
        """
        if self.trained_points_ is None:
            return np.zeros((len(X), len(self.classes_)))

        n_neighbors = min(50, len(self.trained_points_))
        distances, indices = self.nn_index_.kneighbors(X, n_neighbors=n_neighbors)

        proba = []
        for i in range(len(X)):
            in_radius = distances[i] <= self.radius_
            if np.any(in_radius):
                weights = self.sample_weights_[indices[i][in_radius]]
                counts = np.bincount(
                    self.trained_labels_[indices[i][in_radius]],
                    weights=weights,
                    minlength=len(self.classes_)
                )
            else:
                closest = indices[i][0]
                counts = np.zeros(len(self.classes_))
                counts[self.trained_labels_[closest]] = 1.0

            proba.append(counts / counts.sum())

        return np.array(proba)

    def _generate_hypersphere_points(self) -> np.ndarray:
        """
        Generate points on a hypersphere.

        在超球面上生成点

        This function generates new points uniformly distributed on a hypersphere around each class center.

        该方法生成新的点，这些点在每个类别的中心周围均匀分布在超球面上。

        Returns:
        - np.ndarray: An array of generated points.
        返回值：
        - np.ndarray: 生成的点的数组。
        """
        new_points = []
        for class_label in self.classes_:
            class_mask = self.trained_labels_ == class_label
            class_points = self.trained_points_[class_mask]
            class_weights = self.sample_weights_[class_mask]

            if len(class_points) == 0:
                continue

            # Generate points proportionally to the class
            # 按类别比例生成点
            n_points = max(1, int(self.num_points * np.sqrt(class_mask.mean())))
            center_indices = np.random.choice(
                len(class_points),
                size=min(n_points, len(class_points)),
                p=class_weights / class_weights.sum()
            )

            for center in class_points[center_indices]:
                # Generate random direction
                # 生成随机方向
                direction = np.random.normal(size=self.dim_)
                direction /= np.linalg.norm(direction)

                # Apply effective radius
                # 应用有效半径
                radius = self._get_effective_radius()
                new_points.append(center + radius * direction)

        return np.array(new_points) if new_points else np.empty((0, self.dim_))

    def _generate_gaussian_points(self) -> np.ndarray:
        """
        Generate points based on a Gaussian distribution.

        在高斯分布上生成点

        This function generates new points distributed according to a Gaussian distribution centered around each class center.

        该方法根据高斯分布生成新的点，这些点围绕每个类别的中心点分布。

        Returns:
        - np.ndarray: An array of generated points.
        返回值：
        - np.ndarray: 生成的点的数组。
        """
        new_points = []
        for class_label in self.classes_:
            class_mask = self.trained_labels_ == class_label
            class_points = self.trained_points_[class_mask]
            class_weights = self.sample_weights_[class_mask]

            if len(class_points) == 0:
                continue

            # Generate points proportionally to the class
            # 按类别比例生成点
            n_points = max(1, int(self.num_points * np.sqrt(class_mask.mean())))
            center_indices = np.random.choice(
                len(class_points),
                size=min(n_points, len(class_points)),
                p=class_weights / class_weights.sum()
            )

            for center in class_points[center_indices]:
                # Generate points based on Gaussian distribution
                # 基于高斯分布生成点
                direction = np.random.normal(size=self.dim_)  # Gaussian distribution
                direction /= np.linalg.norm(direction)  # Normalize direction

                # Apply effective radius
                # 应用有效半径
                radius = self._get_effective_radius()
                new_points.append(center + radius * direction)

        return np.array(new_points) if new_points else np.empty((0, self.dim_))

    def _get_effective_radius(self) -> float:
        """
        Get the effective radius.

        获取有效半径

        Returns:
        - float: The effective radius.
        返回值：
        - float: 有效半径。
        """
        if len(self.trained_points_) > 1:
            distances = np.linalg.norm(
                self.trained_points_ - np.mean(self.trained_points_, axis=0),
                axis=1
            )
            density = np.median(distances)
            density_factor = 1.0 / (1.0 + density)
        else:
            density_factor = 1.0

        return self.radius_ * density_factor

    def _prune_samples(self):
        """
        Prune samples to keep the number of samples under the maximum limit.

        修剪样本以保持样本数在最大限制内

        """
        if len(self.trained_points_) <= self.max_samples:
            return

        keep_idx = np.argsort(self.sample_weights_)[-self.max_samples:]
        self.trained_points_ = self.trained_points_[keep_idx]
        self.trained_labels_ = self.trained_labels_[keep_idx]
        self.sample_weights_ = self.sample_weights_[keep_idx]

        # Rebuild index
        # 重建索引
        self._build_nn_index()

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> 'PyEGM':
        """
        Incrementally fit the model with new data.

        增量学习，使用新数据训练模型

        Parameters:
        - X: Input data.
        - y: Labels.

        Returns:
        - self: The trained model.
        """
        if self.trained_points_ is None:
            return self.fit(X, y)

        X, y = check_X_y(X, y)

        # Apply decay
        # 应用衰减
        self.sample_weights_ *= self.decay_factor

        # Add new samples
        # 添加新样本
        self.trained_points_ = np.vstack([self.trained_points_, X])
        self.trained_labels_ = np.concatenate([self.trained_labels_, y])
        self.sample_weights_ = np.concatenate([self.sample_weights_,
                                               np.ones(len(X))])

        # Prune and update
        # 修剪并更新
        self._prune_samples()
        self.radius_ = self._adaptive_radius(self.trained_points_)
        self._build_nn_index()

        return self
