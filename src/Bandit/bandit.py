import random
from abc import abstractmethod

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, DotProduct

from src.Bandit.utils import CosineSimilarityKernel, optimizer


class Bandit:
    def __init__(self, kernel='rbf', alpha=1e-3, length_scale=1, nu=2.5):
        self.is_fitted = False
        # Observations
        self.X = []
        self.y = []

        # Setup GP regressor with appropriate kernel:
        if kernel == "rbf":
            kernel = C(1.0) * RBF(length_scale=length_scale, length_scale_bounds=(1e-4, 1e2))
        elif kernel == "matern":
            kernel = C(1.0) * Matern(length_scale=length_scale, length_scale_bounds=(1e-4, 1e2), nu=nu)
        elif kernel == 'dot_product':
            kernel = C(1.0) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-5, 1e5))
        elif kernel == 'cosine_similarity':
            kernel = C(1.0) * CosineSimilarityKernel()
        else:
            raise ValueError("Invalid kernel specified.")

        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=True,
            n_restarts_optimizer=10,
            optimizer=optimizer
        )

    def update(self, x, reward):
        """
        Update the GP model with a new observation
        Args:
            x: The input passage embedding
            reward: Observed reward from LL<
        """
        self.X.append(x)
        self.y.append(float(reward))
        # Mark that model needs refitting after new data
        self.is_fitted = False

    def get_mean_std(self, candidates):
        """
        Get mean and standard deviation predictions for candidates

        Args:
            candidates: List of candidates to get predictions for

        Returns:
            mu: Mean predictions
            sigma: Standard deviation predictions
        """
        if len(self.X) == 0:
            # If no observations, return default values
            return np.zeros(len(candidates)), np.ones(len(candidates))

        # Predict mean and standard deviation
        mu, sigma = self.gp.predict(candidates, return_std=True)

        return mu, sigma

    def fit(self, candidates):
        # Ensure proper X format for fitting
        X_train = np.stack(self.X)
        if X_train.ndim != 1:
            X_train = X_train.reshape(-1, candidates.shape[1])
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        self.gp.fit(X_train, np.array(self.y))
        self.is_fitted = True

    def get_top_k(self, candidates, k, return_scores=False):
        """
        Get the top k candidates with highest mean values and their corresponding scores
        """
        if not self.is_fitted:
            self.fit(candidates)
        mu, sigma = self.get_mean_std(candidates)
        sorted_indices = np.argsort(-mu)[:k]
        top_k_scores = mu[sorted_indices]

        if return_scores:
            return sorted_indices, top_k_scores
        return sorted_indices

    @abstractmethod
    def select(self, candidates, n=1):
        """
        Select the next candidate passage to retrieve based on the GP model.
        Args:
            candidates: List of candidate passages to select from.
            n: Number of candidates to select.
        Returns:
            The selected candidate passage.
        """
        if len(self.X) == 0:
            return random.sample(range(len(candidates)), n)

        if not self.is_fitted:
            self.fit(candidates)
