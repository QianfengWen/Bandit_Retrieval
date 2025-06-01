import numpy as np
from scipy.stats import norm

from src.Bandit.bandit import Bandit


class GPEI(Bandit):
    def __init__(self, kernel='rbf', alpha=1e-3, alpha_method=None, length_scale=1, nu=2.5, xi=0.01):
        super().__init__(kernel=kernel, alpha=alpha, alpha_method=alpha_method, length_scale=length_scale, nu=nu)

        self.xi = xi

    def select(self, candidates, n=1):
        super().select(candidates, n=n)

        mu, sigma = self.gp.predict(candidates, return_std=True)
        sigma = np.maximum(sigma, 1e-9)  # Avoid division by zero

        y_max = np.max(self.y)
        z = (mu - y_max - self.xi) / sigma
        ei = (mu - y_max - self.xi) * norm.cdf(z) + sigma * norm.pdf(z)

        top_idx = np.argsort(-ei)[:n]
        return top_idx
