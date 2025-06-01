import numpy as np

from src.Bandit.bandit import Bandit

class GPRandom(Bandit):
    def __init__(self, kernel='rbf', alpha=1e-3, alpha_method=None, length_scale=1, nu=2.5):
        super().__init__(kernel=kernel, alpha=alpha, alpha_method=alpha_method, length_scale=length_scale, nu=nu)

    def select(self, candidates, n=1):
        """
        Select candidates randomly

        Args:
            candidates: List of candidate embeddings to sample from.
            n: Number of candidates to select.

        Returns:
            List of selected candidate indices.
        """
        super().select(candidates, n=n)
        idx = np.random.choice(len(candidates), size=n, replace=False)
        return idx.tolist()