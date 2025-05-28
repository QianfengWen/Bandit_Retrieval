import numpy as np

from src.Bandit.bandit import Bandit

class GPRandom(Bandit):
    def __init__(self, kernel='rbf', alpha=1e-3, length_scale=1, nu=2.5):
        super().__init__(kernel, alpha, length_scale, nu)

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