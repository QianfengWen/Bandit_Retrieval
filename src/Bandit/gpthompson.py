import numpy as np

from src.Bandit.bandit import Bandit

class GPThompson(Bandit):
    def __init__(self, kernel='rbf', alpha=1e-3, length_scale=1, nu=2.5):
        super().__init__(kernel, alpha, length_scale, nu)

    def select(self, candidates, n=1):
        """
        Select candidates based on Thompson sampling.

        Args:
            candidates: List of candidate embeddings to sample from.
            n: Number of candidates to select.

        Returns:
            List of selected candidate indices.
        """
        super().select(candidates, n=n)

        samples = self.gp.sample_y(candidates,
                                   n_samples=1).ravel()

        top_idx = np.argsort(-samples)[:n]
        return top_idx
