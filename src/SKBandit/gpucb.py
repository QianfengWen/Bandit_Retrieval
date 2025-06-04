import numpy as np

from src.SKBandit.bandit import Bandit


class GPUCB(Bandit):
    """
    GP-UCB implementation specifically for retrieval tasks.
    """
    
    def __init__(self, beta=2.0, kernel='rbf', alpha=1e-3, alpha_method=None, length_scale=1, nu=2.5):
        super(GPUCB, self).__init__(kernel=kernel, alpha=alpha, alpha_method=alpha_method, length_scale=length_scale, nu=nu)
        self.beta = beta

    def select(self, candidates, n=1):
        """
        Select the best n candidates to sample next based on GP-UCB
        
        Args:
            candidates: List of candidates (indices or embeddings)
            n: Number of candidates to select
            
        Returns:
            List of selected candidates
        """
        super(GPUCB, self).select(candidates, n=n)

        mu, sigma = self.get_mean_std(candidates)
        ucb = mu + np.sqrt(self.beta) * sigma

        top_indices = np.argsort(-ucb)[:n]
        return top_indices[:n]
    
