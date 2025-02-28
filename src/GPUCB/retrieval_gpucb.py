import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class RetrievalGPUCB:
    """
    GP-UCB implementation specifically for retrieval tasks.
    Supports both indices-based and embeddings-based approaches.
    """
    
    def __init__(self, beta=2.0, is_embeddings_based=False):
        """
        Initialize the GP-UCB algorithm for retrieval
        
        Args:
            beta: Exploration parameter that balances exploitation vs exploration
            is_embeddings_based: Whether to use embeddings (True) or indices (False) as features
        """
        self.beta = beta
        self.is_embeddings_based = is_embeddings_based
        
        # Observations
        self.X = []  # Features of observed points (indices or embeddings)
        self.y = []  # Rewards (relevance scores) observed so far
        
        # Setup GP regressor with appropriate kernel
        if is_embeddings_based:
            # For embeddings, we use RBF kernel with learned length scale
            kernel = C(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        else:
            # For indices, simpler kernel as we just have 1D or 2D input
            kernel = C(1.0) * RBF(length_scale=1.0)
            
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,  # Small noise to avoid numerical issues
            normalize_y=True,
            n_restarts_optimizer=5
        )
        
    def select(self, candidates, n=1):
        """
        Select the best n candidates to sample next based on GP-UCB
        
        Args:
            candidates: List of candidates (indices or embeddings)
            n: Number of candidates to select
            
        Returns:
            List of selected candidates
        """
        if not self.X:
            # Cold start: if no observations yet, just return the first n candidates
            return candidates[:n]
        
        # Convert candidates to appropriate format
        X_candidates = self._process_candidates(candidates)
        
        # Fit GP model to observed data
        self.gp.fit(np.array(self.X), np.array(self.y))
        
        # Predict mean and standard deviation for all candidates
        mu, sigma = self.gp.predict(X_candidates, return_std=True)
        
        # Calculate upper confidence bound
        ucb = mu + np.sqrt(self.beta) * sigma
        
        # Get indices of candidates with highest UCB values
        top_indices = np.argsort(-ucb)[:n]
        
        # Return selected candidates
        if n == 1:
            return top_indices[0]
        return top_indices[:n]
    
    def update(self, x, reward):
        """
        Update the GP model with a new observation
        
        Args:
            x: The feature of the selected candidate (index or embedding)
            reward: The observed reward (relevance score)
        """
        if self.is_embeddings_based:
            # For embeddings, store as is
            self.X.append(x)
        else:
            # For indices, convert to array format
            self.X.append([x] if isinstance(x, (int, float)) else x)
            
        self.y.append(reward)
    
    def _process_candidates(self, candidates):
        """
        Process candidates into appropriate format for GP
        
        Args:
            candidates: List of candidates (indices or embeddings)
            
        Returns:
            Numpy array of processed candidates
        """
        if self.is_embeddings_based:
            # For embeddings, just return the embeddings as a numpy array
            return np.array(candidates)
        else:
            # For indices, convert to 2D array
            return np.array([[c] if isinstance(c, (int, float)) else c for c in candidates])
            
    def get_mean_std(self, candidates):
        """
        Get mean and standard deviation predictions for candidates
        
        Args:
            candidates: List of candidates to get predictions for
            
        Returns:
            mu: Mean predictions
            sigma: Standard deviation predictions
        """
        if not self.X:
            # If no observations, return default values
            return np.zeros(len(candidates)), np.ones(len(candidates))
        
        # Convert candidates to appropriate format
        X_candidates = self._process_candidates(candidates)
        
        # Fit GP model to observed data
        self.gp.fit(np.array(self.X), np.array(self.y))
        
        # Predict mean and standard deviation
        mu, sigma = self.gp.predict(X_candidates, return_std=True)
        
        return mu, sigma 