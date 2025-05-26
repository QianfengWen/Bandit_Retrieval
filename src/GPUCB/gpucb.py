from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, RationalQuadratic, ExpSineSquared, DotProduct
import scipy
import time
AQ_FUNCS = ['ucb', 'random', 'greedy']

from sklearn.gaussian_process.kernels import Kernel, NormalizedKernelMixin
import numpy as np

class CosineSimilarityKernel(NormalizedKernelMixin, Kernel):
    def __init__(self):
        pass
    
    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        
        # Compute cosine similarity
        X_norm = np.linalg.norm(X, axis=1, keepdims=True)
        Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
        similarity = (X @ Y.T) / (X_norm * Y_norm.T)
        
        if eval_gradient:
            # No gradient available for this kernel
            return similarity, np.empty((X.shape[0], X.shape[0], 0))
        return similarity
    
    def diag(self, X):
        return np.ones(X.shape[0])

    def is_stationary(self):
        return False


random_seed = 42   
class GPUCB:
    """
    GP-UCB implementation specifically for retrieval tasks.
    """
    
    def __init__(self, beta=2.0, kernel='rbf', alpha=1e-3, length_scale=1, acquisition_function='ucb', nu=2.5):
        self.is_fitted = False
        # Observations
        self.X = []  # Features of observed points (indices or embeddings)
        self.y = []  # Rewards (relevance scores) observed so far
        
        # Setup GP regressor with appropriate kernel: 
        if kernel == "rbf":
            kernel = C(1.0) * RBF(length_scale=length_scale, length_scale_bounds=(1e-4, 1e2))
        elif kernel == "matern":
            kernel = C(1.0) * Matern(length_scale=length_scale, length_scale_bounds=(1e-3, 1e2), nu=nu)
        elif kernel == 'dot_product':
            kernel = C(1.0) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-5, 1e5))
        elif kernel == 'cosine_similarity':
            kernel = C(1.0) * CosineSimilarityKernel()
        else:
            raise ValueError("Invalid kernel specified.")

        def optimizer(obj_func, x0, bounds):
            res = scipy.optimize.minimize(
                obj_func, x0, bounds=bounds, method="L-BFGS-B", jac=True,
                options={'maxiter': 100})
            return res.x, res.fun
            
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=True,
            n_restarts_optimizer=10,
            optimizer=optimizer
        )

        if acquisition_function in AQ_FUNCS:
            self.acquisition_function = acquisition_function
            if self.acquisition_function == 'ucb':
                self.beta = beta
        else:
            raise ValueError(f"Invalid acquisition function: {acquisition_function}. ")

    def update(self, x, reward):
        """
        Update the GP model with a new observation
        
        Args:
            x: The feature of the selected candidate (index or embedding)
            reward: The observed reward (relevance score)
        """
        self.X.append(x)  # Add the flattened array
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
        if not self.X:
            # If no observations, return default values
            return np.zeros(len(candidates)), np.ones(len(candidates))

        # Fit GP model only if it hasn't been fitted since last update
        if not self.is_fitted:
            # Ensure proper X format for fitting
            X_train = np.stack(self.X)
            if X_train.ndim != 1:
                X_train = X_train.reshape(-1, candidates.shape[1])
            if X_train.ndim == 1:
                X_train = X_train.reshape(-1, 1)
            
            self.gp.fit(X_train, np.array(self.y))
            self.is_fitted = True
        
        # Predict mean and standard deviation
        mu, sigma = self.gp.predict(candidates, return_std=True)
        
        return mu, sigma

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
            return np.arange(min(n, len(candidates)))
        
        if self.acquisition_function == 'random':
            return np.random.choice(len(candidates), n, replace=False)
        
        # Fit GP model only if it hasn't been fitted since last update
        if not self.is_fitted:
            # Ensure proper X format for fitting
            x_train = np.stack(self.X)
            if x_train.ndim != 1:
                x_train = x_train.reshape(-1, candidates.shape[1])
            if x_train.ndim == 1:
                x_train = x_train.reshape(-1, 1)
                
            # Fit the model
            self.gp.fit(x_train, np.array(self.y))
            self.is_fitted = True
        
        # Predict mean and standard deviation for all candidates
        mu, sigma = self.gp.predict(candidates, return_std=True)
        
        # Calculate upper confidence bound
        ucb = mu + np.sqrt(self.beta) * sigma
        
        # Get indices of candidates with highest UCB values
        top_indices = np.argsort(-ucb)[:n]
        
        # Return selected candidates
        return top_indices[:n]
    
    def get_top_k(self, candidates, k, return_scores=False):
        """
        Get the top k candidates with highest mean values and their corresponding scores
        """
        mu, sigma = self.get_mean_std(candidates)
        sorted_indices = np.argsort(-mu)[:k]
        top_k_scores = mu[sorted_indices]

        if return_scores:
            return sorted_indices, top_k_scores
        return sorted_indices
    
