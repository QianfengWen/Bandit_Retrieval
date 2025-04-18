import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, DotProduct, Kernel, NormalizedKernelMixin
import scipy
import numpy as np
random_seed = 42   

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



class GaussianProcess:
    """
    Gaussian Process implementation for retrieval tasks.
    """

    def __init__(self, kernel="rbf", llm_budget=0):
        """
        Initialize the Gaussian Process model.

        Args:
            kernel (str): The type of kernel to use ('rbf', 'dot_product', 'cosine_similarity').
        """
        self.is_fitted = False
        self.X = []
        self.y = []
        
        if kernel == "rbf":
            self.kernel = C(1.0, constant_value_bounds=(1e-5, 1e5)) * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5))
        elif kernel == 'dot_product':
            self.kernel = C(1.0, constant_value_bounds=(1e-5, 1e5)) * DotProduct()
        elif kernel == 'cosine_similarity':
            self.kernel = C(1.0, constant_value_bounds=(1e-5, 1e5)) * CosineSimilarityKernel()
        else:
            raise ValueError(f"Unsupported kernel type: {kernel}")

        def optimizer(obj_func, x0, bounds):
            res = scipy.optimize.minimize(
                obj_func, x0, bounds=bounds, method="L-BFGS-B", jac=True,
                options={'maxiter': 100})
            return res.x, res.fun
        
        if llm_budget > 0:
            self.gp = GaussianProcessRegressor(
                kernel=self.kernel,
                alpha=1e-3,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=42,
                optimizer=optimizer
            )
        else:
            self.gp = GaussianProcessRegressor(
                kernel=self.kernel
            )
    
    def update(self, x, reward):
        """
        Update the GP model with a new observation
        
        Args:
            x: The feature of the selected candidate (index or embedding)
            reward: The observed reward (relevance score)
        """
        x_processed = self._process_x(x)
        self.X.append(x_processed) 
        self.y.append(reward)
        self.is_fitted = False
    
    def _process_x(self, candidates):
        """
        Process candidates into appropriate format for GP
        
        Args:
            candidates: List of candidates (indices or embeddings), already a numpy array
            
        Returns:
            Numpy array of processed candidates
        """
        if not isinstance(candidates, np.ndarray):
            candidates = np.array(candidates)
        
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)
            
        return candidates
    
    def _process_candidates(self, candidates):
        if isinstance(candidates, list):
            candidates = np.array(candidates)

        if candidates.ndim == 1:
            candidates = candidates.reshape(-1, 1)
        return candidates
            

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
            return np.zeros(len(candidates)), np.ones(len(candidates))
        
        X_candidates = self._process_candidates(candidates)
        
        if not self.is_fitted:
            X_train = np.stack(self.X)
            if X_train.ndim != 1:
                X_train = X_train.reshape(-1, X_candidates.shape[1])
            if X_train.ndim == 1:
                X_train = X_train.reshape(-1, 1)
            
            self.gp.fit(X_train, np.array(self.y))
            self.is_fitted = True
        
        mu, sigma = self.gp.predict(X_candidates, return_std=True)
        
        return mu, sigma
    
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

