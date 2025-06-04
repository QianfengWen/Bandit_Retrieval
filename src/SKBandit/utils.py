import numpy as np
import scipy
from sklearn.gaussian_process.kernels import NormalizedKernelMixin, Kernel


def optimizer(obj_func, x0, bounds):
    res = scipy.optimize.minimize(
        obj_func, x0, bounds=bounds, method="L-BFGS-B", jac=True,
        options={'maxiter': 100})
    return res.x, res.fun


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


