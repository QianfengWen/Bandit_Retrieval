import torch
from torch.distributions import Normal

from src.GPBandit.basegp import BaseGP


class GPPI(BaseGP):
    def __init__(self,
                 xi=0.0,
                 alpha=0.001,
                 alpha_method=None,
                 train_alpha=False,
                 ard=False,
                 length_scale=1,
                 kernel='rbf',
                 verbose=False):
        super().__init__(alpha, alpha_method, train_alpha, ard, length_scale, kernel, verbose)
        self.xi = xi

    def select(self, candidates, n=1):
        self.fit()

        mu, sigma = self.batch_predict(candidates)
        y_max = self.y.max().to('cpu')

        sigma = sigma.clamp_min(1e-9)
        z = (mu - y_max - self.xi) / sigma

        normal = Normal(0, 1)
        pi = normal.cdf(z)
        return torch.topk(pi, n).indices
