import torch

from src.GPBandit.basegp import BaseGP


class GPRandom(BaseGP):
    def __init__(self,
                 kernel='rbf',
                 alpha=0.001,
                 alpha_method=None,
                 train_alpha=False,
                 ard=False,
                 length_scale=1,
                 verbose=False
                 ):
        super().__init__(alpha, alpha_method, train_alpha, ard, length_scale, kernel, verbose)

    def select(self, candidates, n=1):
        # Randomly select n candidates from the provided candidates
        indices = torch.randperm(candidates.size(0))[:n]
        return indices