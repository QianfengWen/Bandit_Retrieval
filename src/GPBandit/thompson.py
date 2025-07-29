import gpytorch
import torch

from src.GPBandit.basegp import BaseGP


class GPThompson(BaseGP):
    def __init__(self,
                 alpha=0.001,
                 alpha_method=None,
                 train_alpha=False,
                 ard=False,
                 length_scale=1,
                 kernel='rbf',
                 verbose=False):
        super().__init__(alpha, alpha_method, train_alpha, ard, length_scale, kernel, verbose)

    def select(self, candidates, n=1):
        self.fit()

        batch_size = 5000
        samples = []
        c_t = torch.as_tensor(candidates, dtype=self.dtype, device='cpu')
        with gpytorch.settings.fast_pred_var():
            for i in range(0, c_t.size(0), batch_size):
                batch = c_t[i:i + batch_size].to(self.device)
                posterior = self.gp(batch)
                sample = posterior.rsample().to('cpu')  # [batch_size]
                samples.append(sample)

        sampled_values = torch.cat(samples)  # [num_candidates]
        return torch.topk(sampled_values, n).indices