import math

import gpytorch
import torch
from botorch import fit_gpytorch_mll
from botorch.models.transforms import Standardize
from gpytorch.constraints import Interval
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood


class _ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, x, y, likelihood, kernel):
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPUCB:
    def __init__(self,
                 kernel='rbf',
                 beta=2.0,
                 alpha=0.001,
                 alpha_method=None,
                 train_alpha=False,
                 ard=False,
                 length_scale=1,
                 verbose=False
                 ):

        self.dtype = torch.double
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.beta = (torch.tensor(beta, dtype=self.dtype, device=self.device))
        self.alpha = alpha
        self.alpha_method = alpha_method
        self.train_alpha = train_alpha
        self.ard = ard
        self.length_scale = length_scale

        self.gp, self.likelihood, self.mll, self.opt = None, None, None, None
        self.dirty = False

    def logit2noise(self, logit):
        k = len(logit)
        if self.alpha_method == "raw":
            noise = logit2entropy(logit)
        elif self.alpha_method == "linear":
            noise = self.alpha * logit2entropy(logit) / math.log(k)
        elif self.alpha_method == "confidence":
            noise = self.alpha * logit2confidence(logit)
        elif self.alpha_method is None:
            noise = self.alpha
        else:
            raise ValueError(f"Invalid alpha method {self.alpha_method}")
        noise = max(1e-6, noise)
        return noise

    def update(self, x, y, logit=None):
        x_t = torch.as_tensor(x, dtype=self.dtype, device=self.device).unsqueeze(0)  # (1, D)
        y_t = torch.as_tensor(y, dtype=self.dtype, device=self.device).unsqueeze(0)  # (1, 1)
        noise_t = torch.as_tensor(self.logit2noise(logit), dtype=self.dtype, device=self.device).unsqueeze(0)

        if self.gp is None:
            self._init_gp(x_t, y_t, noise_t)
        else:
            new_x = torch.cat([self.gp.train_inputs[0], x_t], dim=0)
            new_y = torch.cat([self.gp.train_targets, y_t], dim=0)
            self.gp.set_train_data(new_x, new_y, strict=False)

            if isinstance(self.likelihood, FixedNoiseGaussianLikelihood):
                self.likelihood.noise_covar.noise = torch.cat(
                    [self.likelihood.noise_covar.noise, noise_t], dim=0
                )
        self.dirty = True

    @torch.no_grad()
    def predict(self, candidates):
        self.gp.eval()
        self.likelihood.eval()
        c_t = torch.as_tensor(candidates, dtype=self.dtype, device=self.device)
        with gpytorch.settings.fast_pred_var():
            posterior = self.gp(c_t)
        mean = posterior.mean.to(self.dtype)
        std = posterior.variance.clamp_min(1e-12).sqrt().to(self.dtype)

        return mean, std

    def fit(self,):
        if not self.dirty:
            return

        self.gp.train()
        self.likelihood.train()

        fit_gpytorch_mll(self.mll)

        self.dirty=False


    def _init_gp(self,train_x, train_y, train_noise=None):
        # set kernel (ard, length_scale)
        length_constraint = Interval(lower_bound=1e-4, upper_bound=1e2, initial_value=self.length_scale if self.length_scale is not None else 1)
        kernel = ScaleKernel(RBFKernel(
            lengthscale_constraint=length_constraint,
            ard_num_dims=train_x.shape[1] if self.ard else None,
        ).to(self.device, dtype=self.dtype))


        # set likelihood (alpha)
        if self.alpha_method is None:
            if self.train_alpha:
                self.likelihood = GaussianLikelihood()
                self.likelihood.raw_noise.data.fill_(torch.tensor(self.alpha).log())
            else:
                self.likelihood = FixedNoiseGaussianLikelihood(
                    noise= torch.full_like(train_y, fill_value=self.alpha),
                    learn_additional_noise=False)
        else:
            self.likelihood = FixedNoiseGaussianLikelihood(
                noise=train_noise,
                learn_additional_noise=self.train_alpha
            )

        self.likelihood = self.likelihood.to(self.device, dtype=self.dtype)

        self.gp = _ExactGPModel(train_x, train_y, self.likelihood, kernel).to(self.device, self.dtype)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp)

    def get_top_k(self, candidates, k=1, return_scores=False):
        self.fit()
        mean, _ = self.predict(candidates)
        top_k_scores, top_k_indices = torch.topk(mean, k=k)
        if return_scores:
            return top_k_indices, top_k_scores
        return top_k_indices

    def select(self, candidates, n=1):
        self.fit()
        mu, sigma = self.predict(candidates)
        ucb = mu + torch.sqrt(self.beta) * sigma
        return torch.topk(ucb, n).indices

@torch.no_grad()
def softmax_t(x: torch.Tensor) -> torch.Tensor:
    x = x - x.max()
    return x.exp() / x.exp().sum()

@torch.no_grad()
def logit2entropy(logit: list[float] | torch.Tensor) -> float:
    logit = torch.as_tensor(logit, dtype=torch.double)
    p = softmax_t(logit)
    return float(-(p * (p + 1e-12).log()).sum())

@torch.no_grad()
def logit2confidence(logit: list[float] | torch.Tensor) -> float:
    p = softmax_t(torch.as_tensor(logit, dtype=torch.double))
    return float(1.0 - p.max())