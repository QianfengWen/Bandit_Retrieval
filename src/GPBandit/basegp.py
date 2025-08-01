import math

import gpytorch
import torch
from botorch import fit_gpytorch_mll
from gpytorch.constraints import Interval
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.priors import UniformPrior, GammaPrior
from torch.optim import Adam



class _ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, x, y, likelihood, kernel):
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def transform_inputs(self, X):
        return X

class BaseGP:
    def __init__(self,
                 alpha=0.001,
                 alpha_method=None,
                 train_alpha=False,
                 ard=False,
                 length_scale=1,
                 kernel='rbf',
                 verbose=False
                 ):

        self.x = None
        self.y = None
        self.y_norm = None
        self.y_mean = None
        self.y_std = None
        self.noise = None
        self.dtype = torch.double
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

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
        x_t = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        y_t = torch.as_tensor(y, dtype=self.dtype, device=self.device)
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(0)
        if y_t.ndim == 0:
            y_t = y_t.unsqueeze(0)
        noise_t = torch.as_tensor(self.logit2noise(logit), dtype=self.dtype, device=self.device).unsqueeze(0) if logit is not None else None

        if self.gp is None:
            self.x = x_t
            self.y = y_t
            self.y_norm = y_t
            self.noise = noise_t
            self._init_gp(self.x, self.y, self.noise)
        else:
            self.x = torch.cat([self.x, x_t], dim=0)
            self.y = torch.cat([self.y, y_t], dim=0)

            self.y_mean = self.y.mean()
            self.y_std = self.y.std()
            self.y_norm = (self.y - self.y_mean) / (self.y_std + 1e-12)
            # self.y_norm = self.y

            self.gp.set_train_data(self.x, self.y_norm, strict=False)

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

    @torch.no_grad()
    def batch_predict(self, candidates, batch_size=400000):
        self.gp.eval()
        self.likelihood.eval()

        c_t = torch.as_tensor(candidates, dtype=self.dtype, device='cpu')  # keep on CPU initially
        means, stds = [], []

        with gpytorch.settings.fast_pred_var():
            for i in range(0, c_t.size(0), batch_size):
                batch = c_t[i:i + batch_size].to(self.device)

                posterior = self.gp(batch)
                mean = posterior.mean.to('cpu', dtype=self.dtype)  # move back to CPU
                std = posterior.variance.clamp_min(1e-12).sqrt().to('cpu', dtype=self.dtype)

                means.append(mean)
                stds.append(std)

        return torch.cat(means, dim=0), torch.cat(stds, dim=0)


    def fit(self):
        if not self.dirty:
            return
        self.mll.train()
        self.gp.train()
        self.likelihood.train()

        warmup_steps = 100
        adam_lr = 0.05
        adam = Adam(self.mll.parameters(), lr=adam_lr)

        for i in range(warmup_steps):
            adam.zero_grad()
            output = self.gp(self.x)
            loss = -self.mll(output, self.y_norm)
            loss.backward()
            adam.step()

        fit_gpytorch_mll(self.mll)

        self.mll.eval()
        self.gp.eval()
        self.likelihood.eval()

        self.dirty=False


    def _init_gp(self,train_x, train_y, train_noise=None):
        # set kernel (ard, length_scale)
        ls_prior = UniformPrior(1e-5, 30)
        length_constraint = Interval(lower_bound=1e-5, upper_bound=1e2, initial_value=self.length_scale if self.length_scale is not None else 1)
        base_kernel = RBFKernel(lengthscale_prior=ls_prior, length_constraint=length_constraint, ard_num_dims=train_x.shape[-1] if self.ard else None)
        base_kernel.register_constraint('raw_lengthscale', length_constraint)
        kernel = ScaleKernel(base_kernel, outputscale_prior=GammaPrior(2.0, 0.15),).to(self.device, dtype=self.dtype)

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
            if train_noise is None:
                raise ValueError("train_noise must be provided when alpha_method is specified")
            self.likelihood = FixedNoiseGaussianLikelihood(
                noise=train_noise,
                learn_additional_noise=self.train_alpha
            )

        self.likelihood = self.likelihood.to(self.device, dtype=self.dtype)

        self.gp = _ExactGPModel(train_x, train_y, self.likelihood, kernel).to(self.device, self.dtype)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp)

    def get_top_k(self, candidates, k=1, return_scores=False):
        self.fit()
        mean, _ = self.batch_predict(candidates)
        top_k_scores, top_k_indices = torch.topk(mean, k=k)
        top_k_scores = top_k_scores * (self.y_std.to('cpu')+ 1e-12) + self.y_mean.to('cpu') if self.y_mean is not None else top_k_scores
        if return_scores:
            return top_k_indices, top_k_scores
        return top_k_indices

    def select(self, candidates, n=1):
        raise NotImplementedError("This method should be implemented in subclasses")

@torch.no_grad()
def softmax_t(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(x.double(), dim=-1)

@torch.no_grad()
def logit2entropy(logit: list[float] | torch.Tensor) -> float:
    logit = torch.as_tensor(logit, dtype=torch.double)
    p = softmax_t(logit)
    return float(-(p * (p + 1e-12).log()).sum())

@torch.no_grad()
def logit2confidence(logit: list[float] | torch.Tensor) -> float:
    p = softmax_t(torch.as_tensor(logit, dtype=torch.double))
    return float(1.0 - p.max())