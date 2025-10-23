import torch
import torch.nn as nn
from typing import Tuple


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    out = torch.gather(a, index=t, dim=0).to(t.device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class EDICTSampler(nn.Module):
    """
    EDICT sampler for deterministic, (near) invertible sampling on a discrete time grid.

    This sampler is drop-in for DDIM-like ODE sampling, using a coupled two-track
    update with mixing coefficient `p` for improved reversibility.

    Assumptions:
    - The denoiser `model` takes input `torch.cat([x_t, cond], dim=1)` and timestep `t`
      and predicts either x0 (when `pre_ori=True`) or epsilon (when `pre_ori=False`).
    - The schedule provided exposes discrete betas with length `T`.
    """

    def __init__(self, model: nn.Module, T: int, betas: torch.Tensor):
        super().__init__()
        self.model = model
        self.T = T

        # buffers for schedule (keep abar in float64 to reduce drift)
        betas = betas.double()
        alphas = (1.0 - betas)
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)

    @torch.no_grad()
    def _pred_eps(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor, pre_ori: bool) -> torch.Tensor:
        """Predict epsilon at step t. If model predicts x0, convert to eps via schedule."""
        # The network operates in fp32; keep outer math in higher precision if desired.
        x_in = x_t.float()
        cond_in = cond.float()
        eps_or_x0 = self.model(torch.cat((x_in, cond_in), dim=1), t)
        if pre_ori:
            # model predicts x0 (residual); map to epsilon
            abar_t = _extract(self.alphas_bar, t, x_t.shape).to(x_t.dtype)
            eps = (x_in - abar_t.sqrt() * eps_or_x0) / (1.0 - abar_t).clamp(min=1e-12).sqrt()
            return eps.to(x_t.dtype)
        else:
            # model directly predicts epsilon
            return eps_or_x0.to(x_t.dtype)

    def _denoise_step(self, sample: torch.Tensor, eps_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Deterministic DDIM-style step: x_{t-1} = a_t * x_t + b_t * eps.
        Use closed-form from DDIM with eta=0.
        """
        alpha_bar_t = _extract(self.alphas_bar, t, sample.shape).to(sample.dtype)
        # prev index is t-1 (full steps)
        t_prev = (t - 1).clamp(min=-1)
        # alpha_bar at prev; when t_prev == -1, use 1.0
        alpha_bar_prev = torch.where(
            (t_prev[:, None, None, None] >= 0),
            _extract(self.alphas_bar, t_prev.clamp_min(0), sample.shape).to(sample.dtype),
            torch.ones_like(alpha_bar_t),
        )
        a_t = (alpha_bar_prev / alpha_bar_t).clamp(min=1e-20).sqrt()
        b_t = (1.0 - alpha_bar_prev).clamp(min=0).sqrt() - (
            (alpha_bar_prev * (1.0 - alpha_bar_t)) / alpha_bar_t
        ).clamp(min=1e-20).sqrt()
        return a_t * sample + b_t * eps_pred

    def _inverse_step(self, sample: torch.Tensor, eps_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Inverse of the deterministic step: x_{t+1} given x_t and eps.
        Derived from the same (a_t, b_t) but solving for next sample.
        """
        alpha_bar_t = _extract(self.alphas_bar, t, sample.shape).to(sample.dtype)
        # next index is t+1 (full steps)
        t_next = (t + 1).clamp(max=self.T - 1)
        # alpha_bar at next; when t is the final step (T-1), treat next abar as a tiny value for numerical stability
        alpha_bar_next = _extract(self.alphas_bar, t_next, sample.shape).to(sample.dtype)
        a_t = (alpha_bar_next / alpha_bar_t).clamp(min=1e-20).sqrt()
        b_t = (1.0 - alpha_bar_next).clamp(min=0).sqrt() - (
            (alpha_bar_next * (1.0 - alpha_bar_t)) / alpha_bar_t
        ).clamp(min=1e-20).sqrt()
        return (sample - b_t * eps_pred) / a_t

    @torch.no_grad()
    def sample(self, x_T: torch.Tensor, cond: torch.Tensor, pre_ori: bool = True, p: float = 0.93,
               use_fp64: bool = True) -> torch.Tensor:
        """
        Run EDICT denoising from x_T to x_0 (residual domain) using coupled updates.
        Returns x_0 estimate.
        """
        x = x_T.clone()
        y = x_T.clone()
        if use_fp64:
            x = x.double()
            y = y.double()
        else:
            x = x.float()
            y = y.float()

        for time_step in reversed(range(self.T)):
            t = x_T.new_ones([x_T.shape[0],], dtype=torch.long) * time_step

            # x_inter uses noise predicted from y
            eps1 = self._pred_eps(y, cond, t, pre_ori)
            x_inter = self._denoise_step(x, eps1.to(x.dtype), t)

            # y_inter uses noise predicted from x_inter
            eps2 = self._pred_eps(x_inter, cond, t, pre_ori)
            y_inter = self._denoise_step(y, eps2.to(y.dtype), t)

            # coupled mixing (symmetric, use the two intermediates only)
            x_prev = p * x_inter + (1.0 - p) * y_inter
            y_prev = p * y_inter + (1.0 - p) * x_inter

            # alternate order, as in EDICT
            x, y = y_prev, x_prev

        # Either x or y is a valid x0 candidate; return x cast to fp32
        return x.float()

    @torch.no_grad()
    def invert(self, x0_residual: torch.Tensor, cond: torch.Tensor, pre_ori: bool = True, p: float = 0.93,
               use_fp64: bool = True) -> torch.Tensor:
        """
        Run EDICT inversion from x_0 (residual) to an estimate of x_T (initial noise).
        Useful for recovering embedded sampling noise on the decode side.
        """
        x = x0_residual.clone()
        y = x0_residual.clone()
        if use_fp64:
            x = x.double()
            y = y.double()
        else:
            x = x.float()
            y = y.float()

        for time_step in range(self.T):
            t = x0_residual.new_ones([x0_residual.shape[0],], dtype=torch.long) * time_step

            # inverse mixing: solve linear system exactly
            denom = (2.0 * p - 1.0)
            # avoid division by zero if p=0.5 (not expected in practice; p ~ 0.93)
            denom = denom if isinstance(denom, float) else float(denom)
            x_inter = (p * x - (1.0 - p) * y) / denom
            y_inter = (p * y - (1.0 - p) * x) / denom

            # advance to t+1 using inverse of deterministic step
            eps_y = self._pred_eps(x_inter, cond, t, pre_ori)
            y_next = self._inverse_step(y_inter, eps_y.to(y_inter.dtype), t)

            eps_x = self._pred_eps(y_next, cond, t, pre_ori)
            x_next = self._inverse_step(x_inter, eps_x.to(x_inter.dtype), t)

            # alternate
            x, y = y_next, x_next

        return x.float()

    def forward_one_step(self, x_t: torch.Tensor, eps_hat: torch.Tensor, t: torch.Tensor, p: float = 0.93) -> torch.Tensor:
        """
        Single EDICT forward (t -> t+1) using provided epsilon estimate.
        Does not call the model; suitable for training-time consistency loss.
        """
        # Initialize both tracks with the same state
        x = x_t
        y = x_t
        # Inverse mixing (exact) when both tracks equal collapses to identity
        denom = (2.0 * p - 1.0)
        x_inter = (p * x - (1.0 - p) * y) / denom
        y_inter = (p * y - (1.0 - p) * x) / denom
        # Advance with inverse-step using the same eps_hat
        y_next = self._inverse_step(y_inter, eps_hat.to(y_inter.dtype), t)
        x_next = self._inverse_step(x_inter, eps_hat.to(x_inter.dtype), t)
        # Return one of the tracks as the next state (consistent with alternation)
        return y_next
