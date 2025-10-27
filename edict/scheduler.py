# EDICT Scheduler Module官方实现方法
import torch
import contextlib
from diffusers import DDIMScheduler


class CustomInversionScheduler:
    pass


class CustomDDIMInversionScheduler(DDIMScheduler, CustomInversionScheduler):
    def inverse_step(self, noise_pred: torch.Tensor, timestep: int, sample: torch.Tensor):
        # Perform DDIM inverse step in float64 for numerical stability
        amp_off = torch.autocast(device_type='cuda', enabled=False) if sample.is_cuda else contextlib.nullcontext()
        with amp_off:
            prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
            alpha_prod_t = self.alphas_cumprod[timestep].double()
            alpha_prod_t_prev = (
                self.alphas_cumprod[prev_timestep].double() if prev_timestep >= 0 else self.final_alpha_cumprod.double()
            )
            beta_prod_t = (1.0 - alpha_prod_t)

            if self.config.prediction_type == "epsilon":
                a_t = (alpha_prod_t_prev / alpha_prod_t).clamp(min=1e-20).sqrt()
                b_t = (1.0 - alpha_prod_t_prev).clamp(min=0).sqrt() - (
                    (alpha_prod_t_prev * beta_prod_t) / alpha_prod_t
                ).clamp(min=1e-20).sqrt()
            elif self.config.prediction_type == "sample":
                a_t = alpha_prod_t_prev.clamp(min=0).sqrt() + ((1.0 - alpha_prod_t_prev) / beta_prod_t).clamp(min=1e-20).sqrt()
                b_t = -(((1.0 - alpha_prod_t_prev) * alpha_prod_t) / beta_prod_t).clamp(min=1e-20).sqrt()
            elif self.config.prediction_type == "v_prediction":
                a_t = (alpha_prod_t_prev * alpha_prod_t).clamp(min=1e-20).sqrt() + (
                    (1.0 - alpha_prod_t_prev) * beta_prod_t
                ).clamp(min=0).sqrt()
                b_t = ((1.0 - alpha_prod_t_prev) * alpha_prod_t).clamp(min=0).sqrt() - (
                    alpha_prod_t_prev * beta_prod_t
                ).clamp(min=0).sqrt()
            else:
                raise ValueError(f"Unsupported prediction_type: {self.config.prediction_type}")

            next_sample = (sample.double() - b_t * noise_pred.double()) / a_t

        return next_sample.to(dtype=sample.dtype)

    def denoise_step(self, noise_pred: torch.Tensor, timestep: int, sample: torch.Tensor):
        # Perform deterministic DDIM step in float64 for numerical stability
        amp_off = torch.autocast(device_type='cuda', enabled=False) if sample.is_cuda else contextlib.nullcontext()
        with amp_off:
            prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
            alpha_prod_t = self.alphas_cumprod[timestep].double()
            alpha_prod_t_prev = (
                self.alphas_cumprod[prev_timestep].double() if prev_timestep >= 0 else self.final_alpha_cumprod.double()
            )
            beta_prod_t = (1.0 - alpha_prod_t)

            if self.config.prediction_type == "epsilon":
                a_t = (alpha_prod_t_prev / alpha_prod_t).clamp(min=1e-20).sqrt()
                b_t = (1.0 - alpha_prod_t_prev).clamp(min=0).sqrt() - (
                    (alpha_prod_t_prev * beta_prod_t) / alpha_prod_t
                ).clamp(min=1e-20).sqrt()
            elif self.config.prediction_type == "sample":
                a_t = alpha_prod_t_prev.clamp(min=0).sqrt() + ((1.0 - alpha_prod_t_prev) / beta_prod_t).clamp(min=1e-20).sqrt()
                b_t = -(((1.0 - alpha_prod_t_prev) * alpha_prod_t) / beta_prod_t).clamp(min=1e-20).sqrt()
            elif self.config.prediction_type == "v_prediction":
                a_t = (alpha_prod_t_prev * alpha_prod_t).clamp(min=1e-20).sqrt() + (
                    (1.0 - alpha_prod_t_prev) * beta_prod_t
                ).clamp(min=0).sqrt()
                b_t = ((1.0 - alpha_prod_t_prev) * alpha_prod_t).clamp(min=0).sqrt() - (
                    alpha_prod_t_prev * beta_prod_t
                ).clamp(min=0).sqrt()
            else:
                raise ValueError(f"Unsupported prediction_type: {self.config.prediction_type}")

            prev_sample = a_t * sample.double() + b_t * noise_pred.double()

        return prev_sample.to(dtype=sample.dtype)
