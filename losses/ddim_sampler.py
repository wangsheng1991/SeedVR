# // Copyright (c) 2025 Custom Training Implementation
# //
# // Licensed under the Apache License, Version 2.0 (the "License")

"""
DDIM Sampler for Teacher Model in APT Training

Implements:
- DDIM deterministic sampling
- DDPM stochastic sampling
- Configurable number of steps
- CFG (Classifier-Free Guidance) support
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Callable
import math


class DDIMSampler:
    """
    DDIM Sampler for multi-step diffusion sampling

    Used as the "teacher" in APT training to generate high-quality
    targets for the one-step "student" model.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear',
        prediction_type: str = 'epsilon',  # 'epsilon', 'v_prediction', 'sample'
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        # Create beta schedule
        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == 'scaled_linear':
            # Scaled linear (used in stable diffusion)
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps) ** 2
        elif beta_schedule == 'cosine':
            # Cosine schedule
            steps = num_train_timesteps + 1
            x = torch.linspace(0, num_train_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_train_timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # For SNR weighting
        self.snr = self.alphas_cumprod / (1 - self.alphas_cumprod)

    def _get_variance(self, timestep: int, prev_timestep: int) -> torch.Tensor:
        """Get variance for DDPM sampling"""
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

    def _predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        timestep: int,
        eps: torch.Tensor
    ) -> torch.Tensor:
        """Predict x_0 from noise prediction"""
        sqrt_recip_alpha = self.sqrt_recip_alphas_cumprod[timestep].to(x_t.device)
        sqrt_recipm1_alpha = self.sqrt_recipm1_alphas_cumprod[timestep].to(x_t.device)

        # Reshape for broadcasting
        while sqrt_recip_alpha.dim() < x_t.dim():
            sqrt_recip_alpha = sqrt_recip_alpha.unsqueeze(-1)
            sqrt_recipm1_alpha = sqrt_recipm1_alpha.unsqueeze(-1)

        x_0 = sqrt_recip_alpha * x_t - sqrt_recipm1_alpha * eps
        return x_0

    def _predict_x0_from_v(
        self,
        x_t: torch.Tensor,
        timestep: int,
        v: torch.Tensor
    ) -> torch.Tensor:
        """Predict x_0 from v-prediction"""
        sqrt_alpha = self.sqrt_alphas_cumprod[timestep].to(x_t.device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timestep].to(x_t.device)

        while sqrt_alpha.dim() < x_t.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        x_0 = sqrt_alpha * x_t - sqrt_one_minus_alpha * v
        return x_0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        prev_timestep: int,
        eta: float = 0.0,  # 0 = DDIM (deterministic), 1 = DDPM (stochastic)
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Single DDIM/DDPM step

        Args:
            model_output: Model prediction (noise, v, or sample)
            timestep: Current timestep
            sample: Current noisy sample x_t
            prev_timestep: Previous timestep (target)
            eta: Stochasticity (0=DDIM, 1=DDPM)
            generator: Random generator for reproducibility

        Returns:
            x_{t-1}: Denoised sample
        """
        # Get alpha values
        alpha_prod_t = self.alphas_cumprod[timestep].to(sample.device)
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep].to(sample.device) if prev_timestep >= 0 else torch.tensor(1.0, device=sample.device)

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # Predict x_0 based on prediction type
        if self.prediction_type == 'epsilon':
            pred_x0 = self._predict_x0_from_eps(sample, timestep, model_output)
        elif self.prediction_type == 'v_prediction':
            pred_x0 = self._predict_x0_from_v(sample, timestep, model_output)
        elif self.prediction_type == 'sample':
            pred_x0 = model_output
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # Clip predicted x_0
        if self.clip_sample:
            pred_x0 = pred_x0.clamp(-self.clip_sample_range, self.clip_sample_range)

        # Compute variance
        variance = self._get_variance(timestep, prev_timestep).to(sample.device)
        std_dev = eta * torch.sqrt(variance)

        # Direction pointing to x_t
        sqrt_alpha_prev = torch.sqrt(alpha_prod_t_prev)
        sqrt_one_minus_alpha_prev = torch.sqrt(1 - alpha_prod_t_prev - std_dev ** 2)

        # Reshape for broadcasting
        while sqrt_alpha_prev.dim() < sample.dim():
            sqrt_alpha_prev = sqrt_alpha_prev.unsqueeze(-1)
            sqrt_one_minus_alpha_prev = sqrt_one_minus_alpha_prev.unsqueeze(-1)
            std_dev = std_dev.unsqueeze(-1) if std_dev.dim() < sample.dim() else std_dev

        # Compute predicted noise
        if self.prediction_type == 'epsilon':
            pred_eps = model_output
        else:
            # Derive epsilon from x_0 prediction
            sqrt_alpha = torch.sqrt(alpha_prod_t)
            sqrt_one_minus_alpha = torch.sqrt(beta_prod_t)
            while sqrt_alpha.dim() < sample.dim():
                sqrt_alpha = sqrt_alpha.unsqueeze(-1)
                sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
            pred_eps = (sample - sqrt_alpha * pred_x0) / sqrt_one_minus_alpha

        # DDIM formula
        pred_sample = sqrt_alpha_prev * pred_x0 + sqrt_one_minus_alpha_prev * pred_eps

        # Add noise for DDPM (eta > 0)
        if eta > 0:
            noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            pred_sample = pred_sample + std_dev * noise

        return pred_sample, pred_x0

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to samples (forward diffusion)"""
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].to(original_samples.device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].to(original_samples.device)

        # Reshape for broadcasting
        while sqrt_alpha.dim() < original_samples.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        noisy_samples = sqrt_alpha * original_samples + sqrt_one_minus_alpha * noise
        return noisy_samples

    def get_snr(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get signal-to-noise ratio for timesteps"""
        return self.snr[timesteps]

    @torch.no_grad()
    def sample(
        self,
        model: Callable,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        guidance_scale: float = 1.0,
        negative_condition: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
        return_intermediates: bool = False,
        **model_kwargs,
    ) -> torch.Tensor:
        """
        Full sampling loop

        Args:
            model: Denoising model (takes x_t, t, condition)
            shape: Output shape (B, ...)
            condition: Conditioning input
            num_inference_steps: Number of denoising steps
            eta: DDIM eta (0=deterministic, 1=stochastic)
            guidance_scale: CFG scale (1=no guidance)
            negative_condition: Negative condition for CFG
            generator: Random generator
            device: Device for sampling
            dtype: Data type
            return_intermediates: Return all intermediate samples
            **model_kwargs: Additional model arguments

        Returns:
            samples: Generated samples
        """
        device = device or (condition.device if condition is not None else torch.device('cuda'))
        dtype = dtype or (condition.dtype if condition is not None else torch.float32)

        # Create timestep schedule
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = torch.arange(0, num_inference_steps) * step_ratio
        timesteps = timesteps.flip(0).long()  # Reverse: T -> 0

        # Initialize with noise
        sample = torch.randn(shape, generator=generator, device=device, dtype=dtype)

        intermediates = [sample] if return_intermediates else None

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((shape[0],), t.item(), device=device, dtype=torch.long)

            # CFG: run model twice if guidance_scale > 1
            if guidance_scale > 1.0 and negative_condition is not None:
                # Unconditional prediction
                model_output_uncond = model(sample, t_tensor, negative_condition, **model_kwargs)

                # Conditional prediction
                model_output_cond = model(sample, t_tensor, condition, **model_kwargs)

                # CFG combination
                model_output = model_output_uncond + guidance_scale * (model_output_cond - model_output_uncond)
            else:
                model_output = model(sample, t_tensor, condition, **model_kwargs)

            # Get previous timestep
            prev_t = timesteps[i + 1].item() if i + 1 < len(timesteps) else -1

            # DDIM step
            sample, pred_x0 = self.step(
                model_output=model_output,
                timestep=t.item(),
                sample=sample,
                prev_timestep=prev_t,
                eta=eta,
                generator=generator,
            )

            if return_intermediates:
                intermediates.append(sample)

        if return_intermediates:
            return sample, intermediates

        return sample


# Import F for _get_variance
import torch.nn.functional as F
