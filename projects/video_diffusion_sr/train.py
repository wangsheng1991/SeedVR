# // Copyright (c) 2025 Custom Training Implementation
# //
# // Licensed under the Apache License, Version 2.0 (the "License")

"""
VideoDiffusionTrainer - Core training class for SeedVR2 APT training

Implements:
- Adversarial Post-Training (APT) for one-step video super-resolution
- FSDP distributed training
- EMA model updates
- Gradient checkpointing
- Mixed precision training
"""

import os
import gc
import datetime
from typing import Dict, Optional, Tuple, List, Any
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from einops import rearrange
from omegaconf import DictConfig, OmegaConf

from common.config import create_object, load_config
from common.distributed import get_device, get_global_rank, get_world_size, init_torch
from common.distributed.advanced import (
    init_sequence_parallel,
    init_model_shard_group,
    get_data_parallel_rank,
    get_sequence_parallel_world_size,
)
from common.seed import set_seed
from common.logger import get_logger

from models.discriminator import PatchDiscriminator3D, VideoDiscriminator
from losses.apt_loss import SeedVR2Loss
from data.video_pair_dataset import create_dataloader


class EMAModel:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: torch.device = None):
        self.decay = decay
        self.device = device
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Clone to the same device as the parameter
                self.shadow[name] = param.data.clone().to(device if device else param.device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # Ensure shadow is on the same device
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].to(param.device)

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict, device=None):
        for k, v in state_dict.items():
            self.shadow[k] = v.to(device) if device else v


class VideoDiffusionTrainer:
    """
    Main trainer class for SeedVR2 APT training

    Args:
        config: OmegaConf configuration
    """
    def __init__(self, config: DictConfig):
        self.config = config
        self.logger = get_logger()
        self.device = get_device()
        self.global_step = 0
        self.epoch = 0

        # Will be initialized in configure_*
        self.dit = None
        self.vae = None
        self.discriminator = None
        self.ema = None

        self.optimizer_g = None
        self.optimizer_d = None
        self.scheduler_g = None
        self.scheduler_d = None

        self.loss_fn = None
        self.scaler_g = None
        self.scaler_d = None

        # Diffusion schedule parameters
        self.num_timesteps = config.get("diffusion", {}).get("num_timesteps", 1000)
        self.beta_start = config.get("diffusion", {}).get("beta_start", 0.0001)
        self.beta_end = config.get("diffusion", {}).get("beta_end", 0.02)
        self._setup_diffusion_schedule()

    def _setup_diffusion_schedule(self):
        """Setup diffusion noise schedule (linear beta schedule)"""
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Store as buffers (will be moved to device when needed)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        self.logger.info(f"Diffusion schedule: {self.num_timesteps} timesteps, "
                        f"beta [{self.beta_start}, {self.beta_end}]")

    def _add_noise(self, x: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Add noise to clean samples according to diffusion schedule

        Args:
            x: Clean samples (B, ...)
            noise: Gaussian noise (B, ...)
            timestep: Timestep indices (B,)

        Returns:
            Noisy samples at timestep t
        """
        # Move schedule to device if needed
        sqrt_alpha = self.sqrt_alphas_cumprod.to(x.device)[timestep]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod.to(x.device)[timestep]

        # Reshape for broadcasting
        while sqrt_alpha.dim() < x.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        return sqrt_alpha * x + sqrt_one_minus_alpha * noise

    def configure_dit_model(self, checkpoint: Optional[str] = None):
        """Initialize DiT (generator) model"""
        self.logger.info("Configuring DiT model...")

        # Create model
        self.dit = create_object(self.config.dit.model)

        # Load checkpoint if provided
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", mmap=True)
            self.dit.load_state_dict(state, strict=True)
            self.logger.info(f"Loaded DiT checkpoint from {checkpoint}")

        # Enable gradient checkpointing
        if self.config.dit.get("gradient_checkpoint", True):
            self.dit.set_gradient_checkpointing(True)

        self.dit.to(self.device)

        # Print model size
        num_params = sum(p.numel() for p in self.dit.parameters())
        trainable_params = sum(p.numel() for p in self.dit.parameters() if p.requires_grad)
        self.logger.info(f"DiT parameters: {num_params:,} (trainable: {trainable_params:,})")

    def configure_vae_model(self):
        """Initialize VAE model (frozen)"""
        self.logger.info("Configuring VAE model...")

        dtype = getattr(torch, self.config.vae.get("dtype", "bfloat16"))
        self.vae = create_object(self.config.vae.model)
        self.vae.requires_grad_(False).eval()
        self.vae.to(device=self.device, dtype=dtype)

        # Load checkpoint
        if self.config.vae.get("checkpoint"):
            state = torch.load(self.config.vae.checkpoint, map_location=self.device, mmap=True)
            self.vae.load_state_dict(state)
            self.logger.info(f"Loaded VAE checkpoint from {self.config.vae.checkpoint}")

        # Set causal slicing for memory efficiency
        if hasattr(self.vae, "set_causal_slicing") and hasattr(self.config.vae, "slicing"):
            self.vae.set_causal_slicing(**self.config.vae.slicing)

        if hasattr(self.vae, "set_memory_limit") and hasattr(self.config.vae, "memory_limit"):
            self.vae.set_memory_limit(**self.config.vae.memory_limit)

    def configure_discriminator(self):
        """Initialize discriminator for APT"""
        self.logger.info("Configuring Discriminator...")

        disc_config = self.config.get("discriminator", {})
        disc_type = disc_config.get("type", "patch")

        if disc_type == "patch":
            self.discriminator = PatchDiscriminator3D(
                in_channels=disc_config.get("in_channels", 3),
                base_channels=disc_config.get("base_channels", 64),
                num_layers=disc_config.get("num_layers", 3),
                use_spectral_norm=disc_config.get("use_spectral_norm", True),
            )
        else:
            self.discriminator = VideoDiscriminator(
                in_channels=disc_config.get("in_channels", 3),
                base_channels=disc_config.get("base_channels", 64),
                max_channels=disc_config.get("max_channels", 512),
                num_layers=disc_config.get("num_layers", 4),
            )

        self.discriminator.to(self.device)

        # Wrap with DDP for distributed training
        import torch.distributed as dist
        if dist.is_initialized() and get_world_size() > 1:
            self.discriminator = DDP(
                self.discriminator,
                device_ids=[self.device] if self.device.type == 'cuda' else None,
                find_unused_parameters=False,
            )
            self.logger.info("Discriminator wrapped with DDP")

        num_params = sum(p.numel() for p in self.discriminator.parameters())
        self.logger.info(f"Discriminator parameters: {num_params:,}")

    def configure_ema(self):
        """Initialize EMA for generator"""
        if self.config.get("ema", {}).get("decay", 0) > 0:
            decay = self.config.ema.decay
            self.ema = EMAModel(self.dit, decay=decay, device=self.device)
            self.logger.info(f"Initialized EMA with decay={decay}")

    def configure_optimizers(self):
        """Initialize optimizers and schedulers"""
        opt_config = self.config.get("optimizer", {})

        # Generator optimizer
        g_lr = opt_config.get("g_lr", 1e-5)
        g_betas = tuple(opt_config.get("g_betas", [0.0, 0.99]))
        g_weight_decay = opt_config.get("g_weight_decay", 0.0)

        self.optimizer_g = torch.optim.AdamW(
            self.dit.parameters(),
            lr=g_lr,
            betas=g_betas,
            weight_decay=g_weight_decay,
        )

        # Discriminator optimizer
        d_lr = opt_config.get("d_lr", 1e-4)
        d_betas = tuple(opt_config.get("d_betas", [0.0, 0.99]))
        d_weight_decay = opt_config.get("d_weight_decay", 0.0)

        self.optimizer_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=d_lr,
            betas=d_betas,
            weight_decay=d_weight_decay,
        )

        self.logger.info(f"Generator optimizer: AdamW lr={g_lr}")
        self.logger.info(f"Discriminator optimizer: AdamW lr={d_lr}")

        # Optional: learning rate schedulers
        scheduler_config = self.config.get("scheduler", {})
        if scheduler_config.get("type") == "cosine":
            total_steps = scheduler_config.get("total_steps", 100000)
            warmup_steps = scheduler_config.get("warmup_steps", 1000)

            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

            self.scheduler_g = torch.optim.lr_scheduler.LambdaLR(self.optimizer_g, lr_lambda)
            self.scheduler_d = torch.optim.lr_scheduler.LambdaLR(self.optimizer_d, lr_lambda)

    def configure_loss(self):
        """Initialize loss functions"""
        loss_config = self.config.get("loss", {})

        self.loss_fn = SeedVR2Loss(
            lambda_adv=loss_config.get("lambda_adv", 1.0),
            lambda_fm=loss_config.get("lambda_fm", 10.0),
            lambda_r1=loss_config.get("lambda_r1", 10.0),
            lambda_recon=loss_config.get("lambda_recon", 0.0),
            lambda_lpips=loss_config.get("lambda_lpips", 0.0),
            loss_type=loss_config.get("type", "nonsaturating"),
        )

        self.logger.info(f"Loss configured: lambda_adv={loss_config.get('lambda_adv', 1.0)}, "
                        f"lambda_fm={loss_config.get('lambda_fm', 10.0)}, "
                        f"lambda_r1={loss_config.get('lambda_r1', 10.0)}")

    def configure_amp(self):
        """Configure automatic mixed precision"""
        if self.config.get("training", {}).get("use_amp", True):
            # Use separate scalers for G and D to avoid interference
            self.scaler_g = GradScaler()
            self.scaler_d = GradScaler()
            self.logger.info("Enabled automatic mixed precision (AMP) with separate scalers")
        else:
            self.scaler_g = None
            self.scaler_d = None

        # Gradient clipping config
        self.gradient_clip = self.config.get("training", {}).get("gradient_clip", 0.0)
        if self.gradient_clip > 0:
            self.logger.info(f"Gradient clipping enabled: max_norm={self.gradient_clip}")

    def configure_all(
        self,
        dit_checkpoint: Optional[str] = None,
        sp_size: int = 1,
    ):
        """Configure all components"""
        # Initialize distributed
        init_torch(cudnn_benchmark=True, timeout=datetime.timedelta(seconds=3600))

        if sp_size > 1:
            init_sequence_parallel(sp_size)

        # Initialize FSDP if configured
        if self.config.dit.get("fsdp"):
            sharding_strategy = getattr(
                ShardingStrategy,
                self.config.dit.fsdp.get("sharding_strategy", "FULL_SHARD")
            )
            init_model_shard_group(sharding_strategy=sharding_strategy)

        # Configure components
        self.configure_dit_model(checkpoint=dit_checkpoint)
        self.configure_vae_model()
        self.configure_discriminator()
        self.configure_ema()
        self.configure_optimizers()
        self.configure_loss()
        self.configure_amp()

    @torch.no_grad()
    def vae_encode(self, videos: torch.Tensor) -> torch.Tensor:
        """Encode videos to latent space"""
        dtype = getattr(torch, self.config.vae.get("dtype", "bfloat16"))
        scale = self.config.vae.get("scaling_factor", 0.9152)

        videos = videos.to(self.device, dtype)

        # VAE encode
        if hasattr(self.vae, "preprocess"):
            videos = self.vae.preprocess(videos)

        latent = self.vae.encode(videos).latent
        latent = latent.unsqueeze(2) if latent.ndim == 4 else latent
        latent = rearrange(latent, "b c ... -> b ... c")
        latent = latent * scale

        return latent

    @torch.no_grad()
    def vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to pixel space"""
        dtype = getattr(torch, self.config.vae.get("dtype", "bfloat16"))
        scale = self.config.vae.get("scaling_factor", 0.9152)

        latents = latents.to(self.device, dtype)
        latents = latents / scale
        latents = rearrange(latents, "b ... c -> b c ...")
        latents = latents.squeeze(2) if latents.shape[2] == 1 else latents

        samples = self.vae.decode(latents).sample

        if hasattr(self.vae, "postprocess"):
            samples = self.vae.postprocess(samples)

        return samples

    def get_condition(self, latent_lq: torch.Tensor) -> torch.Tensor:
        """Build condition tensor from LQ latent"""
        # SeedVR format: concatenate LQ latent with mask channel
        t, h, w, c = latent_lq.shape[-4:]
        cond = torch.zeros([*latent_lq.shape[:-1], c + 1], device=latent_lq.device, dtype=latent_lq.dtype)
        cond[..., :-1] = latent_lq
        cond[..., -1:] = 1.0  # Mask indicating condition
        return cond

    def generator_forward(
        self,
        lq_latent: torch.Tensor,
        hq_latent: torch.Tensor,
        text_embeds: Dict[str, torch.Tensor],
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generator forward pass for APT training

        For APT (Adversarial Post-Training), we train the model to denoise
        from a high timestep (near pure noise) to clean samples in one step.

        Args:
            lq_latent: Low-quality latent (B, T, H, W, C) - used as condition
            hq_latent: High-quality latent (B, T, H, W, C) - used to construct noisy input
            text_embeds: Text embeddings dict
            timestep: Optional timestep, if None uses max timestep for one-step generation

        Returns:
            Generated HQ latent (denoised prediction)
        """
        batch_size = lq_latent.shape[0]

        # For APT one-step generation, use high timestep (near T_max)
        # This means we start from heavily noised samples
        if timestep is None:
            # Use timestep near the end of diffusion (e.g., 999 for 1000 steps)
            # For true one-step, use T_max - 1
            t_value = self.num_timesteps - 1
            timestep = torch.full((batch_size,), t_value, device=self.device, dtype=torch.long)

        # Generate noise
        noise = torch.randn_like(hq_latent)

        # Create noisy HQ latent: x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * noise
        noisy_hq = self._add_noise(hq_latent, noise, timestep)

        # Build condition from LQ
        condition = self.get_condition(lq_latent)

        # Concatenate noisy HQ with condition
        # Input format: [noisy_sample, condition]
        dit_input = torch.cat([noisy_hq, condition], dim=-1)

        # Convert timestep to float for model input
        timestep_float = timestep.float()

        # DiT forward - predicts clean sample (or noise, depending on model)
        with autocast("cuda", torch.bfloat16, enabled=self.scaler_g is not None):
            output = self.dit(
                vid=dit_input,
                txt=text_embeds.get("texts_pos", []),
                vid_shape=torch.tensor(lq_latent.shape[1:4], device=self.device).unsqueeze(0).repeat(batch_size, 1),
                txt_shape=text_embeds.get("txt_shape", []),
                timestep=timestep_float,
            )

        return output.vid_sample

    def train_step_discriminator(
        self,
        hq_video: torch.Tensor,
        fake_video: torch.Tensor,
        apply_r1: bool = True,
    ) -> Dict[str, float]:
        """
        Train discriminator for one step

        Args:
            hq_video: Real HQ video (B, C, T, H, W)
            fake_video: Generated video (B, C, T, H, W)
            apply_r1: Whether to apply R1 regularization

        Returns:
            Loss dictionary
        """
        self.optimizer_d.zero_grad()

        # Prepare real input (need gradients for R1)
        if apply_r1:
            hq_video = hq_video.detach().requires_grad_(True)

        # Forward pass with AMP
        with autocast("cuda", torch.bfloat16, enabled=self.scaler_d is not None):
            real_pred, real_features = self.discriminator(hq_video, return_features=True)
            fake_pred, _ = self.discriminator(fake_video.detach(), return_features=False)

            # Compute discriminator adversarial loss
            if self.loss_fn.loss_type == 'nonsaturating':
                d_adv = self.loss_fn.apt_loss.discriminator_loss(real_pred, fake_pred)
            else:
                d_adv = self.loss_fn.apt_loss.hinge_discriminator_loss(real_pred, fake_pred)

        d_loss_dict = {'d_adv': d_adv.item()}
        total_loss = d_adv

        # R1 regularization - compute outside autocast for numerical stability
        # Apply lazy regularization scaling: multiply by r1_interval to compensate
        # for only applying R1 every r1_interval steps
        if apply_r1 and hq_video is not None:
            r1_interval = self.config.get("loss", {}).get("r1_interval", 16)
            # Need to exit autocast for gradient computation
            real_pred_float = real_pred.float()
            d_r1 = self.loss_fn.apt_loss.r1_regularization(real_pred_float, hq_video)
            # Scale by r1_interval for lazy regularization (StyleGAN2 technique)
            d_r1_scaled = d_r1 * r1_interval
            d_loss_dict['d_r1'] = d_r1.item()  # Log unscaled for interpretability
            total_loss = total_loss + d_r1_scaled

        d_loss_dict['d_total'] = total_loss.item()

        # Backward with gradient clipping
        if self.scaler_d is not None:
            self.scaler_d.scale(total_loss).backward()
            if self.gradient_clip > 0:
                self.scaler_d.unscale_(self.optimizer_d)
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.gradient_clip)
            self.scaler_d.step(self.optimizer_d)
            self.scaler_d.update()
        else:
            total_loss.backward()
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.gradient_clip)
            self.optimizer_d.step()

        return d_loss_dict

    def train_step_generator(
        self,
        lq_latent: torch.Tensor,
        hq_latent: torch.Tensor,
        hq_video: torch.Tensor,
        text_embeds: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Train generator for one step

        Args:
            lq_latent: LQ latent from VAE
            hq_latent: HQ latent from VAE (for constructing noisy input)
            hq_video: Real HQ video for feature matching
            text_embeds: Text embeddings

        Returns:
            fake_video: Generated video
            loss_dict: Loss dictionary
        """
        self.optimizer_g.zero_grad()

        # Generate fake - now requires hq_latent for proper noise construction
        with autocast("cuda", torch.bfloat16, enabled=self.scaler_g is not None):
            fake_latent = self.generator_forward(lq_latent, hq_latent, text_embeds)

        # Decode to pixel space (with no_grad since VAE is frozen)
        with torch.no_grad():
            fake_video = self.vae_decode(fake_latent)

        # Discriminator forward (for generator loss)
        with autocast("cuda", torch.bfloat16, enabled=self.scaler_g is not None):
            # Get real features (no grad needed, just for matching)
            with torch.no_grad():
                _, real_features = self.discriminator(hq_video, return_features=True)

            # Get fake prediction and features
            fake_pred, fake_features = self.discriminator(fake_video, return_features=True)

            # Compute generator loss
            g_loss, g_loss_dict = self.loss_fn.compute_generator_loss(
                fake_pred=fake_pred,
                real_features=real_features,
                fake_features=fake_features,
                fake_images=fake_video,
                real_images=hq_video,
            )

        # Backward with gradient clipping
        if self.scaler_g is not None:
            self.scaler_g.scale(g_loss).backward()
            if self.gradient_clip > 0:
                self.scaler_g.unscale_(self.optimizer_g)
                torch.nn.utils.clip_grad_norm_(self.dit.parameters(), self.gradient_clip)
            self.scaler_g.step(self.optimizer_g)
            self.scaler_g.update()
        else:
            g_loss.backward()
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.dit.parameters(), self.gradient_clip)
            self.optimizer_g.step()

        return fake_video, g_loss_dict

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        text_embeds: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Complete training step (D + G)

        Standard GAN training order: D first, then G
        This ensures D sees the latest fake samples before G updates

        Args:
            batch: Data batch with 'lq' and 'hq' keys
            text_embeds: Text embeddings

        Returns:
            Combined loss dictionary
        """
        lq_video = batch['lq'].to(self.device)  # (B, T, C, H, W)
        hq_video = batch['hq'].to(self.device)  # (B, T, C, H, W)

        # Convert to (B, C, T, H, W) format
        lq_video = rearrange(lq_video, 'b t c h w -> b c t h w')
        hq_video = rearrange(hq_video, 'b t c h w -> b c t h w')

        # VAE encode both LQ and HQ
        with torch.no_grad():
            lq_latent = self.vae_encode(lq_video)
            hq_latent = self.vae_encode(hq_video)

        # Apply R1 every N steps (with lazy regularization scaling)
        r1_interval = self.config.get("loss", {}).get("r1_interval", 16)
        apply_r1 = (self.global_step % r1_interval == 0)

        # ============ Step 1: Train Discriminator First ============
        # Generate fake video (no grad for D step)
        with torch.no_grad():
            with autocast("cuda", torch.bfloat16, enabled=self.scaler_g is not None):
                fake_latent = self.generator_forward(lq_latent, hq_latent, text_embeds)
            fake_video_for_d = self.vae_decode(fake_latent)

        d_loss_dict = self.train_step_discriminator(
            hq_video=hq_video,
            fake_video=fake_video_for_d,
            apply_r1=apply_r1,
        )

        # ============ Step 2: Train Generator ============
        fake_video, g_loss_dict = self.train_step_generator(
            lq_latent=lq_latent,
            hq_latent=hq_latent,
            hq_video=hq_video,
            text_embeds=text_embeds,
        )

        # Update EMA (only on rank 0 for consistency in distributed training)
        if self.ema is not None:
            if get_global_rank() == 0 or get_world_size() == 1:
                self.ema.update(self.dit)

        # Update schedulers
        if self.scheduler_g is not None:
            self.scheduler_g.step()
        if self.scheduler_d is not None:
            self.scheduler_d.step()

        self.global_step += 1

        # Combine loss dicts
        loss_dict = {**g_loss_dict, **d_loss_dict}
        loss_dict['step'] = self.global_step

        return loss_dict

    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        if get_global_rank() != 0:
            return

        # Handle DDP wrapped discriminator
        disc_state = self.discriminator.module.state_dict() if hasattr(self.discriminator, 'module') else self.discriminator.state_dict()

        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'dit_state_dict': self.dit.state_dict(),
            'discriminator_state_dict': disc_state,
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
        }

        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()

        if self.scheduler_g is not None:
            checkpoint['scheduler_g_state_dict'] = self.scheduler_g.state_dict()
            checkpoint['scheduler_d_state_dict'] = self.scheduler_d.state_dict()

        # Save GradScaler states for proper AMP resume
        if self.scaler_g is not None:
            checkpoint['scaler_g_state_dict'] = self.scaler_g.state_dict()
            checkpoint['scaler_d_state_dict'] = self.scaler_d.state_dict()

        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']

        self.dit.load_state_dict(checkpoint['dit_state_dict'])

        # Handle DDP wrapped discriminator
        if hasattr(self.discriminator, 'module'):
            self.discriminator.module.load_state_dict(checkpoint['discriminator_state_dict'])
        else:
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

        if self.ema is not None and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'], device=self.device)

        if self.scheduler_g is not None and 'scheduler_g_state_dict' in checkpoint:
            self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
            self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])

        # Load GradScaler states
        if self.scaler_g is not None and 'scaler_g_state_dict' in checkpoint:
            self.scaler_g.load_state_dict(checkpoint['scaler_g_state_dict'])
            self.scaler_d.load_state_dict(checkpoint['scaler_d_state_dict'])

        self.logger.info(f"Loaded checkpoint from {path} at step {self.global_step}")

    def train(
        self,
        dataloader,
        text_embeds: Dict[str, torch.Tensor],
        num_epochs: int = 100,
        log_interval: int = 100,
        save_interval: int = 1000,
        save_dir: str = "./checkpoints",
    ):
        """
        Main training loop

        Args:
            dataloader: Training data loader
            text_embeds: Pre-computed text embeddings
            num_epochs: Number of epochs
            log_interval: Steps between logging
            save_interval: Steps between checkpoints
            save_dir: Directory for checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)

        self.dit.train()
        self.discriminator.train()

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch

            if hasattr(dataloader.sampler, 'set_epoch'):
                dataloader.sampler.set_epoch(epoch)

            pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=get_global_rank() != 0)

            for batch in pbar:
                loss_dict = self.train_step(batch, text_embeds)

                # Logging
                if self.global_step % log_interval == 0 and get_global_rank() == 0:
                    log_str = f"Step {self.global_step}: "
                    log_str += ", ".join([f"{k}={v:.4f}" for k, v in loss_dict.items() if k != 'step'])
                    self.logger.info(log_str)
                    pbar.set_postfix(loss_dict)

                # Save checkpoint
                if self.global_step % save_interval == 0:
                    ckpt_path = os.path.join(save_dir, f"checkpoint_{self.global_step}.pth")
                    self.save_checkpoint(ckpt_path)

                # Memory cleanup (configurable interval, default 1000)
                gc_interval = self.config.get("training", {}).get("gc_interval", 1000)
                if self.global_step % gc_interval == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Final checkpoint
        self.save_checkpoint(os.path.join(save_dir, "checkpoint_final.pth"))
