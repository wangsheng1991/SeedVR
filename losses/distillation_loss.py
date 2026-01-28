# // Copyright (c) 2025 Custom Training Implementation
# //
# // Licensed under the Apache License, Version 2.0 (the "License")

"""
Distillation Loss for APT (Adversarial Post-Training)

Implements:
- Score Distillation Loss (matching teacher's multi-step output)
- Consistency Loss (self-consistency for one-step generation)
- LPIPS Perceptual Loss (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class DistillationLoss(nn.Module):
    """
    Distillation loss for training one-step generator from multi-step teacher

    The key insight of APT/LCM/SDXL-Turbo:
    - Teacher: Multi-step diffusion sampling (slow but high quality)
    - Student: One-step generation (fast)
    - Goal: Student output should match teacher output

    Loss types:
    - 'l2': MSE loss (standard)
    - 'l1': L1 loss (more robust to outliers)
    - 'huber': Huber loss (balanced)
    - 'lpips': Perceptual loss (better visual quality)
    """

    def __init__(
        self,
        loss_type: str = 'l2',
        lambda_distill: float = 1.0,
        lambda_lpips: float = 0.0,
        use_snr_weighting: bool = True,
    ):
        super().__init__()

        self.loss_type = loss_type
        self.lambda_distill = lambda_distill
        self.lambda_lpips = lambda_lpips
        self.use_snr_weighting = use_snr_weighting

        # Optional LPIPS
        self.lpips = None
        if lambda_lpips > 0:
            try:
                import lpips
                self.lpips = lpips.LPIPS(net='vgg').eval()
                for param in self.lpips.parameters():
                    param.requires_grad = False
            except ImportError:
                print("Warning: lpips not installed, disabling perceptual loss")
                self.lambda_lpips = 0.0

    def forward(
        self,
        student_pred: torch.Tensor,
        teacher_target: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        snr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss

        Args:
            student_pred: Student model prediction (B, ...)
            teacher_target: Teacher model target (B, ...)
            timestep: Timestep for SNR weighting (B,)
            snr: Signal-to-noise ratio for weighting (B,)

        Returns:
            loss: Total distillation loss
            loss_dict: Dictionary of individual losses
        """
        loss_dict = {}

        # Base distillation loss
        if self.loss_type == 'l2':
            base_loss = F.mse_loss(student_pred, teacher_target, reduction='none')
        elif self.loss_type == 'l1':
            base_loss = F.l1_loss(student_pred, teacher_target, reduction='none')
        elif self.loss_type == 'huber':
            base_loss = F.huber_loss(student_pred, teacher_target, reduction='none', delta=0.1)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Reduce spatial dimensions, keep batch
        base_loss = base_loss.mean(dim=list(range(1, base_loss.dim())))

        # SNR weighting (from progressive distillation / LCM)
        if self.use_snr_weighting and snr is not None:
            # Weight by SNR: higher SNR (cleaner) = higher weight
            # This focuses learning on cleaner samples first
            weight = snr / (snr + 1)
            base_loss = base_loss * weight

        distill_loss = base_loss.mean() * self.lambda_distill
        loss_dict['distill'] = distill_loss.item()

        total_loss = distill_loss

        # LPIPS perceptual loss (for pixel-space predictions)
        if self.lambda_lpips > 0 and self.lpips is not None:
            # Handle video format (B, C, T, H, W) -> (B*T, C, H, W)
            if student_pred.dim() == 5:
                b, c, t, h, w = student_pred.shape
                student_2d = student_pred.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
                teacher_2d = teacher_target.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            else:
                student_2d = student_pred
                teacher_2d = teacher_target

            with torch.no_grad():
                lpips_loss = self.lpips(student_2d, teacher_2d).mean() * self.lambda_lpips

            loss_dict['lpips'] = lpips_loss.item()
            total_loss = total_loss + lpips_loss

        loss_dict['distill_total'] = total_loss.item()
        return total_loss, loss_dict


class ConsistencyLoss(nn.Module):
    """
    Consistency loss for self-consistency training (LCM-style)

    Instead of using a separate teacher, enforce that:
    f(x_t, t) ≈ f(x_{t'}, t') when both should map to same x_0

    This allows training without a frozen teacher model.
    """

    def __init__(
        self,
        lambda_consistency: float = 1.0,
        skip_steps: int = 1,
    ):
        super().__init__()
        self.lambda_consistency = lambda_consistency
        self.skip_steps = skip_steps

    def forward(
        self,
        pred_from_t: torch.Tensor,
        pred_from_t_skip: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute consistency loss

        Args:
            pred_from_t: Prediction from timestep t
            pred_from_t_skip: Prediction from timestep t - skip_steps

        Returns:
            loss: Consistency loss
            loss_dict: Loss dictionary
        """
        loss = F.mse_loss(pred_from_t, pred_from_t_skip.detach())
        loss = loss * self.lambda_consistency

        return loss, {'consistency': loss.item()}


class APTLossComplete(nn.Module):
    """
    Complete APT loss combining:
    - Distillation loss (teacher matching)
    - Adversarial loss (GAN)
    - Feature matching loss
    - R1 regularization

    This is the full loss function for APT training.
    """

    def __init__(
        self,
        # Distillation
        lambda_distill: float = 1.0,
        distill_loss_type: str = 'l2',
        use_snr_weighting: bool = True,
        # Adversarial
        lambda_adv: float = 0.1,
        lambda_fm: float = 1.0,
        lambda_r1: float = 10.0,
        adv_loss_type: str = 'nonsaturating',
        # Perceptual
        lambda_lpips: float = 0.0,
    ):
        super().__init__()

        self.lambda_distill = lambda_distill
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        self.lambda_r1 = lambda_r1

        # Distillation loss
        self.distill_loss = DistillationLoss(
            loss_type=distill_loss_type,
            lambda_distill=1.0,  # We apply lambda outside
            lambda_lpips=lambda_lpips,
            use_snr_weighting=use_snr_weighting,
        )

        # Adversarial loss type
        self.adv_loss_type = adv_loss_type

    def generator_adversarial_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """Non-saturating generator loss"""
        if self.adv_loss_type == 'nonsaturating':
            return F.softplus(-fake_pred.clamp(-50, 50)).mean()
        else:  # hinge
            return -fake_pred.mean()

    def discriminator_adversarial_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor
    ) -> torch.Tensor:
        """Non-saturating discriminator loss"""
        real_pred = real_pred.clamp(-50, 50)
        fake_pred = fake_pred.clamp(-50, 50)

        if self.adv_loss_type == 'nonsaturating':
            real_loss = F.softplus(-real_pred).mean()
            fake_loss = F.softplus(fake_pred).mean()
        else:  # hinge
            real_loss = F.relu(1.0 - real_pred).mean()
            fake_loss = F.relu(1.0 + fake_pred).mean()

        return real_loss + fake_loss

    def feature_matching_loss(
        self,
        real_features: list,
        fake_features: list
    ) -> torch.Tensor:
        """Feature matching loss"""
        loss = 0.0
        num_features = len(real_features)

        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += F.l1_loss(fake_feat, real_feat.detach()) / num_features

        return loss * self.lambda_fm

    def r1_regularization(
        self,
        real_pred: torch.Tensor,
        real_images: torch.Tensor
    ) -> torch.Tensor:
        """R1 gradient penalty"""
        grad_real = torch.autograd.grad(
            outputs=real_pred.sum(),
            inputs=real_images,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty * self.lambda_r1

    def compute_generator_loss(
        self,
        student_pred: torch.Tensor,
        teacher_target: torch.Tensor,
        fake_pred: torch.Tensor,
        real_features: Optional[list] = None,
        fake_features: Optional[list] = None,
        timestep: Optional[torch.Tensor] = None,
        snr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute complete generator loss for APT

        Args:
            student_pred: Student (one-step) prediction
            teacher_target: Teacher (multi-step) target
            fake_pred: Discriminator prediction on fake
            real_features: Discriminator features on real
            fake_features: Discriminator features on fake
            timestep: Timestep for SNR weighting
            snr: Signal-to-noise ratio

        Returns:
            total_loss: Combined loss
            loss_dict: Individual losses
        """
        loss_dict = {}
        total_loss = 0.0

        # 1. Distillation loss (primary)
        distill_loss, distill_dict = self.distill_loss(
            student_pred, teacher_target, timestep, snr
        )
        distill_loss = distill_loss * self.lambda_distill
        loss_dict['g_distill'] = distill_loss.item()
        total_loss = total_loss + distill_loss

        # 2. Adversarial loss (auxiliary)
        if self.lambda_adv > 0:
            adv_loss = self.generator_adversarial_loss(fake_pred) * self.lambda_adv
            loss_dict['g_adv'] = adv_loss.item()
            total_loss = total_loss + adv_loss

        # 3. Feature matching loss (auxiliary)
        if self.lambda_fm > 0 and real_features is not None and fake_features is not None:
            fm_loss = self.feature_matching_loss(real_features, fake_features)
            loss_dict['g_fm'] = fm_loss.item()
            total_loss = total_loss + fm_loss

        loss_dict['g_total'] = total_loss.item()
        return total_loss, loss_dict

    def compute_discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
        real_images: Optional[torch.Tensor] = None,
        apply_r1: bool = True,
        r1_interval: int = 16,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute discriminator loss

        Args:
            real_pred: Discriminator prediction on real
            fake_pred: Discriminator prediction on fake
            real_images: Real images for R1 (requires_grad=True)
            apply_r1: Whether to apply R1 this step
            r1_interval: R1 interval for lazy regularization scaling

        Returns:
            total_loss: Combined loss
            loss_dict: Individual losses
        """
        loss_dict = {}

        # Adversarial loss
        adv_loss = self.discriminator_adversarial_loss(real_pred, fake_pred)
        loss_dict['d_adv'] = adv_loss.item()
        total_loss = adv_loss

        # R1 regularization
        if apply_r1 and real_images is not None and self.lambda_r1 > 0:
            r1_loss = self.r1_regularization(real_pred.float(), real_images)
            # Scale by r1_interval for lazy regularization
            r1_loss = r1_loss * r1_interval
            loss_dict['d_r1'] = (r1_loss / r1_interval).item()  # Log unscaled
            total_loss = total_loss + r1_loss

        loss_dict['d_total'] = total_loss.item()
        return total_loss, loss_dict
