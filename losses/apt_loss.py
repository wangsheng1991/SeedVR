# // Copyright (c) 2025 Custom Training Implementation
# //
# // Licensed under the Apache License, Version 2.0 (the "License")

"""
APT (Adversarial Post-Training) Loss Functions for SeedVR2

Based on:
- StyleGAN2 discriminator losses
- Feature matching loss
- R1 gradient penalty regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class APTLoss(nn.Module):
    """
    Complete APT loss module combining:
    - Non-saturating GAN loss
    - Feature matching loss
    - R1 gradient penalty
    - Optional perceptual loss

    Args:
        lambda_fm: Weight for feature matching loss
        lambda_r1: Weight for R1 regularization
        lambda_percep: Weight for perceptual loss (if used)
        r1_interval: Apply R1 every N steps (for efficiency)
    """
    def __init__(
        self,
        lambda_fm: float = 10.0,
        lambda_r1: float = 10.0,
        lambda_percep: float = 0.0,
        r1_interval: int = 16,
    ):
        super().__init__()
        self.lambda_fm = lambda_fm
        self.lambda_r1 = lambda_r1
        self.lambda_percep = lambda_percep
        self.r1_interval = r1_interval

    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """
        Non-saturating generator loss

        Args:
            fake_pred: Discriminator prediction on fake samples

        Returns:
            Generator adversarial loss
        """
        # Non-saturating loss: -log(D(G(z)))
        # Equivalent to: softplus(-D(G(z)))
        # Clamp for numerical stability (avoid overflow in exp)
        return F.softplus(-fake_pred.clamp(-50, 50)).mean()

    def discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Non-saturating discriminator loss

        Args:
            real_pred: Discriminator prediction on real samples
            fake_pred: Discriminator prediction on fake samples

        Returns:
            Discriminator adversarial loss
        """
        # Clamp for numerical stability
        real_pred = real_pred.clamp(-50, 50)
        fake_pred = fake_pred.clamp(-50, 50)

        # Real loss: -log(D(x)) = softplus(-D(x))
        real_loss = F.softplus(-real_pred).mean()
        # Fake loss: -log(1 - D(G(z))) = softplus(D(G(z)))
        fake_loss = F.softplus(fake_pred).mean()

        return real_loss + fake_loss

    def feature_matching_loss(
        self,
        real_features: List[torch.Tensor],
        fake_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Feature matching loss - matches intermediate discriminator features

        Uses layer-wise weighting: deeper layers (more semantic) get higher weight
        Following pix2pixHD: weight = 1 / num_layers for each layer

        Args:
            real_features: List of features from real samples
            fake_features: List of features from fake samples

        Returns:
            Feature matching loss
        """
        loss = 0.0
        num_features = len(real_features)

        for i, (real_feat, fake_feat) in enumerate(zip(real_features, fake_features)):
            # L1 loss on features, detach real to not backprop through D
            # Weight each layer equally (1/num_features), but could use
            # layer_weight = 1.0 / (2 ** (num_features - i - 1)) for deeper = higher
            layer_loss = F.l1_loss(fake_feat, real_feat.detach())
            loss += layer_loss / num_features

        return loss * self.lambda_fm

    def r1_regularization(
        self,
        real_pred: torch.Tensor,
        real_images: torch.Tensor
    ) -> torch.Tensor:
        """
        R1 gradient penalty regularization from StyleGAN2

        Penalizes the gradient of D with respect to real images
        Helps stabilize training

        Args:
            real_pred: Discriminator prediction on real samples
            real_images: Real input images (must have requires_grad=True)

        Returns:
            R1 regularization loss
        """
        # Compute gradients of D output w.r.t. real images
        grad_real = torch.autograd.grad(
            outputs=real_pred.sum(),
            inputs=real_images,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # R1 penalty: ||grad||^2
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty * self.lambda_r1

    def hinge_generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """
        Hinge loss variant for generator (alternative to non-saturating)

        Args:
            fake_pred: Discriminator prediction on fake samples

        Returns:
            Generator hinge loss
        """
        return -fake_pred.mean()

    def hinge_discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Hinge loss variant for discriminator

        Args:
            real_pred: Discriminator prediction on real samples
            fake_pred: Discriminator prediction on fake samples

        Returns:
            Discriminator hinge loss
        """
        real_loss = F.relu(1.0 - real_pred).mean()
        fake_loss = F.relu(1.0 + fake_pred).mean()
        return real_loss + fake_loss


class SeedVR2Loss(nn.Module):
    """
    Complete loss module for SeedVR2 training

    Combines:
    - APT adversarial losses
    - Reconstruction loss (optional, for warm-up)
    - LPIPS perceptual loss (optional)

    Args:
        lambda_adv: Weight for adversarial loss
        lambda_fm: Weight for feature matching loss
        lambda_r1: Weight for R1 regularization
        lambda_recon: Weight for reconstruction loss
        lambda_lpips: Weight for LPIPS loss
        loss_type: 'nonsaturating' or 'hinge'
    """
    def __init__(
        self,
        lambda_adv: float = 1.0,
        lambda_fm: float = 10.0,
        lambda_r1: float = 10.0,
        lambda_recon: float = 0.0,
        lambda_lpips: float = 0.0,
        loss_type: str = 'nonsaturating',
    ):
        super().__init__()

        self.lambda_adv = lambda_adv
        self.lambda_recon = lambda_recon
        self.lambda_lpips = lambda_lpips
        self.loss_type = loss_type

        self.apt_loss = APTLoss(
            lambda_fm=lambda_fm,
            lambda_r1=lambda_r1,
        )

        # Optional LPIPS
        self.lpips = None
        if lambda_lpips > 0:
            try:
                import lpips
                self.lpips = lpips.LPIPS(net='vgg').eval()
                for param in self.lpips.parameters():
                    param.requires_grad = False
            except ImportError:
                print("Warning: lpips not installed, skipping perceptual loss")
                self.lambda_lpips = 0.0

    def compute_generator_loss(
        self,
        fake_pred: torch.Tensor,
        real_features: Optional[List[torch.Tensor]] = None,
        fake_features: Optional[List[torch.Tensor]] = None,
        fake_images: Optional[torch.Tensor] = None,
        real_images: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total generator loss

        Args:
            fake_pred: Discriminator prediction on fake
            real_features: Features from real samples
            fake_features: Features from fake samples
            fake_images: Generated images (for reconstruction/perceptual)
            real_images: Target images (for reconstruction/perceptual)

        Returns:
            total_loss: Combined generator loss
            loss_dict: Dictionary of individual losses
        """
        loss_dict = {}

        # Adversarial loss
        if self.loss_type == 'nonsaturating':
            g_adv = self.apt_loss.generator_loss(fake_pred)
        else:
            g_adv = self.apt_loss.hinge_generator_loss(fake_pred)

        loss_dict['g_adv'] = g_adv.item()
        total_loss = g_adv * self.lambda_adv

        # Feature matching loss
        if real_features is not None and fake_features is not None:
            g_fm = self.apt_loss.feature_matching_loss(real_features, fake_features)
            loss_dict['g_fm'] = g_fm.item()
            total_loss = total_loss + g_fm

        # Reconstruction loss (L1)
        if self.lambda_recon > 0 and fake_images is not None and real_images is not None:
            g_recon = F.l1_loss(fake_images, real_images) * self.lambda_recon
            loss_dict['g_recon'] = g_recon.item()
            total_loss = total_loss + g_recon

        # LPIPS perceptual loss
        if self.lambda_lpips > 0 and self.lpips is not None and fake_images is not None and real_images is not None:
            # LPIPS expects (B, C, H, W), handle video by reshaping
            if fake_images.dim() == 5:
                b, c, t, h, w = fake_images.shape
                fake_2d = fake_images.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
                real_2d = real_images.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            else:
                fake_2d = fake_images
                real_2d = real_images

            with torch.no_grad():
                g_lpips = self.lpips(fake_2d, real_2d).mean() * self.lambda_lpips
            loss_dict['g_lpips'] = g_lpips.item()
            total_loss = total_loss + g_lpips

        loss_dict['g_total'] = total_loss.item()
        return total_loss, loss_dict

    def compute_discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
        real_images: Optional[torch.Tensor] = None,
        apply_r1: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total discriminator loss

        Args:
            real_pred: Discriminator prediction on real
            fake_pred: Discriminator prediction on fake
            real_images: Real images (for R1, must have requires_grad=True)
            apply_r1: Whether to apply R1 regularization this step

        Returns:
            total_loss: Combined discriminator loss
            loss_dict: Dictionary of individual losses
        """
        loss_dict = {}

        # Adversarial loss
        if self.loss_type == 'nonsaturating':
            d_adv = self.apt_loss.discriminator_loss(real_pred, fake_pred)
        else:
            d_adv = self.apt_loss.hinge_discriminator_loss(real_pred, fake_pred)

        loss_dict['d_adv'] = d_adv.item()
        total_loss = d_adv

        # R1 regularization
        if apply_r1 and real_images is not None:
            d_r1 = self.apt_loss.r1_regularization(real_pred, real_images)
            loss_dict['d_r1'] = d_r1.item()
            total_loss = total_loss + d_r1

        loss_dict['d_total'] = total_loss.item()
        return total_loss, loss_dict
