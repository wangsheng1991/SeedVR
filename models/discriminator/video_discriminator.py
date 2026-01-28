# // Copyright (c) 2025 Custom Training Implementation
# //
# // Licensed under the Apache License, Version 2.0 (the "License")

"""
Video Discriminator for APT (Adversarial Post-Training)
Based on StyleGAN2 discriminator architecture, adapted for video
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class EqualLinear(nn.Module):
    """Equalized learning rate linear layer from StyleGAN2"""
    def __init__(self, in_features, out_features, bias=True, lr_mul=1.0):
        super().__init__()
        # Initialize with standard normal (not divided by lr_mul)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.scale = (1 / math.sqrt(in_features)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):
        # Apply equalized learning rate scaling at runtime
        bias = self.bias * self.lr_mul if self.bias is not None else None
        out = F.linear(x, self.weight * self.scale, bias)
        return out


class EqualConv3d(nn.Module):
    """Equalized learning rate 3D convolution for video"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        # Initialize with standard normal
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.scale = 1 / math.sqrt(in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2])
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        out = F.conv3d(x, self.weight * self.scale, self.bias, stride=self.stride, padding=self.padding)
        return out


class ResBlock3D(nn.Module):
    """3D Residual block for video discriminator"""
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.conv1 = EqualConv3d(in_channels, out_channels, 3, padding=1)
        self.conv2 = EqualConv3d(out_channels, out_channels, 3, padding=1)
        self.skip = EqualConv3d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        self.downsample = downsample
        # Learnable residual scaling (initialized to 1)
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        residual = self.skip(x)

        out = F.leaky_relu(self.conv1(x), 0.2)
        out = self.conv2(out)  # No activation before residual add

        # Scale residual branch for stable training
        out = residual + out * self.residual_scale
        out = F.leaky_relu(out, 0.2)

        if self.downsample:
            # Downsample spatially, keep temporal or downsample by 2
            out = F.avg_pool3d(out, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        return out


class VideoDiscriminator(nn.Module):
    """
    StyleGAN2-style discriminator adapted for video

    Args:
        in_channels: Input channels (3 for RGB video)
        base_channels: Base channel multiplier
        max_channels: Maximum channels
        num_layers: Number of residual blocks
        input_size: Expected input spatial size (H, W)
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        max_channels: int = 512,
        num_layers: int = 4,
        input_size: Tuple[int, int] = (256, 256),
    ):
        super().__init__()

        self.input_size = input_size
        channels = [min(base_channels * (2 ** i), max_channels) for i in range(num_layers + 1)]

        # Initial convolution
        self.from_rgb = EqualConv3d(in_channels, channels[0], kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # Residual blocks with downsampling
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                ResBlock3D(channels[i], channels[i + 1], downsample=True)
            )

        # Calculate final spatial size after downsampling
        final_h = input_size[0] // (2 ** num_layers)
        final_w = input_size[1] // (2 ** num_layers)

        # Final layers
        self.final_conv = EqualConv3d(channels[-1], channels[-1], kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # Global average pooling + linear
        self.fc = EqualLinear(channels[-1], 1)

        self.channels = channels

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            x: Input video tensor (B, C, T, H, W)
            return_features: Whether to return intermediate features for feature matching

        Returns:
            pred: Discriminator prediction (B, 1)
            features: List of intermediate features if return_features=True
        """
        features = []

        # From RGB
        out = F.leaky_relu(self.from_rgb(x), 0.2)
        if return_features:
            features.append(out)

        # Residual blocks
        for block in self.blocks:
            out = block(out)
            if return_features:
                features.append(out)

        # Final conv
        out = F.leaky_relu(self.final_conv(out), 0.2)
        if return_features:
            features.append(out)

        # Global average pooling over spatial and temporal dimensions
        out = out.mean(dim=[2, 3, 4])  # (B, C)

        # Final prediction
        pred = self.fc(out)  # (B, 1)

        if return_features:
            return pred, features
        return pred, None


class LatentDiscriminator(nn.Module):
    """
    Discriminator that operates on VAE latent space
    More efficient than pixel-space discriminator

    Args:
        in_channels: Latent channels (16 for SeedVR VAE)
        base_channels: Base channel multiplier
        num_layers: Number of residual blocks
        input_format: 'BTHWC' (SeedVR format) or 'BCTHW' (standard PyTorch)
    """
    def __init__(
        self,
        in_channels: int = 16,
        base_channels: int = 128,
        max_channels: int = 512,
        num_layers: int = 3,
        input_format: str = 'BTHWC',
    ):
        super().__init__()

        self.input_format = input_format
        channels = [min(base_channels * (2 ** i), max_channels) for i in range(num_layers + 1)]

        # Initial projection
        self.input_proj = EqualConv3d(in_channels, channels[0], kernel_size=1)

        # Residual blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                ResBlock3D(channels[i], channels[i + 1], downsample=True)
            )

        # Final layers
        self.final_conv = EqualConv3d(channels[-1], channels[-1], kernel_size=1)
        self.fc = EqualLinear(channels[-1], 1)

        self.channels = channels

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        input_format: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            x: Input latent tensor
            return_features: Whether to return intermediate features
            input_format: Override input format ('BTHWC' or 'BCTHW')

        Returns:
            pred: Discriminator prediction (B, 1)
            features: List of intermediate features if return_features=True
        """
        fmt = input_format or self.input_format

        # Convert from SeedVR format (B, T, H, W, C) to standard (B, C, T, H, W)
        if fmt == 'BTHWC':
            x = x.permute(0, 4, 1, 2, 3)

        features = []

        # Input projection
        out = F.leaky_relu(self.input_proj(x), 0.2)
        if return_features:
            features.append(out)

        # Residual blocks
        for block in self.blocks:
            out = block(out)
            if return_features:
                features.append(out)

        # Final conv
        out = F.leaky_relu(self.final_conv(out), 0.2)
        if return_features:
            features.append(out)

        # Global average pooling
        out = out.mean(dim=[2, 3, 4])

        # Final prediction
        pred = self.fc(out)

        if return_features:
            return pred, features
        return pred, None
