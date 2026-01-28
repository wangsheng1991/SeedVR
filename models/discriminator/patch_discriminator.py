# // Copyright (c) 2025 Custom Training Implementation
# //
# // Licensed under the Apache License, Version 2.0 (the "License")

"""
PatchGAN-style 3D Discriminator
Simpler and more stable than full StyleGAN discriminator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class ConvBlock3D(nn.Module):
    """Basic 3D convolution block with spectral normalization"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(1, kernel_size, kernel_size),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False
        )

        if use_spectral_norm:
            conv = nn.utils.spectral_norm(conv)

        self.conv = conv
        self.norm = nn.InstanceNorm3d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class PatchDiscriminator3D(nn.Module):
    """
    PatchGAN discriminator for video
    Outputs a grid of predictions instead of single value
    More stable training than global discriminator

    Args:
        in_channels: Input channels (3 for RGB, 16 for latent)
        base_channels: Base channel count
        num_layers: Number of downsampling layers
        use_spectral_norm: Whether to use spectral normalization
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 3,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        # First layer without normalization
        first_conv = nn.Conv3d(
            in_channels, base_channels,
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1)
        )
        if use_spectral_norm:
            first_conv = nn.utils.spectral_norm(first_conv)

        layers = [first_conv, nn.LeakyReLU(0.2, inplace=True)]

        # Intermediate layers
        in_ch = base_channels
        for i in range(1, num_layers):
            out_ch = min(base_channels * (2 ** i), 512)
            layers.append(
                ConvBlock3D(in_ch, out_ch, use_spectral_norm=use_spectral_norm)
            )
            in_ch = out_ch

        # Final layer - stride 1 to get patch output
        final_conv = nn.Conv3d(
            in_ch, 1,
            kernel_size=(1, 4, 4),
            stride=(1, 1, 1),
            padding=(0, 1, 1)
        )
        if use_spectral_norm:
            final_conv = nn.utils.spectral_norm(final_conv)

        layers.append(final_conv)

        self.model = nn.Sequential(*layers)

        # Store intermediate layers for feature matching
        self._build_feature_layers(in_channels, base_channels, num_layers, use_spectral_norm)

    def _build_feature_layers(self, in_channels, base_channels, num_layers, use_spectral_norm):
        """Build layers for feature extraction (shares weights with self.model)"""
        # We'll extract features by running through model layers directly
        # Store layer indices for feature extraction
        self.feature_layer_indices = []

        # First layer is at index 0-1 (conv + relu)
        self.feature_layer_indices.append(2)  # After first conv block

        # Intermediate layers
        for i in range(1, num_layers):
            # Each ConvBlock3D adds one module
            self.feature_layer_indices.append(2 + i)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            x: Input tensor (B, C, T, H, W)
            return_features: Whether to return intermediate features

        Returns:
            pred: Patch predictions (B, 1, T, H', W')
            features: List of intermediate features if return_features=True
        """
        if not return_features:
            pred = self.model(x)
            return pred, None

        # Extract features while running forward pass (single pass)
        features = []
        out = x

        # Run through model layers and collect features
        for i, layer in enumerate(self.model):
            out = layer(out)
            # Collect features after specific layers (before final conv)
            if i in self.feature_layer_indices:
                features.append(out)

        # out is now the final prediction
        pred = out

        return pred, features

    def get_patch_loss(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """
        Calculate patch-based adversarial loss

        Args:
            pred: Discriminator predictions
            target_is_real: Whether target should be real (1) or fake (0)

        Returns:
            loss: Adversarial loss
        """
        if target_is_real:
            target = torch.ones_like(pred)
        else:
            target = torch.zeros_like(pred)

        return F.mse_loss(pred, target)


class MultiScaleDiscriminator3D(nn.Module):
    """
    Multi-scale discriminator for video
    Uses multiple discriminators at different scales

    Args:
        in_channels: Input channels
        base_channels: Base channel count
        num_discriminators: Number of discriminators at different scales
        num_layers: Number of layers per discriminator
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_discriminators: int = 2,
        num_layers: int = 3,
    ):
        super().__init__()

        self.discriminators = nn.ModuleList()
        for i in range(num_discriminators):
            self.discriminators.append(
                PatchDiscriminator3D(
                    in_channels=in_channels,
                    base_channels=base_channels,
                    num_layers=num_layers - i,  # Fewer layers for smaller scales
                )
            )

        self.downsample = nn.AvgPool3d(
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            padding=0
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[List[torch.Tensor], Optional[List[List[torch.Tensor]]]]:
        """
        Args:
            x: Input tensor (B, C, T, H, W)
            return_features: Whether to return intermediate features

        Returns:
            preds: List of predictions from each scale
            features: List of feature lists from each scale
        """
        preds = []
        all_features = [] if return_features else None

        for i, disc in enumerate(self.discriminators):
            pred, features = disc(x, return_features=return_features)
            preds.append(pred)
            if return_features:
                all_features.append(features)

            # Downsample for next scale
            if i < len(self.discriminators) - 1:
                x = self.downsample(x)

        return preds, all_features
