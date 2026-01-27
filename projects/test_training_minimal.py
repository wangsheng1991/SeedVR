# // Copyright (c) 2025 Custom Training Implementation
# //
# // Licensed under the Apache License, Version 2.0 (the "License")

"""
Minimal end-to-end training test with synthetic data

Tests the complete training loop without requiring:
- Real video data
- GPU
- Pre-trained DiT checkpoint
- Text embeddings

This validates the training logic and component integration.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.discriminator import PatchDiscriminator3D
from losses.apt_loss import SeedVR2Loss


class SyntheticVideoDataset(Dataset):
    """Generate synthetic video data on-the-fly"""

    def __init__(self, num_samples=10, num_frames=8, size=64):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.size = size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random HQ and LQ videos
        hq = torch.randn(3, self.num_frames, self.size * 4, self.size * 4)
        lq = torch.randn(3, self.num_frames, self.size, self.size)

        return {
            'hq': hq,
            'lq': lq,
            'name': f'synthetic_{idx:04d}'
        }


class TinyGenerator(nn.Module):
    """Minimal generator for testing"""

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, out_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def test_training_loop():
    """Test complete training loop with synthetic data"""

    print("=" * 60)
    print("Minimal End-to-End Training Test")
    print("=" * 60)

    # Configuration
    batch_size = 2
    num_frames = 8
    lq_size = 64
    hq_size = 256
    num_iterations = 5

    device = torch.device('cpu')  # Use CPU for testing

    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Frames: {num_frames}")
    print(f"  LQ size: {lq_size}x{lq_size}")
    print(f"  HQ size: {hq_size}x{hq_size}")
    print(f"  Iterations: {num_iterations}")

    # 1. Create models
    print("\n[1/6] Creating models...")
    generator = TinyGenerator().to(device)
    discriminator = PatchDiscriminator3D(
        in_channels=3,
        base_channels=32,
        num_layers=2
    ).to(device)

    print(f"  Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"  Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    # 2. Create loss function
    print("\n[2/6] Creating loss function...")
    loss_fn = SeedVR2Loss(
        lambda_adv=1.0,
        lambda_fm=10.0,
        lambda_r1=10.0,
    )
    r1_interval = 2  # Apply R1 every 2 steps

    # 3. Create optimizers
    print("\n[3/6] Creating optimizers...")
    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.99))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.0, 0.99))

    # 4. Create dataset and dataloader
    print("\n[4/6] Creating dataset...")
    dataset = SyntheticVideoDataset(num_samples=10, num_frames=num_frames, size=lq_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batches per epoch: {len(dataloader)}")

    # 5. Test data loading
    print("\n[5/6] Testing data loading...")
    batch = next(iter(dataloader))
    print(f"  HQ shape: {batch['hq'].shape}")
    print(f"  LQ shape: {batch['lq'].shape}")

    # 6. Run training loop
    print("\n[6/6] Running training loop...")
    print("-" * 60)

    generator.train()
    discriminator.train()

    for iteration, batch in enumerate(dataloader):
        if iteration >= num_iterations:
            break

        lq_video = batch['lq'].to(device)
        hq_video = batch['hq'].to(device)

        # ============ Generator Step ============
        opt_g.zero_grad()

        # Generate fake video
        fake_video = generator(lq_video)

        # Upsample to match HQ size
        fake_video = torch.nn.functional.interpolate(
            fake_video,
            size=(num_frames, hq_size, hq_size),
            mode='trilinear',
            align_corners=False
        )

        # Get discriminator features
        with torch.no_grad():
            _, real_features = discriminator(hq_video, return_features=True)

        fake_pred, fake_features = discriminator(fake_video, return_features=True)

        # Compute generator loss
        g_loss, g_dict = loss_fn.compute_generator_loss(
            fake_pred=fake_pred,
            real_features=real_features,
            fake_features=fake_features,
        )

        g_loss.backward()
        opt_g.step()

        # ============ Discriminator Step ============
        opt_d.zero_grad()

        # Detach fake video and enable gradients for real video
        hq_video_grad = hq_video.detach().requires_grad_(True)
        fake_video_detached = fake_video.detach()

        real_pred, _ = discriminator(hq_video_grad, return_features=False)
        fake_pred, _ = discriminator(fake_video_detached, return_features=False)

        # Compute discriminator loss (apply R1 every r1_interval steps)
        apply_r1 = (iteration % r1_interval == 0)
        d_loss, d_dict = loss_fn.compute_discriminator_loss(
            real_pred=real_pred,
            fake_pred=fake_pred,
            real_images=hq_video_grad if apply_r1 else None,
            apply_r1=apply_r1,
        )

        d_loss.backward()
        opt_d.step()

        # Print progress
        r1_str = f", r1: {d_dict.get('d_r1', 0):.4f}" if apply_r1 else ""
        print(f"Iter {iteration + 1}/{num_iterations} | "
              f"G_loss: {g_loss.item():.4f} (adv: {g_dict['g_adv']:.4f}, fm: {g_dict['g_fm']:.4f}) | "
              f"D_loss: {d_loss.item():.4f} (adv: {d_dict['d_adv']:.4f}{r1_str})")

    print("-" * 60)
    print("\n[OK] Training loop completed successfully!")

    # 7. Test checkpoint saving
    print("\n[7/7] Testing checkpoint save/load...")
    checkpoint = {
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'opt_g': opt_g.state_dict(),
        'opt_d': opt_d.state_dict(),
        'iteration': num_iterations,
    }

    checkpoint_path = 'test_checkpoint.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f"  Saved checkpoint: {checkpoint_path}")

    # Load checkpoint
    loaded = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(loaded['generator'])
    discriminator.load_state_dict(loaded['discriminator'])
    print(f"  Loaded checkpoint successfully")

    # Cleanup
    os.remove(checkpoint_path)
    print(f"  Cleaned up test checkpoint")

    print("\n" + "=" * 60)
    print("All Tests Passed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run on GPU with real data")
    print("2. Test with pre-trained DiT checkpoint")
    print("3. Test multi-GPU training")
    print("4. Monitor training stability over longer runs")


if __name__ == "__main__":
    test_training_loop()
