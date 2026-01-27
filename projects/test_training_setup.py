# // Copyright (c) 2025 Custom Training Implementation
# //
# // Licensed under the Apache License, Version 2.0 (the "License")

"""
Test script to verify training setup without actual training

Tests:
1. Model initialization
2. Data loading
3. Forward pass
4. Loss computation
5. Backward pass (dry run)
"""

import os
import sys
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.discriminator import PatchDiscriminator3D, VideoDiscriminator
from losses.apt_loss import SeedVR2Loss
from data.video_pair_dataset import VideoPairDataset


def test_discriminator():
    """Test discriminator forward pass"""
    print("\n=== Testing Discriminator ===")

    # Test PatchGAN
    disc = PatchDiscriminator3D(in_channels=3, base_channels=32, num_layers=2)

    # Create dummy video (B, C, T, H, W)
    video = torch.randn(2, 3, 8, 64, 64)

    pred, features = disc(video, return_features=True)

    print(f"Input shape: {video.shape}")
    print(f"Output shape: {pred.shape}")
    print(f"Number of feature layers: {len(features)}")
    print(f"Feature shapes: {[f.shape for f in features]}")

    # Test VideoDiscriminator
    disc2 = VideoDiscriminator(in_channels=3, base_channels=32, num_layers=2)
    pred2, features2 = disc2(video, return_features=True)

    print(f"\nVideoDiscriminator output: {pred2.shape}")
    print(f"Number of features: {len(features2)}")

    print("[OK] Discriminator test passed")


def test_loss():
    """Test loss computation"""
    print("\n=== Testing Loss Functions ===")

    loss_fn = SeedVR2Loss(
        lambda_adv=1.0,
        lambda_fm=10.0,
        lambda_r1=10.0,
    )

    # Dummy predictions (need requires_grad for R1 test)
    real_pred = torch.randn(2, 1, requires_grad=True)
    fake_pred = torch.randn(2, 1)

    # Dummy features
    real_features = [torch.randn(2, 64, 8, 32, 32), torch.randn(2, 128, 8, 16, 16)]
    fake_features = [torch.randn(2, 64, 8, 32, 32), torch.randn(2, 128, 8, 16, 16)]

    # Test generator loss
    g_loss, g_dict = loss_fn.compute_generator_loss(
        fake_pred=fake_pred,
        real_features=real_features,
        fake_features=fake_features,
    )

    print(f"Generator loss: {g_loss.item():.4f}")
    print(f"Loss components: {g_dict}")

    # Test discriminator loss (without R1 - R1 requires proper computation graph)
    d_loss, d_dict = loss_fn.compute_discriminator_loss(
        real_pred=real_pred,
        fake_pred=fake_pred,
        real_images=None,
        apply_r1=False,
    )

    print(f"\nDiscriminator loss: {d_loss.item():.4f}")
    print(f"Loss components: {d_dict}")

    print("[OK] Loss test passed")


def test_dataset():
    """Test dataset loading"""
    print("\n=== Testing Dataset ===")

    # Create dummy data directory
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    hq_dir = os.path.join(temp_dir, 'hq')
    os.makedirs(hq_dir, exist_ok=True)

    # Create dummy video using opencv
    try:
        import cv2
        import numpy as np

        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(hq_dir, 'test_video.mp4')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (256, 256))

        for i in range(30):
            frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

        print(f"Created test video: {video_path}")

        # Test dataset
        dataset = VideoPairDataset(
            data_root=temp_dir,
            num_frames=8,
            crop_size=(128, 128),
            scale_factor=4,
            augment=False,
            generate_lq=True,
        )

        print(f"Dataset size: {len(dataset)}")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"HQ shape: {sample['hq'].shape}")
            print(f"LQ shape: {sample['lq'].shape}")
            print(f"Video name: {sample['name']}")
            print("[OK] Dataset test passed")
        else:
            print("[WARN] Dataset is empty (no videos found)")

    except ImportError:
        print("[WARN] OpenCV not installed, skipping dataset test")
    except Exception as e:
        print(f"[WARN] Dataset test failed: {e}")
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_training_step():
    """Test a complete training step"""
    print("\n=== Testing Training Step ===")

    # Create models
    generator = nn.Sequential(
        nn.Conv3d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv3d(64, 3, kernel_size=3, padding=1),
    )

    discriminator = PatchDiscriminator3D(in_channels=3, base_channels=32, num_layers=2)

    loss_fn = SeedVR2Loss(lambda_adv=1.0, lambda_fm=10.0, lambda_r1=10.0)

    # Optimizers
    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    # Dummy data
    lq_video = torch.randn(1, 3, 8, 64, 64)
    hq_video = torch.randn(1, 3, 8, 256, 256)

    # Generator step
    opt_g.zero_grad()
    fake_video = generator(lq_video)

    # Upsample to match HQ size
    fake_video = torch.nn.functional.interpolate(
        fake_video, size=(8, 256, 256), mode='trilinear', align_corners=False
    )

    with torch.no_grad():
        _, real_features = discriminator(hq_video, return_features=True)

    fake_pred, fake_features = discriminator(fake_video, return_features=True)

    g_loss, g_dict = loss_fn.compute_generator_loss(
        fake_pred=fake_pred,
        real_features=real_features,
        fake_features=fake_features,
    )

    g_loss.backward()
    opt_g.step()

    print(f"Generator step completed: loss={g_loss.item():.4f}")

    # Discriminator step
    opt_d.zero_grad()

    hq_video_grad = hq_video.detach().requires_grad_(True)
    real_pred, _ = discriminator(hq_video_grad, return_features=False)
    fake_pred, _ = discriminator(fake_video.detach(), return_features=False)

    d_loss, d_dict = loss_fn.compute_discriminator_loss(
        real_pred=real_pred,
        fake_pred=fake_pred,
        real_images=hq_video_grad,
        apply_r1=True,
    )

    d_loss.backward()
    opt_d.step()

    print(f"Discriminator step completed: loss={d_loss.item():.4f}")
    print("[OK] Training step test passed")


def test_config_loading():
    """Test configuration loading"""
    print("\n=== Testing Config Loading ===")

    config_path = "configs_3b/train.yaml"

    if os.path.exists(config_path):
        from common.config import load_config
        config = load_config(config_path)

        print(f"Config loaded successfully")
        print(f"Discriminator type: {config.get('discriminator', {}).get('type', 'N/A')}")
        print(f"Generator LR: {config.get('optimizer', {}).get('g_lr', 'N/A')}")
        print(f"Loss type: {config.get('loss', {}).get('type', 'N/A')}")
        print("[OK] Config test passed")
    else:
        print(f"[WARN] Config file not found: {config_path}")


def main():
    print("=" * 60)
    print("SeedVR2 Training Setup Test")
    print("=" * 60)

    try:
        test_discriminator()
    except Exception as e:
        print(f"[FAIL] Discriminator test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_loss()
    except Exception as e:
        print(f"[FAIL] Loss test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_dataset()
    except Exception as e:
        print(f"[FAIL] Dataset test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_training_step()
    except Exception as e:
        print(f"[FAIL] Training step test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_config_loading()
    except Exception as e:
        print(f"[FAIL] Config test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print("All basic tests completed. Check output above for any failures.")
    print("\nNext steps:")
    print("1. Prepare your training data in /path/to/data/hq/")
    print("2. Run: python projects/prepare_text_embeddings.py")
    print("3. Start training: python projects/train_seedvr2.py --data_root /path/to/data")


if __name__ == "__main__":
    main()
