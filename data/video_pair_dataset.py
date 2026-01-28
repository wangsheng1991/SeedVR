# // Copyright (c) 2025 Custom Training Implementation
# //
# // Licensed under the Apache License, Version 2.0 (the "License")

"""
Video Pair Dataset for SeedVR2 Training

Loads paired low-quality (LQ) and high-quality (HQ) videos for super-resolution training
"""

import os
import random
from typing import Tuple, Optional, List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.io import read_video, read_image
import numpy as np


class VideoPairDataset(Dataset):
    """
    Dataset for loading paired LQ/HQ videos

    Directory structure:
        data_root/
            lq/
                video1.mp4
                video2.mp4
                ...
            hq/
                video1.mp4
                video2.mp4
                ...

    Args:
        data_root: Root directory containing lq/ and hq/ folders
        num_frames: Number of frames to sample per clip
        frame_interval: Interval between sampled frames
        crop_size: Random crop size (H, W)
        scale_factor: Downscale factor for LQ (if generating LQ on-the-fly)
        augment: Whether to apply data augmentation
        max_samples: Maximum number of samples (for debugging)
    """
    def __init__(
        self,
        data_root: str,
        num_frames: int = 16,
        frame_interval: int = 1,
        crop_size: Tuple[int, int] = (256, 256),
        scale_factor: int = 4,
        augment: bool = True,
        max_samples: Optional[int] = None,
        generate_lq: bool = False,
    ):
        super().__init__()

        self.data_root = data_root
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.augment = augment
        self.generate_lq = generate_lq

        # Find video pairs
        self.lq_dir = os.path.join(data_root, 'lq')
        self.hq_dir = os.path.join(data_root, 'hq')

        if generate_lq:
            # Only need HQ videos, will generate LQ on-the-fly
            self.video_names = self._find_videos(self.hq_dir)
        else:
            # Need paired LQ/HQ videos
            lq_videos = set(self._find_videos(self.lq_dir))
            hq_videos = set(self._find_videos(self.hq_dir))
            self.video_names = list(lq_videos & hq_videos)

        if max_samples is not None:
            self.video_names = self.video_names[:max_samples]

        print(f"Found {len(self.video_names)} video pairs")

    def _find_videos(self, directory: str) -> List[str]:
        """Find all video files in directory"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        videos = []

        if not os.path.exists(directory):
            return videos

        for f in os.listdir(directory):
            if os.path.splitext(f.lower())[1] in video_extensions:
                videos.append(f)

        return sorted(videos)

    def _load_video(self, path: str) -> torch.Tensor:
        """Load video and return as tensor (T, C, H, W)"""
        video, _, info = read_video(path, output_format="TCHW")
        return video.float() / 255.0

    def _sample_frames(self, video: torch.Tensor, start_idx: int = None, indices: list = None) -> Tuple[torch.Tensor, int, list]:
        """
        Sample frames from video

        Args:
            video: Input video tensor (T, C, H, W)
            start_idx: Optional fixed start index (for syncing LQ/HQ)
            indices: Optional fixed indices (for syncing LQ/HQ)

        Returns:
            sampled_video: Sampled frames
            start_idx: The start index used
            indices: The indices used
        """
        total_frames = video.shape[0]
        required_frames = self.num_frames * self.frame_interval

        if total_frames < required_frames:
            # Pad by temporal reflection (more natural than repeating last frame)
            padding = required_frames - total_frames
            # Reflect padding: [..., n-2, n-1, n-1, n-2, ...]
            pad_frames = video.flip(dims=[0])[:padding]
            video = torch.cat([video, pad_frames], dim=0)
            if start_idx is None:
                start_idx = 0
        else:
            if start_idx is None:
                # Random start position
                max_start = total_frames - required_frames
                start_idx = random.randint(0, max_start)

        # Sample frames with interval
        if indices is None:
            indices = list(range(start_idx, start_idx + required_frames, self.frame_interval))

        return video[indices], start_idx, indices

    def _random_crop(
        self,
        hq: torch.Tensor,
        lq: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random crop to HQ and corresponding LQ"""
        _, _, h, w = hq.shape
        crop_h, crop_w = self.crop_size

        # Random crop position
        top = random.randint(0, max(0, h - crop_h))
        left = random.randint(0, max(0, w - crop_w))

        hq_crop = hq[:, :, top:top+crop_h, left:left+crop_w]

        if lq is not None:
            # Scale crop position for LQ
            lq_top = top // self.scale_factor
            lq_left = left // self.scale_factor
            lq_crop_h = crop_h // self.scale_factor
            lq_crop_w = crop_w // self.scale_factor
            lq_crop = lq[:, :, lq_top:lq_top+lq_crop_h, lq_left:lq_left+lq_crop_w]
        else:
            lq_crop = None

        return hq_crop, lq_crop

    def _generate_lq(self, hq: torch.Tensor) -> torch.Tensor:
        """Generate LQ from HQ by downscaling"""
        t, c, h, w = hq.shape
        lq_h, lq_w = h // self.scale_factor, w // self.scale_factor

        # Bicubic downscale
        lq = torch.nn.functional.interpolate(
            hq,
            size=(lq_h, lq_w),
            mode='bicubic',
            align_corners=False,
            antialias=True
        )

        # Optional: add degradation (blur, noise, compression artifacts)
        if self.augment and random.random() < 0.5:
            # Add slight Gaussian noise
            noise_level = random.uniform(0.01, 0.05)
            lq = lq + torch.randn_like(lq) * noise_level
            lq = lq.clamp(0, 1)

        return lq

    def _augment(
        self,
        hq: torch.Tensor,
        lq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation"""
        # Random horizontal flip
        if random.random() < 0.5:
            hq = torch.flip(hq, dims=[-1])
            lq = torch.flip(lq, dims=[-1])

        # Random vertical flip
        if random.random() < 0.5:
            hq = torch.flip(hq, dims=[-2])
            lq = torch.flip(lq, dims=[-2])

        # Random 90 degree rotation
        if random.random() < 0.5:
            k = random.randint(1, 3)
            hq = torch.rot90(hq, k, dims=[-2, -1])
            lq = torch.rot90(lq, k, dims=[-2, -1])

        return hq, lq

    def __len__(self) -> int:
        return len(self.video_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_name = self.video_names[idx]

        # Load HQ video
        hq_path = os.path.join(self.hq_dir, video_name)
        hq = self._load_video(hq_path)

        # Load or generate LQ
        if self.generate_lq:
            lq = None
        else:
            lq_path = os.path.join(self.lq_dir, video_name)
            lq = self._load_video(lq_path)

        # Sample frames (use same indices for HQ and LQ to ensure sync)
        hq, start_idx, indices = self._sample_frames(hq)
        if lq is not None:
            lq, _, _ = self._sample_frames(lq, start_idx=start_idx, indices=indices)

        # Random crop
        hq, lq = self._random_crop(hq, lq)

        # Generate LQ if needed
        if lq is None:
            lq = self._generate_lq(hq)

        # Augmentation
        if self.augment:
            hq, lq = self._augment(hq, lq)

        # Normalize to [-1, 1]
        hq = hq * 2 - 1
        lq = lq * 2 - 1

        return {
            'hq': hq,  # (T, C, H, W)
            'lq': lq,  # (T, C, H//scale, W//scale)
            'name': video_name,
        }


class ImagePairDataset(Dataset):
    """
    Dataset for loading paired LQ/HQ images
    Useful for pre-training or mixed training

    Args:
        data_root: Root directory containing lq/ and hq/ folders
        crop_size: Random crop size
        scale_factor: Downscale factor
        augment: Whether to apply augmentation
    """
    def __init__(
        self,
        data_root: str,
        crop_size: Tuple[int, int] = (256, 256),
        scale_factor: int = 4,
        augment: bool = True,
        generate_lq: bool = False,
        max_samples: Optional[int] = None,
    ):
        super().__init__()

        self.data_root = data_root
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.augment = augment
        self.generate_lq = generate_lq

        self.hq_dir = os.path.join(data_root, 'hq')
        self.lq_dir = os.path.join(data_root, 'lq')

        # Find images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        if generate_lq:
            self.image_names = [
                f for f in os.listdir(self.hq_dir)
                if os.path.splitext(f.lower())[1] in image_extensions
            ]
        else:
            hq_images = set(
                f for f in os.listdir(self.hq_dir)
                if os.path.splitext(f.lower())[1] in image_extensions
            )
            lq_images = set(
                f for f in os.listdir(self.lq_dir)
                if os.path.splitext(f.lower())[1] in image_extensions
            )
            self.image_names = list(hq_images & lq_images)

        if max_samples is not None:
            self.image_names = self.image_names[:max_samples]

        print(f"Found {len(self.image_names)} image pairs")

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_name = self.image_names[idx]

        # Load HQ
        hq_path = os.path.join(self.hq_dir, image_name)
        hq = read_image(hq_path).float() / 255.0  # (C, H, W)

        # Load or generate LQ
        if self.generate_lq:
            lq_h, lq_w = hq.shape[1] // self.scale_factor, hq.shape[2] // self.scale_factor
            lq = torch.nn.functional.interpolate(
                hq.unsqueeze(0),
                size=(lq_h, lq_w),
                mode='bicubic',
                align_corners=False,
                antialias=True
            ).squeeze(0)
        else:
            lq_path = os.path.join(self.lq_dir, image_name)
            lq = read_image(lq_path).float() / 255.0

        # Random crop
        _, h, w = hq.shape
        crop_h, crop_w = self.crop_size

        top = random.randint(0, max(0, h - crop_h))
        left = random.randint(0, max(0, w - crop_w))

        hq = hq[:, top:top+crop_h, left:left+crop_w]

        lq_top = top // self.scale_factor
        lq_left = left // self.scale_factor
        lq_crop_h = crop_h // self.scale_factor
        lq_crop_w = crop_w // self.scale_factor
        lq = lq[:, lq_top:lq_top+lq_crop_h, lq_left:lq_left+lq_crop_w]

        # Augmentation
        if self.augment:
            if random.random() < 0.5:
                hq = torch.flip(hq, dims=[-1])
                lq = torch.flip(lq, dims=[-1])
            if random.random() < 0.5:
                hq = torch.flip(hq, dims=[-2])
                lq = torch.flip(lq, dims=[-2])

        # Add temporal dimension for consistency with video
        hq = hq.unsqueeze(0)  # (1, C, H, W)
        lq = lq.unsqueeze(0)  # (1, C, H//s, W//s)

        # Normalize to [-1, 1]
        hq = hq * 2 - 1
        lq = lq * 2 - 1

        return {
            'hq': hq,
            'lq': lq,
            'name': image_name,
        }


def create_dataloader(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    num_frames: int = 16,
    crop_size: Tuple[int, int] = (256, 256),
    scale_factor: int = 4,
    augment: bool = True,
    generate_lq: bool = False,
    is_video: bool = True,
    distributed: bool = False,
    **kwargs
) -> DataLoader:
    """
    Create dataloader for training

    Args:
        data_root: Root data directory
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        num_frames: Frames per video clip
        crop_size: Random crop size
        scale_factor: LQ downscale factor
        augment: Whether to augment
        generate_lq: Generate LQ on-the-fly
        is_video: Video or image dataset
        distributed: Use distributed sampler

    Returns:
        DataLoader instance
    """
    if is_video:
        dataset = VideoPairDataset(
            data_root=data_root,
            num_frames=num_frames,
            crop_size=crop_size,
            scale_factor=scale_factor,
            augment=augment,
            generate_lq=generate_lq,
            **kwargs
        )
    else:
        dataset = ImagePairDataset(
            data_root=data_root,
            crop_size=crop_size,
            scale_factor=scale_factor,
            augment=augment,
            generate_lq=generate_lq,
            **kwargs
        )

    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader
