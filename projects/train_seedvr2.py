# // Copyright (c) 2025 Custom Training Implementation
# //
# // Licensed under the Apache License, Version 2.0 (the "License")

"""
Training script for SeedVR2 with APT (Adversarial Post-Training)

Usage:
    # Single GPU
    python projects/train_seedvr2.py --config configs_3b/train.yaml --data_root /path/to/data

    # Multi-GPU with torchrun
    torchrun --nproc-per-node=4 projects/train_seedvr2.py \
        --config configs_3b/train.yaml \
        --data_root /path/to/data \
        --sp_size 1

    # Multi-node
    torchrun --nnodes=2 --nproc-per-node=8 --node_rank=0 \
        projects/train_seedvr2.py --config configs_3b/train.yaml
"""

import os
import sys
import argparse
import datetime

import torch
import torch.distributed as dist

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import load_config
from common.distributed import init_torch, get_device, get_global_rank
from common.distributed.advanced import init_sequence_parallel
from common.seed import set_seed
from common.logger import get_logger

from projects.video_diffusion_sr.train import VideoDiffusionTrainer
from data.video_pair_dataset import create_dataloader


def load_text_embeddings(device: torch.device) -> dict:
    """
    Load pre-computed text embeddings

    For SeedVR2, we use fixed positive/negative prompts
    """
    # Check if pre-computed embeddings exist
    pos_emb_path = './pos_emb.pt'
    neg_emb_path = './neg_emb.pt'

    if os.path.exists(pos_emb_path) and os.path.exists(neg_emb_path):
        text_pos_embeds = torch.load(pos_emb_path, map_location=device)
        text_neg_embeds = torch.load(neg_emb_path, map_location=device)
    else:
        # Create dummy embeddings if not available
        # In practice, you should pre-compute these using a text encoder
        print("Warning: Text embeddings not found, using dummy embeddings")
        print("Please pre-compute embeddings using the text encoder")

        # Dummy embeddings (replace with actual dimensions)
        embed_dim = 5120  # T5-XXL dimension
        seq_len = 256

        text_pos_embeds = torch.zeros(1, seq_len, embed_dim, device=device)
        text_neg_embeds = torch.zeros(1, seq_len, embed_dim, device=device)

    return {
        'texts_pos': [text_pos_embeds],
        'texts_neg': [text_neg_embeds],
        'txt_shape': [torch.tensor([[seq_len]], device=device)],
    }


def main():
    parser = argparse.ArgumentParser(description="Train SeedVR2 with APT")

    # Config
    parser.add_argument("--config", type=str, default="./configs_3b/train.yaml",
                        help="Path to training config")

    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory for training data")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Number of frames per video clip")
    parser.add_argument("--crop_size", type=int, nargs=2, default=[256, 256],
                        help="Random crop size (H W)")

    # Model
    parser.add_argument("--dit_checkpoint", type=str, default="./ckpts/seedvr2_ema_3b.pth",
                        help="Path to pre-trained DiT checkpoint")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to training checkpoint to resume from")

    # Training
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Steps between logging")
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="Steps between checkpoints")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory for saving checkpoints")

    # Distributed
    parser.add_argument("--sp_size", type=int, default=1,
                        help="Sequence parallel size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Initialize distributed
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=3600))

    # Set seed
    set_seed(args.seed, same_across_ranks=True)

    # Load config
    config = load_config(args.config)

    # Create trainer
    trainer = VideoDiffusionTrainer(config)

    # Configure all components
    trainer.configure_all(
        dit_checkpoint=args.dit_checkpoint,
        sp_size=args.sp_size,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Create dataloader
    dataloader = create_dataloader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_frames=args.num_frames,
        crop_size=tuple(args.crop_size),
        scale_factor=4,
        augment=True,
        generate_lq=True,  # Generate LQ on-the-fly from HQ
        is_video=True,
        distributed=dist.is_initialized(),
    )

    # Load text embeddings
    text_embeds = load_text_embeddings(get_device())

    # Train
    trainer.train(
        dataloader=dataloader,
        text_embeds=text_embeds,
        num_epochs=args.num_epochs,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
    )

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
