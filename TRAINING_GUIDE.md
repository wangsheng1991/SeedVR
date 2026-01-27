# SeedVR2 APT Training Guide

## 新增文件结构

```
F:\2026\SeedVR\
├── models/
│   └── discriminator/
│       ├── __init__.py
│       ├── video_discriminator.py      # StyleGAN2风格判别器
│       └── patch_discriminator.py      # PatchGAN判别器
├── losses/
│   ├── __init__.py
│   └── apt_loss.py                     # APT损失函数
├── data/
│   └── video_pair_dataset.py           # 视频对数据集
├── projects/
│   ├── video_diffusion_sr/
│   │   └── train.py                    # VideoDiffusionTrainer
│   ├── train_seedvr2.py                # 训练启动脚本
│   └── prepare_text_embeddings.py      # 文本嵌入预计算
└── configs_3b/
    └── train.yaml                      # 训练配置
```

## 数据准备

### 目录结构
```
/path/to/your/data/
├── hq/                    # 高质量视频
│   ├── video001.mp4
│   ├── video002.mp4
│   └── ...
└── lq/                    # 低质量视频 (可选，可自动生成)
    ├── video001.mp4
    ├── video002.mp4
    └── ...
```

### 数据要求
- HQ视频: 720p或更高分辨率
- LQ视频: 可以手动准备，或设置 `generate_lq=True` 自动从HQ降采样生成
- 支持格式: mp4, avi, mov, mkv, webm

## 训练步骤

### 1. 准备文本嵌入
```bash
cd F:\2026\SeedVR
python projects/prepare_text_embeddings.py
```

### 2. 单卡训练
```bash
python projects/train_seedvr2.py \
    --config configs_3b/train.yaml \
    --data_root /path/to/your/data \
    --dit_checkpoint ./ckpts/seedvr2_ema_3b.pth \
    --batch_size 1 \
    --num_frames 16 \
    --crop_size 256 256 \
    --save_dir ./checkpoints
```

### 3. 多卡训练 (推荐)
```bash
# 4卡训练
torchrun --nproc-per-node=4 projects/train_seedvr2.py \
    --config configs_3b/train.yaml \
    --data_root /path/to/your/data \
    --dit_checkpoint ./ckpts/seedvr2_ema_3b.pth \
    --batch_size 1 \
    --num_frames 16 \
    --crop_size 256 256 \
    --save_dir ./checkpoints

# 8卡训练 + 序列并行
torchrun --nproc-per-node=8 projects/train_seedvr2.py \
    --config configs_3b/train.yaml \
    --data_root /path/to/your/data \
    --dit_checkpoint ./ckpts/seedvr2_ema_3b.pth \
    --batch_size 1 \
    --num_frames 32 \
    --crop_size 512 512 \
    --sp_size 2 \
    --save_dir ./checkpoints
```

### 4. 从检查点恢复
```bash
torchrun --nproc-per-node=4 projects/train_seedvr2.py \
    --config configs_3b/train.yaml \
    --data_root /path/to/your/data \
    --resume ./checkpoints/checkpoint_10000.pth \
    --save_dir ./checkpoints
```

## 配置说明

### 关键参数 (configs_3b/train.yaml)

```yaml
# 判别器
discriminator:
  type: patch           # 'patch' (稳定) 或 'video' (更强)
  base_channels: 64     # 基础通道数
  num_layers: 3         # 层数

# 优化器
optimizer:
  g_lr: 1.0e-5         # 生成器学习率 (建议较小)
  d_lr: 1.0e-4         # 判别器学习率

# 损失权重
loss:
  lambda_adv: 1.0      # 对抗损失权重
  lambda_fm: 10.0      # 特征匹配损失权重
  lambda_r1: 10.0      # R1正则化权重
  r1_interval: 16      # R1计算间隔 (节省计算)
```

## 显存需求估算

| 配置 | 显存需求 | 推荐GPU |
|------|---------|---------|
| 3B + 256x256 + 16帧 | ~40GB | 1x A100-80G |
| 3B + 512x512 + 16帧 | ~60GB | 1x H100-80G |
| 3B + 720p + 32帧 | ~160GB | 4x A100-40G |
| 7B + 720p + 32帧 | ~320GB | 8x A100-40G |

## 训练技巧

1. **从小分辨率开始**: 先用256x256训练，验证流程正确后再提高分辨率
2. **监控损失**:
   - `d_adv` 应该在0.5-1.5之间波动
   - `g_adv` 应该逐渐下降
   - 如果`d_adv`趋近0，说明判别器太强，降低`d_lr`
3. **R1正则化**: 如果训练不稳定，增加`lambda_r1`
4. **特征匹配**: `lambda_fm`是稳定训练的关键，不建议设为0

## 常见问题

### Q: 训练不稳定/模式崩溃
A:
- 增加 `lambda_r1` (如 20.0)
- 降低 `d_lr` (如 5e-5)
- 使用 `loss_type: hinge`

### Q: 生成结果模糊
A:
- 增加 `lambda_adv`
- 减少 `lambda_fm`
- 训练更多步数

### Q: 显存不足
A:
- 减小 `batch_size`
- 减小 `crop_size`
- 减少 `num_frames`
- 使用 `gradient_checkpointing`
- 使用多卡 + 序列并行

## 推理测试

训练完成后，使用EMA权重进行推理:
```bash
python projects/inference_seedvr2_3b.py \
    --video_path ./test_videos \
    --output_dir ./results \
    --checkpoint ./checkpoints/checkpoint_final.pth
```
