# 完整 APT (Adversarial Post-Training) 训练实现

## 已实现的核心组件

### 1. Distillation Loss (`losses/distillation_loss.py`)

完整的知识蒸馏损失函数，包含：

- **DistillationLoss**: 主要蒸馏损失
  - L2/L1/Huber loss 支持
  - SNR weighting (信噪比加权)
  - LPIPS 感知损失 (可选)

- **ConsistencyLoss**: 自一致性损失 (LCM-style)
  - 无需独立 teacher 模型
  - 强制不同时间步预测一致性

- **APTLossComplete**: 完整 APT 损失组合
  - Distillation loss (主要)
  - Adversarial loss (辅助)
  - Feature matching loss (辅助)
  - R1 regularization (稳定性)

### 2. DDIM Sampler (`losses/ddim_sampler.py`)

Teacher Model 的多步采样器：

- **DDIMSampler**: 确定性/随机采样
  - DDIM (eta=0, 确定性)
  - DDPM (eta=1, 随机性)
  - 可配置采样步数
  - CFG (Classifier-Free Guidance) 支持
  - 多种 beta schedule (linear, scaled_linear, cosine)
  - 多种预测类型 (epsilon, v_prediction, sample)

### 3. 训练流程更新

已修复的关键问题：
- ✅ 正确的 timestep (t=T_max)
- ✅ 正确的 noise 构建 (diffusion schedule)
- ✅ DDP 包装 discriminator
- ✅ R1 lazy regularization
- ✅ Scaler 状态保存
- ✅ 数值稳定性修复

---

## 完整 APT 训练流程

### 训练算法

```
For each training iteration:
  1. 加载 LQ/HQ 视频对
  2. VAE encode: lq_latent, hq_latent

  3. Teacher Forward (多步采样):
     - 使用 DDIM Sampler
     - 50 步采样得到高质量 teacher_output
     - 作为 student 的学习目标

  4. Student Forward (单步生成):
     - 添加噪声: noisy_hq = add_noise(hq_latent, t=999)
     - 单步预测: student_output = DiT(noisy_hq, lq_condition)

  5. Compute Losses:
     a) Distillation Loss (主要):
        L_distill = MSE(student_output, teacher_output)

     b) Adversarial Loss (辅助):
        - Decode: fake_video = VAE.decode(student_output)
        - Discriminator: D(fake_video) vs D(real_video)
        - L_adv = GAN_loss

     c) Feature Matching (辅助):
        L_fm = L1(D_features(fake), D_features(real))

  6. Total Loss:
     L_total = λ_distill * L_distill + λ_adv * L_adv + λ_fm * L_fm

  7. Update:
     - Update Student (DiT)
     - Update Discriminator
     - Update EMA
```

---

## 使用方法

### 方案 A: 使用现有 Teacher (推荐)

如果你已经有训练好的 DiT checkpoint：

```python
from losses import APTLossComplete, DDIMSampler

# 1. 初始化 Teacher Sampler
teacher_sampler = DDIMSampler(
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    prediction_type='epsilon',  # 根据你的模型调整
)

# 2. 初始化完整 APT Loss
apt_loss = APTLossComplete(
    lambda_distill=1.0,      # 蒸馏损失权重 (主要)
    lambda_adv=0.1,          # 对抗损失权重 (辅助)
    lambda_fm=1.0,           # 特征匹配权重 (辅助)
    lambda_r1=10.0,          # R1 正则化
    distill_loss_type='l2',  # L2 loss
    use_snr_weighting=True,  # SNR 加权
)

# 3. 训练循环
for batch in dataloader:
    lq_video, hq_video = batch['lq'], batch['hq']

    # VAE encode
    with torch.no_grad():
        lq_latent = vae.encode(lq_video)
        hq_latent = vae.encode(hq_video)

    # Teacher forward (多步采样)
    with torch.no_grad():
        teacher_output = teacher_sampler.sample(
            model=dit_teacher,  # 使用 EMA 权重或原始权重
            shape=hq_latent.shape,
            condition=lq_latent,
            num_inference_steps=50,  # 50 步采样
            eta=0.0,  # DDIM
        )

    # Student forward (单步生成)
    timestep = torch.full((batch_size,), 999, device=device)
    noise = torch.randn_like(hq_latent)
    noisy_hq = teacher_sampler.add_noise(hq_latent, noise, timestep)

    condition = get_condition(lq_latent)
    dit_input = torch.cat([noisy_hq, condition], dim=-1)

    student_output = dit_student(dit_input, timestep, text_embeds)

    # Decode for discriminator
    with torch.no_grad():
        fake_video = vae.decode(student_output)
        real_video = vae.decode(hq_latent)

    # Discriminator features
    real_pred, real_features = discriminator(real_video, return_features=True)
    fake_pred, fake_features = discriminator(fake_video, return_features=True)

    # Compute generator loss
    snr = teacher_sampler.get_snr(timestep)
    g_loss, g_dict = apt_loss.compute_generator_loss(
        student_pred=student_output,
        teacher_target=teacher_output,
        fake_pred=fake_pred,
        real_features=real_features,
        fake_features=fake_features,
        timestep=timestep,
        snr=snr,
    )

    # Backward and update
    g_loss.backward()
    optimizer_g.step()

    # Update discriminator (same as before)
    ...
```

### 方案 B: Self-Distillation (无需独立 Teacher)

使用 EMA 权重作为 teacher：

```python
# 使用 student 的 EMA 权重作为 teacher
ema_model.apply_shadow(dit_student)  # 临时应用 EMA 权重

with torch.no_grad():
    teacher_output = teacher_sampler.sample(
        model=dit_student,  # 使用 EMA 权重
        shape=hq_latent.shape,
        condition=lq_latent,
        num_inference_steps=50,
    )

ema_model.restore(dit_student)  # 恢复原始权重

# 然后正常训练 student
...
```

### 方案 C: Progressive Distillation

逐步减少采样步数：

```python
# Stage 1: 50 steps -> 25 steps
teacher_steps = 50
student_steps = 25

# Stage 2: 25 steps -> 10 steps
teacher_steps = 25
student_steps = 10

# Stage 3: 10 steps -> 1 step
teacher_steps = 10
student_steps = 1
```

---

## 配置文件更新

在 `configs_3b/train.yaml` 中添加：

```yaml
# APT Training Configuration
apt:
  # Distillation
  use_distillation: true
  teacher_steps: 50        # Teacher 采样步数
  student_steps: 1         # Student 采样步数 (one-step)
  distill_loss_type: l2    # l2, l1, huber
  use_snr_weighting: true  # SNR 加权

  # Loss weights
  lambda_distill: 1.0      # 蒸馏损失 (主要)
  lambda_adv: 0.1          # 对抗损失 (辅助)
  lambda_fm: 1.0           # 特征匹配 (辅助)
  lambda_lpips: 0.0        # 感知损失 (可选)

  # Teacher configuration
  use_ema_teacher: true    # 使用 EMA 作为 teacher
  teacher_checkpoint: null # 或指定独立 teacher checkpoint

  # Sampling
  sampler_type: ddim       # ddim or ddpm
  sampler_eta: 0.0         # 0=DDIM, 1=DDPM
  prediction_type: epsilon # epsilon, v_prediction, sample

# Diffusion schedule
diffusion:
  num_timesteps: 1000
  beta_start: 0.0001
  beta_end: 0.02
  beta_schedule: linear    # linear, scaled_linear, cosine
```

---

## 预期效果

### 训练收敛指标

```
Epoch 1:
  distill_loss: 0.15 -> 0.08  (下降)
  g_adv: 0.5 -> 0.3           (下降)
  g_fm: 5.0 -> 2.0            (下降)
  d_adv: 1.5 (稳定)
  d_r1: 25.0 (稳定)

Epoch 10:
  distill_loss: 0.02 (收敛)
  g_adv: 0.1 (收敛)
  g_fm: 0.5 (收敛)
```

### 质量提升

- **速度**: 50 步 -> 1 步 (50x 加速)
- **质量**: 保持 95%+ 的 teacher 质量
- **PSNR**: 接近 teacher 的 PSNR
- **LPIPS**: 略高于 teacher (可接受)

---

## 下一步工作

### 必须完成 (才能训练)

1. ✅ Distillation Loss - 已完成
2. ✅ DDIM Sampler - 已完成
3. ⏳ 更新 `train.py` 集成 teacher forward
4. ⏳ 添加 teacher model 加载逻辑
5. ⏳ 测试完整训练流程

### 可选优化

1. Progressive distillation (50->25->10->1)
2. CFG distillation (蒸馏 guidance)
3. Latent consistency models (LCM)
4. 多分辨率训练
5. 在线 hard example mining

---

## 参考实现

- **SDXL-Turbo**: https://github.com/Stability-AI/generative-models
- **LCM**: https://github.com/luosiallen/latent-consistency-model
- **Progressive Distillation**: https://arxiv.org/abs/2202.00512
- **Consistency Models**: https://arxiv.org/abs/2303.01469

---

## 故障排查

### 问题 1: Distillation loss 不下降

**原因**: Teacher 和 Student 输出格式不匹配

**解决**: 检查 `prediction_type` 配置
```python
# 确保一致
teacher_sampler.prediction_type == 'epsilon'
dit.prediction_type == 'epsilon'
```

### 问题 2: 训练不稳定

**原因**: Loss 权重不平衡

**解决**: 调整权重
```yaml
lambda_distill: 1.0   # 保持为 1
lambda_adv: 0.01      # 降低 GAN loss 权重
lambda_fm: 0.1        # 降低 FM loss 权重
```

### 问题 3: 生成质量差

**原因**: Teacher 采样步数太少

**解决**: 增加 teacher 步数
```yaml
teacher_steps: 100  # 从 50 增加到 100
```

### 问题 4: 显存不足

**原因**: Teacher forward 占用额外显存

**解决**:
- 减少 batch size
- 使用 gradient checkpointing
- Teacher forward 使用 fp16
- 每 N 步才做一次 teacher forward

---

## 总结

当前实现已包含完整的 APT 训练所需的核心组件：

✅ **Distillation Loss** - 知识蒸馏
✅ **DDIM Sampler** - Teacher 采样
✅ **APT Loss Complete** - 完整损失组合
✅ **数值稳定性** - 所有修复已完成
✅ **分布式训练** - DDP 支持

**下一步**: 集成到 `train.py` 并进行端到端测试。
