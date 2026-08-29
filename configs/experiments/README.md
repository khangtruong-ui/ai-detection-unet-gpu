# Scaled & Production Experiment Configurations

This subfolder contains scaled configurations designed for higher throughput, larger batch sizes, deeper architectures, higher image resolutions, and alternative loss functions. All configurations feature bounded sample budgets per epoch (`train_samples_per_epoch` and `val_samples`), preventing full-dataset memory or resource exhaustion on streaming datasets.

---

## Configuration Overview

| Configuration File | Model Architecture | Features / Channels | Batch Size | Image Resolution | Target Use Case & Rationale |
|---|---|---|---|---|---|
| [`unet_wide_b32.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/unet_wide_b32.yaml) | Wide UNet | `[128, 256, 512, 1024]` | 32 | $256 \times 256$ | Doubled channel width per level (~124M params) for capturing rich representation of subtle AI artifacts. |
| [`unet_deep_5stage_b32.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/unet_deep_5stage_b32.yaml) | Deep 5-Stage UNet | `[64, 128, 256, 512, 1024]` | 32 | $256 \times 256$ | 5 hierarchical downsampling stages ($256 \to 8$) for expanding the global receptive field. |
| [`unet_large_batch_b64.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/unet_large_batch_b64.yaml) | Large Batch UNet | `[64, 128, 256, 512]` | 64 | $256 \times 256$ | High GPU throughput with batch size 64, linear LR scaling ($1.5 \times 10^{-3}$), and cosine warmup. |
| [`unet_highres_512_b16.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/unet_highres_512_b16.yaml) | High-Resolution Deep UNet | `[64, 128, 256, 512, 1024]` | 16 | $512 \times 512$ | Higher resolution processing to resolve sub-pixel diffusion boundaries and sharp inpainting edges. |
| [`unet_heavy_wide_deep_b32.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/unet_heavy_wide_deep_b32.yaml) | Heavy UNet | `[128, 256, 512, 1024]` | 32 | $256 \times 256$ | High capacity with 8,000 samples/epoch, 0.2 dropout, and larger shuffle buffer. |
| [`unet_focal_hard_mining_b32.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/unet_focal_hard_mining_b32.yaml) | Focal Loss UNet | `[64, 128, 256, 512, 1024]` | 32 | $256 \times 256$ | Binary Focal Loss ($\gamma=2.5, \alpha=0.35$) targeting extreme foreground-background mask imbalance. |
| [`unet_convtranspose_learned_up_b32.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/unet_convtranspose_learned_up_b32.yaml) | Learned Upsampling UNet | `[64, 128, 256, 512]` | 32 | $256 \times 256$ | `bilinear: false` with learnable `ConvTranspose2d` decoder blocks instead of fixed bilinear interpolation. |
| [`unet_non_streaming_b32.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/unet_non_streaming_b32.yaml) | Scaled Map Dataset UNet | `[64, 128, 256, 512, 1024]` | 32 | $256 \times 256$ | `streaming: false` indexable dataset loading with multi-worker shuffling and batch size 32. |

---

## How to Run

### 1. Single Experiment Execution
```bash
python -m sid_unet.train --config configs/experiments/unet_wide_b32.yaml
```

### 2. Multi-Experiment Comparative Suite
Run multiple architectures sequentially to compare performance and generate a consolidated report:
```bash
python -m sid_unet.train --configs \
  configs/experiments/unet_wide_b32.yaml \
  configs/experiments/unet_deep_5stage_b32.yaml \
  configs/experiments/unet_large_batch_b64.yaml
```

### 3. Overriding Sample Budget on the Fly
```bash
python -m sid_unet.train \
  --config configs/experiments/unet_highres_512_b16.yaml \
  --override data.train_samples_per_epoch=2000 data.val_samples=400
```
