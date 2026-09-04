# Scaled & Production Experiment Configurations

This directory contains organized configurations designed for higher throughput, architectural variants, pretrained CNN backbones, alternative loss functions, and higher image resolutions.

Directory Layout:
- **`unet_scratch/`**: Standard UNet architectures trained from scratch with varying widths, depths, loss formulations, and resolution budgets.
- **`efficientnet/`**: Pretrained EfficientNet backbones with UNet multi-scale feature skip connections or the **Sacrifice of Pixel** linear-zoom architecture.

---

## 1. EfficientNet Configurations (`configs/experiments/efficientnet/`)

| Configuration File | Backbone & Mode | Batch Size | Image Resolution | Target Use Case & Rationale |
| :--- | :--- | :--- | :--- | :--- |
| [`efficientnet_b0_unet.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/efficientnet/efficientnet_b0_unet.yaml) | EfficientNet-B0 (UNet Multi-Scale) | 16 | $256 \times 256$ | Pretrained ImageNet CNN encoder with multi-scale skip connections ($/2, /4, /8, /16, /32$) into progressive UNet decoder. |
| [`efficientnet_b0_sacrifice_of_pixel.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/efficientnet/efficientnet_b0_sacrifice_of_pixel.yaml) | EfficientNet-B0 (Sacrifice of Pixel) | 16 | $256 \times 256$ | **Sacrifice of Pixel**: Uses only the final bottleneck feature map ($8 \times 8$), feeds through a single Linear layer, then zooms out (bilinear) to pixel resolution. |
| [`efficientnet_b2_unet.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/efficientnet/efficientnet_b2_unet.yaml) | EfficientNet-B2 (UNet Multi-Scale) | 16 | $256 \times 256$ | Scaled EfficientNet-B2 backbone providing larger model capacity and deeper receptive fields. |
| [`efficientnet_b0_sacrifice_of_pixel_b32.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/efficientnet/efficientnet_b0_sacrifice_of_pixel_b32.yaml) | EfficientNet-B0 (Sacrifice of Pixel) | 32 | $256 \times 256$ | Accelerated training with batch size 32 leveraging low VRAM footprint of the sacrifice-of-pixel architecture. |

---

## 2. UNet Scratch Configurations (`configs/experiments/unet_scratch/`)

| Configuration File | Model Architecture | Features / Channels | Batch Size | Image Resolution | Target Use Case & Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`unet_wide_b32.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/unet_scratch/unet_wide_b32.yaml) | Wide UNet | `[128, 256, 512, 1024]` | 16 | $256 \times 256$ | Doubled channel width per level (~124M params) for capturing rich representation of subtle AI artifacts. |
| [`unet_deep_5stage_b32.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/unet_scratch/unet_deep_5stage_b32.yaml) | Deep 5-Stage UNet | `[64, 128, 256, 512, 1024]` | 16 | $256 \times 256$ | 5 hierarchical downsampling stages ($256 \to 8$) for expanding global receptive field. |
| [`unet_large_batch_b64.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/unet_scratch/unet_large_batch_b64.yaml) | Scaled LR UNet | `[64, 128, 256, 512]` | 16 | $256 \times 256$ | Scaled learning rate ($1.5 \times 10^{-3}$) and cosine warmup for accelerated optimization. |
| [`unet_highres_512_b16.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/unet_scratch/unet_highres_512_b16.yaml) | High-Resolution Deep UNet | `[64, 128, 256, 512, 1024]` | 16 | $512 \times 512$ | Higher resolution processing to resolve sub-pixel diffusion boundaries and sharp inpainting edges. |
| [`unet_heavy_wide_deep_b32.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/unet_scratch/unet_heavy_wide_deep_b32.yaml) | Heavy UNet | `[128, 256, 512, 1024]` | 16 | $256 \times 256$ | High capacity with 0.2 dropout and larger shuffle buffer. |
| [`unet_focal_hard_mining_b32.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/unet_scratch/unet_focal_hard_mining_b32.yaml) | Focal Loss UNet | `[64, 128, 256, 512, 1024]` | 16 | $256 \times 256$ | Binary Focal Loss ($\gamma=2.5, \alpha=0.35$) targeting extreme foreground-background mask imbalance. |
| [`unet_convtranspose_learned_up_b32.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/unet_scratch/unet_convtranspose_learned_up_b32.yaml) | Learned Upsampling UNet | `[64, 128, 256, 512]` | 16 | $256 \times 256$ | `bilinear: false` with learnable `ConvTranspose2d` decoder blocks instead of fixed bilinear interpolation. |
| [`unet_non_streaming_b32.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/unet_scratch/unet_non_streaming_b32.yaml) | Scaled Map Dataset UNet | `[64, 128, 256, 512, 1024]` | 16 | $256 \times 256$ | `streaming: false` indexable dataset loading with multi-worker shuffling and batch size 16. |

---

## How to Run

### 1. Training with Pretrained EfficientNet (Default UNet Mode)
```bash
sid-train --config configs/experiments/efficientnet/efficientnet_b0_unet.yaml
```

### 2. Training with 'Sacrifice of Pixel' Mode
```bash
sid-train --config configs/experiments/efficientnet/efficientnet_b0_sacrifice_of_pixel.yaml
```

### 3. Multi-Experiment Comparative Suite
```bash
sid-train --configs \
  configs/experiments/unet_scratch/unet_wide_b32.yaml \
  configs/experiments/efficientnet/efficientnet_b0_unet.yaml \
  configs/experiments/efficientnet/efficientnet_b0_sacrifice_of_pixel.yaml
```
