# Scaled & Production Experiment Configurations

This directory contains organized configurations designed for higher throughput, architectural variants, pretrained CNN backbones, alternative loss functions, and higher image resolutions.

Directory Layout:
- **`unet_scratch/`**: Standard UNet architectures trained from scratch with varying widths, depths, loss formulations, and resolution budgets.
- **`efficientnet/`**: Pretrained EfficientNet backbones with UNet multi-scale feature skip connections or the **Sacrifice of Pixel** linear-zoom architecture.
- **`sam3-qlora/`**: Meta SAM3 foundation model with 4-bit NormalFloat quantization (bitsandbytes) and Low-Rank Adaptation (LoRA) fine-tuned on streamed datasets like `KhangTruong/COCO-inpainted`.
- **`sam3_distil/`**: Distilled EfficientSAM3 foundation model (TinyViT, EfficientViT backbones) with PEFT Low-Rank Adaptation (LoRA / QLoRA) trained in the diffusion-diff environment.
- **`gap_sam/`**: GAP-SAM (Global Artifact Prior + sam-distil models: TinyViT, EfficientViT, RepViT) with frozen VAE reconstruction, Paired Artifact Encoder, zero-gated FiLM FPN modulation, and artifact classification.
- **`sd_vae_finetune/`**: Finetuned Stable Diffusion 1.5 VAE (AutoencoderKL) adapted for binary synthetic image mask segmentation.
- **`diffusion_diff/`**: Diffusion multi-noise feature decoder combining multi-step perturbations, frozen diffuser noise predictions, and sinusoidal embeddings with a configurable trainable decoder.
- **`diffusion_diff_v2/`**: Diffusion multi-noise feature decoder V2 with frozen encoder by default, no encoder skips, parallel frozen decoder, and perpendicular skip injection into trainable decoder.
- **`diffusion_diff_minimized/`**: Minimized ultra-fast diffusion-diff architecture using Segmind Tiny-SD (`segmind/tiny-sd`) in FP16 on CUDA with only 2 lines of computation (1 real latent, 1 chosen noisy latent from the diffusion model).


---

## 1. SAM3 + QLoRA Configurations (`configs/experiments/sam3-qlora/`)

| Configuration File | Model & Quantization | Dataset & Mode | LoRA Config | Loss | Target Use Case & Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`sam3_qlora_beyondthebrush_b1.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/sam3-qlora/sam3_qlora_beyondthebrush_b1.yaml) | SAM3 + 4-bit NF4 QLoRA | COCO-inpainted (streaming) | $r=8, \alpha=16$ ($q, v$) | Combined (BCE + Dice + Focal) | Baseline parameter-efficient fine-tuning on 12GB GPUs with gradient accumulation steps 16. |
| [`sam3_qlora_beyondthebrush_r16.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/sam3-qlora/sam3_qlora_beyondthebrush_r16.yaml) | SAM3 + 4-bit NF4 QLoRA | COCO-inpainted (streaming) | $r=16, \alpha=32$ ($q, k, v, out$) | Combined (BCE + Dice + Focal) | Expanded adaptation capacity across all attention projection layers. |
| [`sam3_qlora_beyondthebrush_focal.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/sam3-qlora/sam3_qlora_beyondthebrush_focal.yaml) | SAM3 + 4-bit NF4 QLoRA | COCO-inpainted (streaming) | $r=8, \alpha=16$ ($q, v$) | Focal ($\gamma=2.0, \alpha=0.25$) | Hard-mining loss addressing extreme foreground-background mask imbalance in subtle inpainting boundaries. |
| [`sam3_qlora_beyondthebrush_dice.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/sam3-qlora/sam3_qlora_beyondthebrush_dice.yaml) | SAM3 + 4-bit NF4 QLoRA | COCO-inpainted (streaming) | $r=8, \alpha=16$ ($q, v$) | Dice | Soft Sørensen-Dice direct optimization for sharp mask boundary overlap. |

---

## 2. EfficientNet Configurations (`configs/experiments/efficientnet/`)

| Configuration File | Backbone & Mode | Batch Size | Image Resolution | Target Use Case & Rationale |
| :--- | :--- | :--- | :--- | :--- |
| [`efficientnet_b0_unet.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/efficientnet/efficientnet_b0_unet.yaml) | EfficientNet-B0 (UNet Multi-Scale) | 64 | $256 \times 256$ | Pretrained ImageNet CNN encoder with multi-scale skip connections ($/2, /4, /8, /16, /32$) into progressive UNet decoder. |
| [`efficientnet_b0_sacrifice_of_pixel.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/efficientnet/efficientnet_b0_sacrifice_of_pixel.yaml) | EfficientNet-B0 (Sacrifice of Pixel) | 64 | $256 \times 256$ | **Sacrifice of Pixel**: Uses only the final bottleneck feature map ($8 \times 8$), feeds through a single Linear layer, then zooms out (bilinear) to pixel resolution. |
| [`efficientnet_b2_unet.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/efficientnet/efficientnet_b2_unet.yaml) | EfficientNet-B2 (UNet Multi-Scale) | 64 | $256 \times 256$ | Scaled EfficientNet-B2 backbone providing larger model capacity and deeper receptive fields. |
| [`efficientnet_b0_sacrifice_of_pixel_b32.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/efficientnet/efficientnet_b0_sacrifice_of_pixel_b32.yaml) | EfficientNet-B0 (Sacrifice of Pixel) | 128 | $256 \times 256$ | Accelerated training with batch size 128 leveraging low VRAM footprint of the sacrifice-of-pixel architecture. |

---

## 3. UNet Scratch Configurations (`configs/experiments/unet_scratch/`)

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

## 4. Finetuned Diffusion VAE Configurations (`configs/experiments/sd_vae_finetune/`)

| Configuration File | Model & Base VAE | Decoder Conv Out | Freeze Encoder | Loss | Target Use Case & Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`default.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/sd_vae_finetune/default.yaml) | `DiffusionVAEFinetune` (SD1.5 AutoencoderKL) | $128 \to 1$ Conv2d | `false` (End-to-End) | Combined (BCE + Dice) + Aux (0.2) | Fine-tuning Stable Diffusion 1.5 VAE directly to decode compressed latent distributions into binary tampering masks. |

---

## 5. Diffusion Multi-Noise Feature Decoder (`diffusion_diff`) (`configs/experiments/diffusion_diff/`)

| Configuration File | Model Architecture | Timesteps | Diffuser & VAE | Trainable Decoder | Loss | Target Use Case & Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [`default.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/diffusion_diff/default.yaml) | `DiffusionDiffModel` | `[100, 250, 500]` | Diffuser Frozen, VAE Trainable | `[256, 128, 64, 32]` Bilinear | Combined (BCE + Dice) + Aux (0.2) | Real image $x \to z_0$, inject noise at multiple timesteps, compute diffuser predicted noise $\hat{\epsilon}$ and sinusoidal embeddings ($t, \sigma$), concatenating into high-dimensional $Z$ with a trainable VAE autoencoder and decoder. |

---

## 6. Diffusion Multi-Noise Feature Decoder V2 (`diffusion_diff_v2`) (`configs/experiments/diffusion_diff_v2/`)

| Configuration File | Model Architecture | Timesteps | Frozen & Trainable Components | Trainable Decoder | Loss | Target Use Case & Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [`default.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/diffusion_diff_v2/default.yaml) | `DiffusionDiffV2Model` | `[100, 250, 500]` | Diffuser Frozen, Encoder Frozen, Parallel Decoder Frozen | `[256, 128, 64, 32]` Bilinear + Perpendicular Skips | Combined (BCE + Dice) + Aux (0.2) | Real image $x \to z_0$ with frozen encoder, no encoder skips, parallel frozen decoder decodes $z_0$ and injects multi-scale perpendicular skip features into trainable decoder fusing high-dimensional representation $Z$. |

---

## 7. SAM3-Distil (EfficientSAM3) + LoRA Configurations (`configs/experiments/sam3_distil/`)

| Configuration File | Backbone & Model | LoRA Adaptation | Dataset & Mode | Loss | Target Use Case & Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`default.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/sam3_distil/default.yaml) | EfficientSAM3 TinyViT-11m | LoRA ($r=16, \alpha=32$) | COCO-inpainted (streaming, $B=8$) | Combined (BCE + Dice + Focal) + Aux (0.2) | Baseline distilled SAM3 adaptation using identical environment (batch size 8, grad accum 4, AdamW, cosine) to diffusion-diff. |
| [`sam3_distil_tinyvit_lora.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/sam3_distil/sam3_distil_tinyvit_lora.yaml) | EfficientSAM3 TinyViT-11m | LoRA ($r=16, \alpha=32$) | COCO-inpainted (streaming, $B=8$) | Combined (BCE + Dice + Focal) + Aux (0.2) | Dedicated TinyViT LoRA targeting attention projections and DETR transformer layers with MobileCLIP-S0 text encoder. |
| [`sam3_distil_efficientvit_lora.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/sam3_distil/sam3_distil_efficientvit_lora.yaml) | EfficientSAM3 EfficientViT-b0 | LoRA ($r=16, \alpha=32$) | COCO-inpainted (streaming, $B=8$) | Combined (BCE + Dice + Focal) + Aux (0.2) | High-speed mobile-efficient backbone adapted via LoRA for ultra-low inference latency. |
| [`sam3_distil_qlora.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/sam3_distil/sam3_distil_qlora.yaml) | EfficientSAM3 TinyViT-11m | 4-bit NF4 QLoRA ($r=16, \alpha=32$) | COCO-inpainted (streaming, $B=8$) | Combined (BCE + Dice + Focal) + Aux (0.2) | 4-bit quantized base weights for minimal GPU VRAM consumption (~1.5GB - 2GB). |

---

## 8. Diffusion-Diff Minimized (`diffusion_diff_minimized`) (`configs/experiments/diffusion_diff_minimized/`)

| Configuration File | Model Architecture | Timesteps | Diffuser & Base Model | Trainable Decoder | Loss | Target Use Case & Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [`default.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/diffusion_diff_minimized/default.yaml) | `DiffusionDiffMinimizedModel` | `[250]` (1 noisy latent) | `segmind/tiny-sd` FP16 CUDA, Diffuser Frozen, Encoder Frozen, Parallel Decoder Frozen | `[256, 128, 64, 32]` Bilinear + Perpendicular Skips | Combined (BCE + Dice) + Aux (0.2) | Ultra-fast diffusion forensic architecture cutting compute time via `segmind/tiny-sd` and strictly 2 lines of computation (1 real latent, 1 chosen noisy latent from the diffusion model) with perpendicular skip injection. |
| [`diffusion_diff_minimized_bootstrap.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/diffusion_diff_minimized/diffusion_diff_minimized_bootstrap.yaml) | `DiffusionDiffMinimizedModel` | `[250]` (1 noisy latent) | `segmind/tiny-sd` FP16 CUDA, Diffuser Frozen, Encoder Frozen, Parallel Decoder Frozen | `[256, 128, 64, 32]` Bilinear + Perpendicular Skips | Combined (BCE + Dice) + Aux (0.2) | Bootstrapping kickstart enabled (30 epochs on 512 samples with channel-stream freeze) prior to full training. |

---

## 9. GAP-SAM (Global Artifact Prior + Distilled SAM3) (`configs/experiments/gap_sam/`)

| Configuration File | Backbone & Model | VAE Architecture | LoRA Config | Loss | Target Use Case & Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`default.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/gap_sam/default.yaml) | EfficientSAM3 TinyViT-11m | `stabilityai/sd-vae-ft-mse` | $r=16, \alpha=32$ | Combined + Aux (0.2) + Artifact (1.0) | Standard GAP-SAM baseline on COCO-inpainted with zero-gated FiLM multiscale FPN conditioning. |
| [`gap_sam_tinyvit_lora.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/gap_sam/gap_sam_tinyvit_lora.yaml) | EfficientSAM3 TinyViT-11m | `stabilityai/sd-vae-ft-mse` | $r=16, \alpha=32$ | Combined + Aux (0.2) + Artifact (1.0) | Dedicated TinyViT LoRA pairing observed and frozen VAE reconstruction features. |
| [`gap_sam_efficientvit_lora.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/gap_sam/gap_sam_efficientvit_lora.yaml) | EfficientSAM3 EfficientViT-b0 | `stabilityai/sd-vae-ft-mse` | $r=16, \alpha=32$ | Combined + Aux (0.2) + Artifact (1.0) | Ultra-fast mobile/edge inference latency profile. |
| [`gap_sam_repvit_lora.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/gap_sam/gap_sam_repvit_lora.yaml) | EfficientSAM3 RepViT-m1.1 | `stabilityai/sd-vae-ft-mse` | $r=16, \alpha=32$ | Combined + Aux (0.2) + Artifact (1.0) | Structural re-parameterization vision backbone. |
| [`gap_sam_sd21_vae.yaml`](file:///workspace/ai-detection-unet-gpu/configs/experiments/gap_sam/gap_sam_sd21_vae.yaml) | EfficientSAM3 TinyViT-11m | `stabilityai/stable-diffusion-2-1-base` (`vae`) | $r=8, \alpha=16$ | Combined + Aux (0.2) + Artifact (1.0) | Direct reproduction matching the research paper recipe. |

---

## How to Run

### 1. Training with SAM3 + QLoRA
```bash
sid-train --config configs/experiments/sam3-qlora/sam3_qlora_beyondthebrush_b1.yaml
```

### 2. Training with SAM3-Distil + LoRA (Diffusion-Diff Environment)
```bash
sid-train --config configs/experiments/sam3_distil/default.yaml
```

### 3. Training with GAP-SAM (Global Artifact Prior + sam-distil)
```bash
sid-train --config configs/experiments/gap_sam/default.yaml
```

### 4. Training with Pretrained EfficientNet (Default UNet Mode)
```bash
sid-train --config configs/experiments/efficientnet/efficientnet_b0_unet.yaml
```

### 5. Training with 'Sacrifice of Pixel' Mode
```bash
sid-train --config configs/experiments/efficientnet/efficientnet_b0_sacrifice_of_pixel.yaml
```

### 6. Training with Finetuned Diffusion VAE
```bash
sid-train --config configs/experiments/sd_vae_finetune/default.yaml
```

### 7. Training with Diffusion Multi-Noise Feature Decoder (Diffusion-Diff)
```bash
sid-train --config configs/experiments/diffusion_diff/default.yaml
```

### 8. Training with Diffusion Multi-Noise Feature Decoder V2 (Diffusion-Diff-V2)
```bash
sid-train --config configs/experiments/diffusion_diff_v2/default.yaml
```

### 9. Training with Diffusion-Diff Minimized (Ultra-Fast 2-Line Tiny-SD)
```bash
sid-train --config configs/experiments/diffusion_diff_minimized/default.yaml
```

### 10. Multi-Experiment Comparative Suite
```bash
sid-train --configs \
  configs/experiments/unet_scratch/unet_wide_b32.yaml \
  configs/experiments/efficientnet/efficientnet_b0_unet.yaml \
  configs/experiments/gap_sam/default.yaml \
  configs/experiments/sam3_distil/default.yaml \
  configs/experiments/diffusion_diff_minimized/default.yaml
```

