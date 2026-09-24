# SID-UNet: UNet & Pretrained CNN Backbones for AI-Generated Synthetic Image Masking & Classification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](https://pytorch.org/)
[![Datasets](https://img.shields.io/badge/HuggingFace-Datasets-orange.svg)](https://huggingface.co/datasets/saberzl/SID_Set)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A modular, config-driven PyTorch framework for detecting and segmenting AI-generated or tampered regions in images. Supports standard **UNet** architectures as well as **Pretrained EfficientNet Backbones** with either multi-scale skip connections or the novel **Sacrifice of Pixel** bottleneck-linear architecture, alongside an optional **3-class auxiliary classification head** (**`Real`**, **`Fully AI`**, **`Partially AI / Inpainting`**).

Supports large-scale streaming and local datasets including standard 2-column image/mask datasets like [**KhangTruong/IMD2020**](https://huggingface.co/datasets/KhangTruong/IMD2020) and multi-class datasets like [**saberzl/SID_Set**](https://huggingface.co/datasets/saberzl/SID_Set). Features native **streaming dataset support** (`streaming = true`), flexible loss functions (BCE, Soft Dice, Focal, Combined), comprehensive metric tracking, **continuous master reports with collision checking**, automated qualitative random sample visualizer (**`sid-illu`**), and a complete unit/integration test suite.

---

## Table of Contents

- [Key Features](#key-features)
- [Dataset Specification](#dataset-specification)
- [Architectures & Backbones](#architectures--backbones)
  - [1. Standard UNet](#1-standard-unet)
  - [2. Pretrained EfficientNet Backbone (Default UNet Multi-Scale Decoder)](#2-pretrained-efficientnet-backbone-default-unet-multi-scale-decoder)
  - [3. EfficientNet 'Sacrifice of Pixel' Architecture](#3-efficientnet-sacrifice-of-pixel-architecture)
  - [4. SAM3 + QLoRA Architecture](#4-sam3--qlora-architecture)
  - [5. Finetuned Diffusion VAE (SD1.5 AutoencoderKL)](#5-finetuned-diffusion-vae-sd15-autoencoderkl)
  - [6. Diffusion Multi-Noise Feature Decoder (Diffusion-Diff)](#6-diffusion-multi-noise-feature-decoder-diffusion-diff)
  - [7. Diffusion Multi-Noise Latent Feature Decoder V2 (Diffusion-Diff-V2)](#7-diffusion-multi-noise-latent-feature-decoder-v2-diffusion-diff-v2)
- [Mechanisms & Architectural Principles](#mechanisms--architectural-principles)
  - [1. Problem Formulation & Task Definition](#1-problem-formulation--task-definition)
  - [2. Multi-Scale Feature Representation & Skip Connections](#2-multi-scale-feature-representation--skip-connections)
  - [3. Auxiliary Classifier & Multi-Task Semantic Regularization](#3-auxiliary-classifier--multi-task-semantic-regularization)
  - [4. Streaming Dataset Engine & Dynamic Mask Synthesis](#4-streaming-dataset-engine--dynamic-mask-synthesis)
  - [5. Mask Post-Processing Pipeline (Noise Suppression, Hole Filling, Morphology)](#5-mask-post-processing-pipeline-noise-suppression-hole-filling-morphology)
  - [6. SAM3 Spatial Join & Boundary Contrast Refinement](#6-sam3-spatial-join--boundary-contrast-refinement)
  - [7. Simultaneous Multi-Stage Ablation Evaluation](#7-simultaneous-multi-stage-ablation-evaluation)
  - [8. Continuous Master Reports & Collision Skipping](#8-continuous-master-reports--collision-skipping)
  - [9. Automated Visual Illustration (`sid-illu`) & Heatmap Generation](#9-automated-visual-illustration-sid-illu--heatmap-generation)
  - [10. Memory Management & OOM Dynamic Auto-Recovery](#10-memory-management--oom-dynamic-auto-recovery)
  - [11. Runtime Learnability Diagnostics & Debug Mode (nn-toolbox)](#11-runtime-learnability-diagnostics--debug-mode-nn-toolbox)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration System](#configuration-system)
  - [1. Configuration Field Reference](#1-configuration-field-reference)
  - [2. Checkpoint Timing & Resumption Parameters](#2-checkpoint-timing--resumption-parameters)
  - [3. Architecture-Specific Config Examples](#3-architecture-specific-config-examples)
- [Quickstart: How to Run](#quickstart-how-to-run)
  - [1. Training](#1-training)
  - [2. Evaluation & Benchmarking](#2-evaluation--benchmarking)
  - [3. Random Sample Visual Illustration (`sid-illu`)](#3-random-sample-visual-illustration-sid-illu)
  - [4. Inference & Mask Prediction](#4-inference--mask-prediction)
- [Loss Functions & Metrics](#loss-functions--metrics)
- [Running Tests](#running-tests)
- [License](#license)

---

## Key Features

- **Multiple Backbone Architectures**:
  - **Standard UNet**: Modular depth, configurable channel dimensions, bilinear or transposed convolutions.
  - **Pretrained EfficientNet-UNet**: Leverage ImageNet pretrained CNN representations with multi-scale skip connections ($/2, /4, /8, /16, /32$) feeding into a progressive decoder.
  - **Sacrifice of Pixel Mode**: Uses **only the final bottleneck feature map** ($8 \times 8$ or $7 \times 7$), routes through a single Linear layer, and zooms out to match full image resolution.
  - **Finetuned Diffusion VAE (SD1.5 AutoencoderKL)**: Adapts pretrained latent diffusion VAE to decode compressed latent representations directly into binary tampering mask space with end-to-end or decoder-only fine-tuning.
  - **Diffusion Multi-Noise Feature Decoder (Diffusion-Diff)**: Advanced latent perturbation forensics extracting $z_0$, adding noise across multiple diffusion timesteps, computing frozen diffuser predicted noise and sinusoidal schedule embeddings ($t, \sigma$), concatenated into high-dimensional $Z$ decoded by a fully configurable trainable decoder.
  - **Diffusion Multi-Noise Latent Feature Decoder V2 (Diffusion-Diff-V2)**: State-of-the-art dual-decoder generative perturbation architecture with **frozen VAE encoder by default**, **eliminated encoder-to-decoder skips by default**, and a **parallel pretrained, frozen VAE decoder** providing rich generative decoding priors through **perpendicular skip connections** directly into the trainable decoder.
- **Continuous Master Reports & Automatic Checkpoint Continuation**:
  - **Automatic Repository Checkpoint Discovery**: Automatically scans repository and output directories (`outputs/RUN/...`, `checkpoints/`, etc.) for existing checkpoints (`checkpoint_latest.pt`, `checkpoint_periodic.pt`, `checkpoint_best.pt`) and displays a highlighted on-screen notification with detailed resume metadata (epoch, global step, metric score).
  - **Hugging Face Model Repository Resumption**: Download and resume training or evaluation directly from Hugging Face Hub repositories (`--resume-repo <owner/repo>` or `--resume hf://<owner/repo>`).
  - **Full Training State Resumption**: Seamlessly restores model weights, optimizer state, LR scheduler, GradScaler, global step, best metric score, and training history across training epochs.
  - **Collision Detection & Skip**: Automatically detects if a combination of model config (with checkpoint) and dataset config has already been trained or evaluated, skipping duplicate work with clear notifications while preserving and updating consolidated continuous master reports.
- **Random Sample Visual Illustration CLI (`sid-illu`)**:
  - Interactively sample random examples from datasets and produce side-by-side comparative grids across multiple checkpoints, detailing per-sample IoU, Dice/F1, overlays, and error maps.
- **Universal Dataset Support**:
  - **Standard 2-Column Datasets ([KhangTruong/IMD2020](https://huggingface.co/datasets/KhangTruong/IMD2020))**: Contains `image` and `mask` across **`train`**, **`validation`**, and **`test`** subsets.
  - **3-Class Labeled Datasets ([saberzl/SID_Set](https://huggingface.co/datasets/saberzl/SID_Set))**:
    - **Label `0` (Real/Authentic)**: Pure zero mask ($\mathbf{0}$).
    - **Label `1` (Fully Synthetic)**: Pure one mask ($\mathbf{1}$).
    - **Label `2` (Partially Synthetic / Tampered)**: Ground truth mask binarized to $\{0.0, 1.0\}$.
- **Ablation & Refinement Pipelines**:
  - Built-in multi-stage ablation: Raw model, Post-Processing (component filtering, hole filling, morphology), and SAM3 zero-shot boundary refinement.

---

## Dataset Specification

The framework supports multiple dataset formats:

### 1. Common / Regular 2-Column Format ([KhangTruong/IMD2020](https://huggingface.co/datasets/KhangTruong/IMD2020))
- **Subsets Available**: `train`, `validation`, and `test`.
- **`image`**: RGB image (PIL Image or tensor).
- **`mask`**: Binary / grayscale segmentation mask for manipulated or inpainted regions.
- When **`label`** is not explicitly provided, class indicators are inferred automatically from pixel statistics ($0$: authentic, $1$: fully synthetic, $2$: tampered).

### 2. Multi-Class Labeled Format ([saberzl/SID_Set](https://huggingface.co/datasets/saberzl/SID_Set))
- **`image`**: RGB image ($1024 \times 1024$ or variable resolutions).
- **`label`**: Integer class indicator ($0, 1, 2$).
- **`mask`**: Segmentation mask for tampered/inpainted regions:
  - When $\mathrm{label} = 0$, target mask is **all zeros** ($\mathbf{0}$).
  - When $\mathrm{label} = 1$, target mask is **all ones** ($\mathbf{1}$).
  - When $\mathrm{label} = 2$, target mask is thresholded to binary $\{0.0, 1.0\}$.

### 3. High-Resolution Inpainting & Tampering Format ([KhangTruong/BeyondTheBrush](https://huggingface.co/datasets/KhangTruong/BeyondTheBrush))
- **Subsets Available**: `train`, `validation`, and `test` splits.
- **`image`**: RGB images at resolutions up to $4000 \times 3000$.
- **`mask`**: Binary mask ($L$ mode, values $0$ and $255$) highlighting fine inpainting, brush edits, and object synthesis.
- Native streaming execution via `streaming: true` avoiding local disk storage of large image collections.

---

## Architectures & Backbones

### 1. Standard UNet

```
                      Input Image (3, H, W)
                               │
                ┌──────────────▼───────────────┐
                │    DoubleConv (in -> 64)      │───────────────┐ (Skip 1: /1)
                └──────────────┬───────────────┘                │
                ┌──────────────▼───────────────┐                │
                │   Down: MaxPool -> Conv (128) │─────────────┐ │ (Skip 2: /2)
                └──────────────┬───────────────┘              │ │
                ┌──────────────▼───────────────┐              │ │
                │   Down: MaxPool -> Conv (256) │───────────┐ │ │ (Skip 3: /4)
                └──────────────┬───────────────┘            │ │ │
                ┌──────────────▼───────────────┐            │ │ │
                │   Down: MaxPool -> Conv (512) │─────────┐ │ │ │ (Skip 4: /8)
                └──────────────┬───────────────┘          │ │ │ │
                               ▼                          │ │ │ │
                ┌───────────────────────────────┐         │ │ │ │
                │     Bottleneck Conv (512)     │         │ │ │ │ (/16)
                └───────┬───────────────┬───────┘         │ │ │ │
                        │               │                 │ │ │ │
                        │       ┌───────▼──────────────┐  │ │ │ │
                        │       │ Auxiliary Classifier │  │ │ │ │
                        │       │ AdaptivePool -> MLP  │  │ │ │ │
                        │       └───────┬──────────────┘  │ │ │ │
                        │               ▼                 │ │ │ │
                        │     Class Logits (B, 3)         │ │ │ │
                        │                                 │ │ │ │
                ┌───────▼───────────────────────┐         │ │ │ │
                │ Up: Bilinear + Conv (256)     │◄────────┘ │ │ │
                └───────┬───────────────────────┘           │ │ │
                ┌───────▼───────────────────────┐           │ │ │
                │ Up: Bilinear + Conv (128)     │◄──────────┘ │ │
                └───────┬───────────────────────┘             │ │
                ┌───────▼───────────────────────┐             │ │
                │ Up: Bilinear + Conv (64)      │◄────────────┘ │
                └───────┬───────────────────────┘               │
                ┌───────▼───────────────────────┐               │
                │ Up: Bilinear + Conv (64)      │◄──────────────┘
                └───────┬───────────────────────┘
                ┌───────▼───────────────┐
                │   OutConv: 1x1 Conv   │
                └───────┬───────────────┘
                        ▼
              Binary Mask Logits (1, H, W)
```

---

### 2. Pretrained EfficientNet Backbone (Default UNet Multi-Scale Decoder)

Combines an ImageNet pretrained **EfficientNet** feature extractor (`efficientnet_b0` through `b7`) with a multi-stage progressive decoder.

```
                      Input Image (3, H, W)
                               │
                ┌──────────────▼───────────────┐
                │ EfficientNet Stage 0..1 (/2) │────────────────┐ (Skip 1: /2)
                └──────────────┬───────────────┘                │
                ┌──────────────▼───────────────┐                │
                │ EfficientNet Stage 2    (/4) │──────────────┐ │ (Skip 2: /4)
                └──────────────┬───────────────┘              │ │
                ┌──────────────▼───────────────┐              │ │
                │ EfficientNet Stage 3    (/8) │────────────┐ │ │ (Skip 3: /8)
                └──────────────┬───────────────┘            │ │ │
                ┌──────────────▼───────────────┐            │ │ │
                │ EfficientNet Stage 4..5 (/16)│──────────┐ │ │ │ (Skip 4: /16)
                └──────────────┬───────────────┘          │ │ │ │
                               ▼                          │ │ │ │
                ┌───────────────────────────────┐         │ │ │ │
                │ EfficientNet Stage 6..8 (/32) │         │ │ │ │ (Bottleneck)
                └───────┬───────────────┬───────┘         │ │ │ │
                        │               │                 │ │ │ │
                        │       ┌───────▼──────────────┐  │ │ │ │
                        │       │ Auxiliary Classifier │  │ │ │ │
                        │       │ AdaptivePool -> MLP  │  │ │ │ │
                        │       └───────┬──────────────┘  │ │ │ │
                        │               ▼                 │ │ │ │
                        │      Class Logits (B, 3)        │ │ │ │
                        │                                 │ │ │ │
                ┌───────▼───────────────────────┐         │ │ │ │
                │ UpBlock 4: Up + Skip (C4)     │◄────────┘ │ │ │
                └───────┬───────────────────────┘           │ │ │
                ┌───────▼───────────────────────┐           │ │ │
                │ UpBlock 3: Up + Skip (C3)     │◄──────────┘ │ │
                └───────┬───────────────────────┘             │ │
                ┌───────▼───────────────────────┐             │ │
                │ UpBlock 2: Up + Skip (C2)     │◄────────────┘ │
                └───────┬───────────────────────┘               │
                ┌───────▼───────────────────────┐               │
                │ UpBlock 1: Up + Skip (C1)     │◄──────────────┘
                └───────┬───────────────────────┘
                ┌───────▼───────────────────────┐
                │ Final Bilinear Upsample (2x)  │
                └───────┬───────────────────────┘
                ┌───────▼───────────────┐
                │   OutConv: 1x1 Conv   │
                └───────┬───────────────┘
                        ▼
              Binary Mask Logits (1, H, W)
```

---

### 3. EfficientNet 'Sacrifice of Pixel' Architecture

In this specialized mode (`sacrifice_of_pixel: true`), intermediate feature skip connections are bypassed entirely. The network utilizes **only the final bottleneck feature map** ($8 \times 8$ or $7 \times 7$), feeds it through a single Linear layer, and zooms out to match full pixel image size:

```
                      Input Image (3, H, W)
                               │
                ┌──────────────▼───────────────┐
                │ EfficientNet Backbone (All)  │
                └──────────────┬───────────────┘
                               ▼
                ┌───────────────────────────────┐
                │ Final Bottleneck Feature Map  │
                │     (B, C_bot, 8, 8)          │
                └───────┬───────────────┬───────┘
                        │               │
                        │       ┌───────▼──────────────┐
                        │       │ Auxiliary Classifier │
                        │       │ AdaptivePool -> MLP  │
                        │       └───────┬──────────────┘
                        │               ▼
                        │      Class Logits (B, 3)
                        │
                ┌───────▼───────────────────────┐
                │  Single Linear Projection     │
                │  (C_bot -> out_channels)      │
                └───────┬───────────────────────┘
                        ▼
                ┌───────────────────────────────┐
                │ Low-Res Logits (B, 1, 8, 8)   │
                └───────┬───────────────────────┘
                        ▼
                ┌───────────────────────────────┐
                │  Zoom Out (Bilinear Interp)   │
                │  to Original Image Size (H, W)│
                └───────┬───────────────────────┘
                        ▼
              Binary Mask Logits (1, H, W)
```

---

### 4. SAM3 + QLoRA Architecture

Meta's **Segment Anything Model 3 (SAM3)** integrated with **Quantized Low-Rank Adaptation (QLoRA)** allows fine-tuning foundation vision-language segmentation models directly on consumer GPUs (e.g. 12GB RTX 3060):

```
                      Input Image (3, H, W)             Text Prompt: "tampered region"
                               │                                       │
                               ▼                                       ▼
                     Bilinear Resize (1008x1008)               CLIP Text Tokenizer
                               │                                       │
                               ▼                                       ▼
                    ┌─────────────────────────────────────────────────────┐
                    │               SAM3 Foundation Backbone              │
                    │   - ViT Vision Encoder (4-bit NF4 Quantized)        │
                    │   - LoRA Adapters (r=8..16 on q_proj, v_proj)       │
                    │   - Text Encoder & Cross-Attention                  │
                    │   - DETR Encoder & Decoder with Object Queries      │
                    │   - Multiscale Mask Decoder & FPN Pixel Head        │
                    └──────────────────────────┬──────────────────────────┘
                                               │
                                               ▼
                              Semantic Segmentation Output (B, 1, H_fpn, W_fpn)
                                               │
                                               ▼
                                   Bilinear Interpolate (H, W)
                                               │
                                               ▼
                                  Binary Mask Logits (B, 1, H, W)
```

Key features:
- **4-Bit NormalFloat (NF4) Quantization**: Reduces the 3.4GB FP32/BF16 base model footprint to only **~660 MB** of VRAM.
- **PEFT LoRA Adapters**: Injects low-rank decomposition matrices ($r=8, \alpha=16$) into self-attention projection layers (`q_proj`, `v_proj`). Less than **0.5%** of total parameters are trainable.
- **Hardware Compatibility**: Verified to fit and train on **12GB VRAM GPUs** (such as NVIDIA RTX 3060) with batch size 1 and gradient accumulation (e.g. 16 steps).
- **Prompt Conditioning**: Defaults to text prompt `"tampered region"` to focus the DETR queries and mask decoder cross-attention on manipulated or inpainted artifacts.

---

### 5. Finetuned Diffusion VAE (SD1.5 AutoencoderKL)

This architecture adapts a pretrained Variational Autoencoder from latent diffusion models (such as Stable Diffusion 1.5 `runwayml/stable-diffusion-v1-5` or `stabilityai/sd-vae-ft-mse`) for dense binary mask segmentation:

```
                      Input Image x in R^(B, 3, H, W)
                                │
                 ┌──────────────▼───────────────┐
                 │     VAE Encoder (ResNet)     │
                 └──────────────┬───────────────┘
                                ▼
                 Posterior Distribution q(z|x)
                   - sample: z = mu + sigma * eps
                   - mode:   z = mu
                                │
                 ┌──────────────┴───────────────┐
                 │                              │
         ┌───────▼──────────────┐       ┌───────▼────────────────────────┐
         │ Auxiliary Classifier │       │ Latent Scaling (z * 0.18215)   │
         │ AdaptivePool -> MLP  │       └───────┬────────────────────────┘
         └───────┬──────────────┘               │
                 ▼                              ▼
        Class Logits (B, 3)     ┌────────────────────────────────────────┐
                                │          VAE Decoder (8x Up)           │
                                │  - ResNet UpBlocks + Attention         │
                                │  - Adapted ConvOut: 128 -> out_channels│
                                └───────────────┬────────────────────────┘
                                                ▼
                                   Binary Mask Logits (B, 1, H, W)
```

Key features:
- **Pretrained Generative Prior**: Uses the rich semantic image priors captured by diffusion model VAE encoders trained on hundreds of millions of images.
- **Direct Mask Projection**: Replaces the final RGB projection convolution (`conv_out: 128 -> 3`) with a dedicated binary mask projection layer (`128 -> 1`).
- **Flexible Fine-Tuning**: Supports end-to-end training (`freeze_encoder: false`) or freezing the encoder to train only the decoder (`freeze_encoder: true`).
- **Automatic Sizing & Normalization**: Automatically handles reflection padding if input dimensions are not divisible by 8, and maps inputs to the expected $[-1, 1]$ range.

---

### 6. Diffusion Multi-Noise Feature Decoder (Diffusion-Diff)

A state-of-the-art forensic architecture analyzing how images respond to generative diffusion denoising dynamics across multiple perturbation scales, enhanced with UNet-style multi-scale skip connections and a trainable VAE autoencoder:

```
                                        Input Real Image x (B, 3, H, W)
                                                       │
                           ┌───────────────────────────┴───────────────────────────┐
                           │            VAE Encoder (Trainable by Default)         │
                           │                 (autoencoder_trainable = True)        │
                           │  conv_in ──────────────┐                              │
                           │    │                   │ Skip 3: (B, C_3, H, W)       │
                           │  down_blocks[0] ─┐     │                              │
                           │    │             │     │ Skip 2: (B, C_2, H/2, W/2)   │
                           │  down_blocks[1]  │     │                              │
                           │    │             │     │ Skip 1: (B, C_1, H/4, W/4)   │
                           │  down_blocks[2]  │     │                              │
                           │    │             │     │                              │
                           │  mid_block       │     │                              │
                           │    │             │     │                              │
                           └────┼─────────────┼─────┼──────────────────────────────┘
                                ▼             │     │
                   Latent z0 (B, 4, H/8, W/8) │     │
                                │             │     │
        ┌───────────────────────┴─────────────┼─────┼───────────────────────┐
        │                                     │     │                       │
[Timestep t_1: e.g. 100]                      │     │           [Timestep t_K: e.g. 500]
        │                                     │     │                       │
 Add Noise: eps_1 ~ N(0, I)                   │     │            Add Noise: eps_K ~ N(0, I)
 z_t1 = sqrt(a_bar1)*z0 + s_1*eps_1           │     │            z_tK = sqrt(a_barK)*z0 + s_K*eps_K
        │                                     │     │                       │
┌───────┴───────────────┐                     │     │           ┌───────────┴───────────┐
│  Frozen Diffuser UNet │                     │     │           │  Frozen Diffuser UNet │
│(requires_grad = False)│                     │     │           │(requires_grad = False)│
└───────┬───────────────┘                     │     │           └───────────┬───────────┘
        ▼                                     │     │                       ▼
 Predicted Noise eps_hat_1                    │     │            Predicted Noise eps_hat_K
        │                                     │     │                       │
 Sinusoidal Embeddings:                       │     │            Sinusoidal Embeddings:
  - t_emb:   Sinusoidal(t_1)                  │     │             - t_emb:   Sinusoidal(t_K)
  - sig_emb: Sinusoidal(s_1)                  │     │             - sig_emb: Sinusoidal(s_K)
        │                                     │     │                       │
        └───────────────────────┬─────────────┼─────┼───────────────────────┘
                                ▼             │     │
        ┌─────────────────────────────────────┴─────┼───────────────────────┐
        │        Concatenate along Channel:         │                       │
        │ Z = [z0, z_t1, eps_1, eps_hat_1, ..., z_tK│                       │
        │           Shape: (B, C_Z, H/8, W/8)       │                       │
        └───────────────────────┬───────────────────┼───────────────────────┘
                                │                   │
                ┌───────────────┴──────────┐        │
                │                          │        │
        ┌───────▼──────────────┐   ┌───────▼────────┼──────────────────────────────────┐
        │ Auxiliary Classifier │   │        Trainable Latent Decoder (UNet-style)      │
        │ AdaptivePool -> MLP  │   │  Stage 1: H/8 -> H/4 ◄── Skip 1: (H/4, W/4)       │
        └───────┬──────────────┘   │  Stage 2: H/4 -> H/2 ◄── Skip 2: (H/2, W/2)       │
                ▼                  │  Stage 3: H/2 -> H   ◄── Skip 3: (H, W)           │
       Class Logits (B, 3)         │  Final 1x1 ConvOut                                │
                                   └────────────────┬──────────────────────────────────┘
                                                    ▼
                                       Binary Mask Logits (B, 1, H, W)
```

#### Mathematical & Architectural Principles

- **1. Latent Inversion & Multi-Scale Feature Extraction**: The input image $x$ is processed by the VAE encoder (trainable by default, `autoencoder_trainable = True`) to extract clean latent $z_0 \in \mathbb{R}^{B \times 4 \times H/8 \times W/8}$ and multi-scale intermediate skip features $\{s_{H/4}, s_{H/2}, s_H\}$.

- **2. Multi-Scale Forward Diffusion Perturbations**: For a configurable set of discrete timesteps $\{t_1, t_2, \dots, t_K\}$, standard Gaussian noise $\epsilon_k \sim \mathcal{N}(0, I)$ is added according to the diffusion schedule:

$$
z_{t_k} = \sqrt{\bar{\alpha}_{t_k}} z_0 + \sqrt{1 - \bar{\alpha}_{t_k}} \, \epsilon_k
$$

where $\sigma_k = \sqrt{1 - \bar{\alpha}_{t_k}}$ represents the noise standard deviation at timestep $t_k$.

- **3. Frozen Diffuser Noise Prediction**: The frozen pretrained UNet diffuser estimates the injected noise:

$$
\hat{\epsilon}_k = \mathrm{Diffuser}(z_{t_k}, t_k)
$$

Real versus synthetic image regions exhibit distinct noise reconstruction residuals $(\hat{\epsilon}_k - \epsilon_k)$, exposing subtle generative fingerprint anomalies.

- **4. Sinusoidal Condition Embeddings**: Continuous scalar timesteps $t_k$ and noise deviations $\sigma_k$ are projected into sinusoidal harmonic vector embeddings of dimensions $D_t$ and $D_\sigma$:

$$
\mathrm{emb}_{2i}(s) = \sin\left( s \cdot 10000^{-2i/D} \right), \qquad \mathrm{emb}_{2i+1}(s) = \cos\left( s \cdot 10000^{-2i/D} \right)
$$

and spatially broadcast across the latent grid to shape $(B, D, H/8, W/8)$.

- **5. High-Dimensional Latent Representation $Z$**: All components are concatenated along the channel axis:

$$
Z = \left[ z_0, \; z_{t_1}, \; \epsilon_1, \; \hat{\epsilon}_1, \; (\hat{\epsilon}_1 - \epsilon_1), \; e_{t_1}, \; e_{\sigma_1}, \; \dots, \; z_{t_K}, \; \epsilon_K, \; \hat{\epsilon}_K, \; (\hat{\epsilon}_K - \epsilon_K), \; e_{t_K}, \; e_{\sigma_K} \right]
$$

- **6. Multi-Scale Skip Connections**: To preserve sharp edge boundaries and prevent spatial resolution degradation during 8x latent downsampling, intermediate activation representations from the VAE encoder are routed directly to the decoder's upsampling stages via convolutional `SkipFusion` blocks at scales $H/4$, $H/2$, and $H$.

- **7. Trainable VAE Autoencoder & Decoder with Frozen Diffuser Prior**: The VAE autoencoder is **trainable by default** (`autoencoder_trainable: true`), enabling end-to-end feature adaptation while the multi-billion parameter Diffuser UNet remains **strictly frozen** (`requires_grad = False`) for compute and VRAM efficiency. The decoder is fully configurable from YAML configurations (channel dimensions, upsampling modes, normalization layers, and activations).

---

### 7. Diffusion Multi-Noise Latent Feature Decoder V2 (Diffusion-Diff-V2)

Diffusion-Diff-V2 advances the generative latent perturbation paradigm by decoupling generative representation decoding from encoder skips. By default, the **VAE encoder is frozen**, preventing degradation of pretrained latent manifold geometry. Rather than relying on contracting encoder skips, Diffusion-Diff-V2 introduces a **parallel pretrained, frozen VAE decoder** running side-by-side with the trainable decoder, injecting multi-scale generative decoding features through **perpendicular skip connections**:

```
                                      Input Real Image x (B, 3, H, W)
                                                     │
                          ┌──────────────────────────┴──────────────────────────┐
                          │                Frozen VAE Encoder                   │
                          │             (freeze_encoder = True)                 │
                          │  Encoder Skips Default: Disabled (no skip)          │
                          └──────────────────────────┬──────────────────────────┘
                                                     ▼
                                     Diffusion Latent z0 (B, 4, H/8, W/8)
                                                     │
               ┌─────────────────────────────────────┴─────────────────────────────────────┐
               │                                                                           │
               ▼                                                                           ▼
   [Diffusion Perturbation & UNet]                                       [Parallel Pretrained Frozen Decoder]
  For timesteps t_1, ..., t_K:                                                    (requires_grad = False)
   - Add noise eps_k ~ N(0, I)                                                             │
   - z_tk = sqrt(a_bar_k)*z0 + s_k*eps_k                                                   │
   - Frozen Diffuser UNet -> pred_eps_k                                     z_in = z0 / scaling_factor
   - Sinusoidal embeddings (t_k, sigma_k)                                                  │
               │                                                                           ▼
               ▼                                                              conv_in -> mid_block (H/8)
  High-Dimensional Representation Z                                                        │
   Z = [z0, z_t1, eps_1, eps_hat_1, ..., z_tK]                                             ▼
   Shape: (B, C_Z, H/8, W/8)                                              UpBlock 0 (H/4) ──┐
               │                                                                           │
               ├─────────────────────────┐                                                 │ (Perpendicular Skip 1)
               │                         │                                                 │
       ┌───────▼──────────────┐  ┌───────▼────────────────────────────────────────┐        │
       │ Auxiliary Classifier │  │        Trainable Latent Decoder V2             │        │
       │ AdaptivePool -> MLP  │  │                                                │        │
       └───────┬──────────────┘  │  Stage 0 (H/8 -> H/4) ◄─────────────────────────────────┘
               ▼                 │    feat = PerpSkipFusion_0(feat, Frozen_H4)    │
      Class Logits (B, 3)        │                                                │
                                 │                                                │
                                 │  Parallel Frozen UpBlock 1 (H/2) ──────────────┼────────┐
                                 │                                                │        │ (Perpendicular Skip 2)
                                 │  Stage 1 (H/4 -> H/2) ◄────────────────────────┴────────┘
                                 │    feat = PerpSkipFusion_1(feat, Frozen_H2)    │
                                 │                                                │
                                 │                                                │
                                 │  Parallel Frozen UpBlock 2 (H) ────────────────┼────────┐
                                 │                                                │        │ (Perpendicular Skip 3)
                                 │  Stage 2 (H/2 -> H)   ◄────────────────────────┴────────┘
                                 │    feat = PerpSkipFusion_2(feat, Frozen_H)     │
                                 │                                                │
                                 │  Final 1x1 ConvOut                             │
                                 └───────────────────────┬────────────────────────┘
                                                         ▼
                                          Binary Mask Logits (B, 1, H, W)
```

#### Mathematical & Architectural Principles

- **1. Frozen Pretrained Latent Space**: The input image $x$ is mapped to latent space using the frozen VAE encoder (`freeze_encoder = True`, `requires_grad = False`):

$$
z_0 = \mu(x) \cdot s
$$

where $s = 0.18215$ is the latent scaling factor. Freezing the encoder prevents catastrophic forgetting of the rich semantic generative manifolds trained on large-scale natural image distributions.

- **2. Elimination of Contracting Encoder Skips**: In standard UNet and Diffusion-Diff-V1, shallow spatial features from the contracting encoder bypass the latent bottleneck. In Diffusion-Diff-V2, encoder skips default to disabled (`use_encoder_skips = False`). This forces the model to rely strictly on generative perturbation discrepancies and pretrained decoding priors rather than low-level pixel color artifacts.

- **3. Parallel Pretrained Generative Decoder**: The diffusion latent $z_0$ is unscaled and fed to the parallel frozen VAE decoder:

$$
z_{\text{in}} = \frac{z_0}{s}, \qquad s_0 = \mathrm{MidBlock}\left(\mathrm{ConvIn}\left(z_{\text{in}}\right)\right)
$$

At each progressive upsampling layer $j \in \{0, 1, 2\}$, intermediate latent reconstructions are extracted:

$$
F_j^{\text{frozen}} = \mathrm{UpBlock}_j\left(s_j\right)
$$

These feature maps contain the generative model's intrinsic multi-scale synthesis representations of clean natural imagery.

- **4. Perpendicular Skip Connection Injection**: Rather than contracting from encoder to decoder (horizontal skip), the skip connection flows perpendicularly from the parallel frozen generative decoder into the trainable latent decoder:

$$
\text{Flow}: \quad z_0 \;\longrightarrow\; \text{Frozen Decoder} \;\longrightarrow\; F_j^{\text{frozen}} \;\xrightarrow{\text{Perpendicular Skip}}\; \text{Trainable Decoder Stage } j
$$

At each stage $j$, the trainable decoder fuses its intermediate feature map $h_j$ with the perpendicular skip $F_j^{\text{frozen}}$:

$$
h_j^{\text{fused}} = \sigma\left( \mathrm{Norm}\left( \mathrm{Conv}_{3\times3}\left( \left[ h_j, \; F_j^{\text{frozen}} \right] \right) \right) \right)
$$

- **5. Generative Perturbation & Diffuser Anomaly Cues**: Concurrently, $z_0$ undergoes multi-timestep forward diffusion perturbations $z_{t_k} = \sqrt{\bar{\alpha}_{t_k}} z_0 + \sqrt{1 - \bar{\alpha}_{t_k}} \epsilon_k$. The frozen diffuser UNet predicts $\hat{\epsilon}_k$, and residual anomalies $(\hat{\epsilon}_k - \epsilon_k)$ alongside harmonic schedule embeddings $(e_{t_k}, e_{\sigma_k})$ are concatenated to construct high-dimensional representation $Z$:

$$
Z = \left[ z_0, \; z_{t_1}, \; \epsilon_1, \; \hat{\epsilon}_1, \; (\hat{\epsilon}_1 - \epsilon_1), \; e_{t_1}, \; e_{\sigma_1}, \; \dots, \; z_{t_K}, \; \epsilon_K, \; \hat{\epsilon}_K, \; (\hat{\epsilon}_K - \epsilon_K), \; e_{t_K}, \; e_{\sigma_K} \right]
$$

- **6. Isolated Gradient Optimization**: Throughout training, the Diffuser UNet, VAE encoder, and parallel VAE decoder remain strictly frozen (`requires_grad = False`). Gradients backpropagate exclusively through the Trainable Decoder (including its perpendicular skip fusion layers) and the optional auxiliary classifier head.

- **7. Deterministic Evaluation Mode**: In `model.eval()`, random noise perturbations are seeded deterministically via `torch.Generator(device=z0.device).manual_seed(t_clamped + 42)` instead of unseeded `torch.randn_like`. This guarantees bitwise reproducible validation metrics and eliminates stochastic jitter during evaluation while preserving true stochastic diffusion perturbation during `train()`.

- **8. Memory Contiguity Guarantees**: Following spatial bilinear interpolation and resolution-matching cropping (`mask_logits[:, :, :orig_h, :orig_w]`), tensor representations are explicitly made contiguous via `.contiguous()`. This prevents downstream `RuntimeError` stride mismatches in loss functions, reshape operations, and GPU kernel fusions.

- **9. Disconnected Subgraph Elimination in V1**: In `DiffusionDiffModel`, when `autoencoder_trainable = True`, only `self.vae.encoder` is set to trainable. `self.vae.decoder` remains strictly frozen (`requires_grad = False`, `eval()`). Because output synthesis is performed by `TrainableLatentDecoder`, this prevents 74 unused decoder layers from becoming disconnected, zero-gradient parameters in the autograd computation graph.

## Mechanisms & Architectural Principles

### 1. Problem Formulation & Task Definition

Synthetic image forensics in SID-UNet addresses two complementary levels of visual inspection:
1. **Global Scene Categorization**: Classifying whether an entire image is natural / authentic ($y=0$), completely synthesized by a generative model ($y=1$), or authentic with localized synthetic inpainting or object splicing ($y=2$).
2. **Dense Pixel Localization**: Estimating a dense binary probability map $\hat{M} \in [0, 1]^{H \times W}$, where each spatial coordinate $(i, j)$ represents the posterior probability that pixel $(i, j)$ was artificially generated or modified:

$$
\hat{M}_{i,j} = P\bigl(\text{Pixel } (i,j) \text{ is synthetic} \mid I\bigr)
$$

---

### 2. Multi-Scale Feature Representation & Skip Connections

- **Hierarchical Contracting Encoder**: Successive downsampling layers contract spatial resolution while expanding feature channels. The contracting path extracts deep semantic descriptors and identifies global structural inconsistencies typical of generative models.
- **Multi-Scale Skip Connections**: Convolutional downsampling inevitably loses high-frequency spatial boundaries. Skip connections route high-resolution feature activations directly from contracting layers to expanding layers, providing local edge gradients and frequency traces essential for crisp tampering borders.
- **Progressive Upsampling Decoder**: Bilinear interpolation (or learned transposed convolutions) doubles spatial resolution at each step while reducing feature channels.
- **OutConv Layer**: A final $1 \times 1$ convolution projects the decoded representation to a 1-channel logit map:

$$
z = \mathrm{OutConv}(f_{\mathrm{dec}}) \in \mathbb{R}^{1 \times H \times W}, \qquad \hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

- **Sacrifice of Pixel Formulation**:
  When `sacrifice_of_pixel: true` is configured:

$$
f_{\mathrm{bot}} = \mathrm{Backbone}(x) \in \mathbb{R}^{C_{\mathrm{bot}} \times H_{\mathrm{bot}} \times W_{\mathrm{bot}}}
$$

$$
z_{\mathrm{low}} = \mathrm{Linear}(f_{\mathrm{bot}}) \in \mathbb{R}^{1 \times H_{\mathrm{bot}} \times W_{\mathrm{bot}}}
$$

$$
\hat{M}_{\mathrm{sac}} = \mathrm{Interpolate}\bigl(z_{\mathrm{low}}, \mathrm{size}=(H, W), \mathrm{mode}=\mathrm{bilinear}\bigr)
$$

---

### 3. Auxiliary Classifier & Multi-Task Semantic Regularization

Stand-alone pixel segmentation can overfit to local textures without understanding scene composition. To enforce semantic grounding:
- **Global Context Extraction**: At the bottleneck ($C_{\mathrm{bot}}$ channels), an `AdaptiveAvgPool2d((1, 1))` operation collapses spatial dimensions to produce a compact 1D latent vector $v \in \mathbb{R}^{C_{\mathrm{bot}}}$.
- **Multi-Layer Perceptron (MLP)**: The latent vector is routed through an auxiliary classifier head:

$$
v \xrightarrow{\mathrm{Linear}(C_{\mathrm{bot}}, 128)} h_1 \xrightarrow{\mathrm{ReLU}} h_2 \xrightarrow{\mathrm{Dropout}(p=0.2)} h_3 \xrightarrow{\mathrm{Linear}(128, 3)} \hat{y}_{\mathrm{cls}} \in \mathbb{R}^3
$$

- **Joint Multi-Task Optimization**:

$$
\mathcal{L}_{\mathrm{Total}} = \mathcal{L}_{\mathrm{Mask}}(\hat{M}, M_{\mathrm{gt}}) + \lambda_{\mathrm{aux}} \, \mathcal{L}_{\mathrm{CE}}(\hat{y}_{\mathrm{cls}}, y_{\mathrm{cls}})
$$

---

### 4. Streaming Dataset Engine & Dynamic Mask Synthesis

To train on massive multi-gigabyte or terabyte forensic datasets without saturating local storage:
- **Streaming Pipeline (`streaming: true`)**: Samples are streamed on-the-fly via Hugging Face `IterableDataset` with shuffle buffers and non-blocking worker prefetching.
- **Automatic Label & Mask Synthesis Logic**:
  - **Label 0 (Real / Authentic)**: Pure zero mask $\mathbf{0}_{H \times W}$.
  - **Label 1 (Fully Synthetic)**: Pure one mask $\mathbf{1}_{H \times W}$.
  - **Label 2 (Tampered / Inpainted)**: Ground truth mask binarized to $\{0.0, 1.0\}$.
- **Dynamic 2-Column Inferencing**: In 2-column image/mask datasets where explicit labels are omitted:

$$
\mathrm{Ratio} = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W M_{i,j}, \qquad \mathrm{Label} = \begin{cases} 
0 & \text{if } \mathrm{Ratio} = 0.0 \quad (\text{Real}) \\ 
1 & \text{if } \mathrm{Ratio} = 1.0 \quad (\text{Fully Synthetic}) \\ 
2 & \text{if } 0.0 < \mathrm{Ratio} < 1.0 \quad (\text{Tampered}) 
\end{cases}
$$

---

### 5. Mask Post-Processing Pipeline (Noise Suppression, Hole Filling, Morphology)

Raw model probability maps can suffer from isolated false positive speckles or small cavities. The post-processing module applies three consecutive algorithms:
1. **Connected Component Analysis & Small Area Suppression (`remove_small_components`)**:
   Computes the pixel area of every disjoint connected component $C_k$:

$$
\mathrm{Area}(C_k) = \sum_{(i,j) \in C_k} 1
$$

   Any component with $\mathrm{Area}(C_k) < 64$ pixels (parameter `min_area`) is suppressed to background ($0$).
2. **Topological Hole Filling (`fill_mask_holes`)**:
   Background cavities enclosed by positive foreground components with area $\le 256$ pixels (parameter `max_hole_size`) are filled with $1$s.
3. **Mathematical Morphological Smoothing (`apply_morphology`)**:
   Applies morphological opening ($\mathrm{Erode} \circ \mathrm{Dilate}$) followed by closing ($\mathrm{Dilate} \circ \mathrm{Erode}$) for boundary regularization.

---

### 6. SAM3 Spatial Join & Boundary Contrast Refinement

The hybrid integration (`sid_unet.models.sam3_refiner.SAMRefiner`) executes a **Spatial Join**:
1. Bounding boxes are derived from connected components of the model tampering mask.
2. The bounding boxes are supplied to SAM as prompt coordinates.
3. A segment is joined into the output tampering mask if its intersection exceeds the overlap threshold:

$$
\mathrm{IoU}\bigl(S_{\mathrm{SAM}}^k, M_{\mathrm{Model}}\bigr) \ge \tau_{\mathrm{join}}
$$

---

### 7. Simultaneous Multi-Stage Ablation Evaluation

The evaluation engine simultaneously tracks four independent variants in a single forward pass:
- **1. Baseline (Raw Model)**: Raw output logits $\to$ Sigmoid $\to$ Threshold.
- **2. + Post-Processing**: Component filter $\to$ Hole fill $\to$ Morphology.
- **3. + SAM Refinement**: Spatial join with SAM bounding box prompts.
- **4. + SAM & Post-Processing**: Full production refined pipeline.

---

### 8. Continuous Master Reports & Collision Skipping

Both training (`sid-train`) and evaluation (`sid-eval`, `sid-cross-eval`) natively support continuous master reports and collision avoidance:
- **Default Checkpoint Continuation**: When executing training or evaluation, the framework automatically searches for and continues from existing checkpoints (`checkpoint_latest.pt` or `checkpoint_best.pt`).
- **Collision Checking**: If a combination of model config (with checkpoint) and dataset config has already been evaluated:
  - The job detects the collision, logs a clear notification, and skips duplicate computation:
    ```
    ⚡ [COLLISION DETECTED - SKIPPED] Checkpoint 'checkpoint_best.pt' with Config 'casia_v2.0.yaml' (Dataset: 'CASIA_v2.0') has already been evaluated. Skipping...
    ```
  - Reuses the existing report metrics and merges them seamlessly into the consolidated master reports (`master_cross_evaluation_report.json`, `multi_experiment_comparison.json`, `multi_checkpoint_evaluation.json`).
- Can be overridden with `--force` or `--no-skip-collision` if re-computation is explicitly desired.

---

### 9. Automated Visual Illustration (`sid-illu`) & Heatmap Generation

- **Qualitative Comparison Grids (`sid-illu`)**:
  Generates comparative sample figures side-by-side:
  `[Input Image]` | `[Ground Truth]` | `[Model 1 Mask]` | `[Model 1 Overlay]` | `[Model 1 Error Map]` | `[Model 2 Mask]` | `[Model 2 Overlay]` | `[Model 2 Error Map]`
- **Cross-Evaluation 2D Heatmaps**:
  Visualizes generalization performance across all checkpoints and cross-evaluation datasets (`cross_eval_*_heatmap.png`).

---

### 10. Memory Management & OOM Dynamic Auto-Recovery

- **Pre-Execution VRAM Probing**: Automatically searches safe batch sizes and scales gradient accumulation.
- **Activation Checkpointing**: Recomputes forward activations during backpropagation, saving 60-70% activation VRAM.
- **Dynamic OOM Fallback Catching**: Recursively bisects batches into micro-batches on CUDA OOM without crashing.

---

### 11. Runtime Learnability Diagnostics & Debug Mode (nn-toolbox)

Before committing hours or days to a full training run, SID-UNet provides an automated **runtime learnability diagnostic laboratory** powered by the **`nn-toolbox`** package.

Rather than mere metric logging or basic syntax/shape validation (which fails automatically at runtime), this diagnostic evaluates **fundamental learning viability**:
1. **Signal Propagation Viability**: Measures activation standard deviations across every encoder, bottleneck, and decoder block to ensure forward signals neither explode ($>20\times$ amplification) nor vanish ($<0.05\times$ attenuation).
2. **Backward Gradient Flow & Reachability**: Verifies that 100% of trainable parameters receive active gradients, detecting detached subgraphs, broken autograd chains, and vanishing gradients.
3. **True Parameter Update Dynamics**: Computes the actual displacement-to-weight ratio $||\Delta \theta|| / ||\theta||$ after an optimizer step. This separates raw gradient magnitude from actual parameter progress, flagging dead learning rates, frozen weights, or explosive parameter divergence.
4. **Structural Graph Connectivity & Disconnection Detection**: Pinpoints trainable parameters and submodules that are marked `requires_grad=True` but receive 0 gradients due to detached autograd pathways or unused submodules.
5. **Inter-Branch Gradient Balance Tracking**: Evaluates scale-invariant RMS gradient distributions across distinct sub-networks (e.g., encoder vs decoder vs auxiliary heads) to detect branch starvation ($>500\times$ disparity).
6. **Tensor Memory Layout & Contiguity Checks**: Verifies that intermediate and output representations maintain contiguous strides, preventing `.view()` crashes and CUDA kernel latency.
7. **Verified Healthy Confirmations**: Clearly and concisely reports every analyzed dimension that is operating stably, giving the practitioner immediate confidence that signal pathways and update dynamics are healthy.
8. **Prioritized Actionable Hypotheses**: When anomalies are detected, the tool highlights the specific layer/parameter target, formulates cautious causal hypotheses (e.g. *saturated activation functions*, *excessive regularization*, *detached tensor logic*), and suggests targeted remediation steps.
9. **Active Diagnostic Experiments (`--debug-mode deep`)**:
   - **Tiny-Batch Memorization Capacity (`overfit_test`)**: Tests whether the architecture and optimizer can memorize $N=1, 8$ samples, distinguishing optimization/loss capability from dataset capacity constraints.
   - **Logarithmic Learning Rate Sweep (`lr_sweep`)**: Probes loss response across $10^{-6}$ to $10^{-1}$ to identify the stable learning regime versus divergence or stagnation.
   - **Train/Eval Mode Consistency (`train_eval_test`)**: Isolates differences between `model.train()` and `model.eval()` to detect improper normalization state shifts or stochastic bugs.
   - **Evaluation Determinism & Drift (`eval_determinism_test`)**: Validates bitwise reproducible inferences across repeated forward passes under `model.eval()`.

#### Running Diagnostics via CLI:
```bash
# Light mode: Non-destructive telemetry (signal propagation, gradient flow, update ratios)
sid-train --config configs/default.yaml --debug

# Deep mode: Includes tiny-batch memorization experiments and learning rate sweeps
sid-train --config configs/default.yaml --debug --debug-mode deep
```

#### Diagnostic Artifacts Generated:
- **Interactive HTML Report**: `reports/diagnostics/diagnostic_report.html` (visual status cards, verified healthy checks, actionable issues table, and layer metric progression).
- **Structured JSON Report**: `reports/diagnostics/diagnostic_report.json` (machine-readable telemetry, metrics, and investigation targets).

#### Example Console Output:
```text
[INFO] 🔬 [DEBUG MODE] Initializing nn-toolbox diagnostic laboratory (Mode: LIGHT)...
[INFO] 🔬 [DEBUG MODE] Diagnostic report saved to: reports/diagnostics/diagnostic_report.json and .html
[INFO] ✅ [DEBUG MODE] Verified healthy learnability dimensions (4):
[INFO]    ✓ [HEALTHY] FORWARD: Forward activation propagation is stable across 16 layer(s) (std range: 0.56 - 1.00).
[INFO]    ✓ [HEALTHY] BACKWARD: Active gradient flow verified on 100% of trainable parameters (48/48, mean norm: 3.42e-02).
[INFO]    ✓ [HEALTHY] OPTIMIZATION: Healthy parameter update ratio: ||Δθ||/||θ|| = 1.81e-02 (displacement norm: 4.59e-01).
[INFO]    ✓ [HEALTHY] DATA: Input data is finite and non-constant (variance: 1.00e+00 range: [-4.37, 4.07]).
[INFO] 🎉 [DEBUG MODE] All learnability diagnostic checks passed cleanly with zero warnings.
```

---

## Installation

Install in editable mode using `pip` or `uv`:

```bash
# Clone and enter directory
cd /workspace

# Install package and all CLI commands (sid-train, sid-eval, sid-cross-eval, sid-predict, sid-illu, sid-check-8bit)
pip install -e .

# Or with debug diagnostics dependencies (nn-toolbox):
pip install -e ".[debug]"

# Or with 8-bit training dependencies (bitsandbytes):
pip install -e ".[8bit]"

# Or with all optional features:
pip install -e ".[all]"

# Run automated 8-bit hardware and library compatibility check:
sid-check-8bit
# or via training CLI:
python -m sid_unet.train --check-8bit

# Or using uv (compatible with torch>=2.5.0, preserving your existing PyTorch installation):
uv pip install -e .

# With development and testing dependencies:
pip install -e ".[dev]"
# or:
uv pip install -e ".[dev]"
```

---

## Project Structure

```
├── configs/
│   ├── default.yaml
│   ├── train_streaming.yaml
│   ├── train_non_streaming.yaml
│   ├── evaluate.yaml
│   ├── test_smoke.yaml
│   ├── test_quick.yaml
│   ├── cross-eval/                   # Benchmark evaluation dataset configs
│   │   ├── casia_v2.0.yaml
│   │   ├── cocoglide.yaml
│   │   ├── diffseg30k.yaml
│   │   └── open-sdid.yaml
│   └── experiments/
│       ├── sam3-qlora/               # SAM3 + QLoRA (4-bit NF4) configs on BeyondTheBrush
│       │   ├── sam3_qlora_beyondthebrush_b1.yaml
│       │   ├── sam3_qlora_beyondthebrush_r16.yaml
│       │   ├── sam3_qlora_beyondthebrush_focal.yaml
│       │   └── sam3_qlora_beyondthebrush_dice.yaml
│       ├── efficientnet/             # Pretrained EfficientNet experiment configs
│       │   ├── efficientnet_b0_unet.yaml
│       │   ├── efficientnet_b0_sacrifice_of_pixel.yaml
│       │   ├── efficientnet_b0_sacrifice_of_pixel_b32.yaml
│       │   └── efficientnet_b2_unet.yaml
│       └── unet_scratch/             # UNet scratch variants
│           ├── unet_wide_b32.yaml
│           ├── unet_deep_5stage_b32.yaml
│           ├── unet_focal_hard_mining_b32.yaml
│           ├── unet_heavy_wide_deep_b32.yaml
│           ├── unet_highres_512_b16.yaml
│           ├── unet_large_batch_b64.yaml
│           ├── unet_convtranspose_learned_up_b32.yaml
│           └── unet_non_streaming_b32.yaml
├── sid_unet/
│   ├── dataset/
│   ├── models/
│   │   ├── blocks.py
│   │   ├── unet.py                   # UNet architecture
│   │   ├── efficientnet.py           # EfficientNet UNet & Sacrifice of Pixel
│   │   ├── sam3_qlora.py             # SAM3 foundation model with 4-bit QLoRA
│   │   └── sam3_refiner.py           # SAM3 spatial join refinement
│   ├── losses/
│   ├── metrics/
│   ├── training/
│   ├── utils/
│   ├── train.py                      # CLI: sid-train
│   ├── evaluate.py                   # CLI: sid-eval
│   ├── cross_eval.py                 # CLI: sid-cross-eval
│   ├── predict.py                    # CLI: sid-predict
│   └── illustration.py               # CLI: sid-illu
├── tests/
├── pyproject.toml
└── README.md
```

---

## Configuration System

Configurations are organized into modular, human-readable YAML files located in `configs/`. Any config setting can also be dynamically overridden from the CLI using `--override key.nested=value`.

### 1. Configuration Field Reference

The configuration file is divided into modular top-level sections:

#### `project`
| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `"sid_unet_baseline"` | Experiment run identifier used to name output directories. |
| `seed` | `int` | `42` | Random seed for deterministic reproducibility across PyTorch, NumPy, and random. |
| `device` | `str` | `"auto"` | Execution device: `"auto"` (selects CUDA if available), `"cuda"`, or `"cpu"`. |
| `output_dir` | `str` | `"outputs"` | Base output directory where runs, checkpoints, logs, and reports are saved. |

#### `data`
| Field | Type | Default | Description |
|---|---|---|---|
| `dataset_name` | `str` | `"KhangTruong/IMD2020"` | Hugging Face dataset identifier or local path. |
| `streaming` | `bool` | `false` | When `true`, streams samples on-the-fly without downloading entire datasets to disk. |
| `image_size` | `[H, W]` | `[256, 256]` | Target input image resolution `[height, width]` passed to model. |
| `batch_size` | `int` | `16` | Micro-batch size per forward/backward pass. |
| `num_workers` | `int` | `2` | DataLoader worker processes for multi-process sample prefetching. |
| `pin_memory` | `bool` | `true` | Pins CPU memory pages to accelerate host-to-GPU data transfers. |
| `shuffle_buffer_size` | `int` | `1000` | Sample buffer size for pseudo-random shuffling when `streaming: true`. |
| `train_split` | `str` | `"train"` | Dataset split name for training. |
| `val_split` | `str` | `"validation"` | Dataset split name for validation. |
| `test_split` | `str` | `"test"` | Dataset split name for evaluation/benchmarking. |
| `train_samples_per_epoch` | `int` | `-1` | Cap on training samples per epoch (`-1` to consume until dataset is depleted). |
| `val_samples_per_epoch` | `int` | `-1` | Cap on validation samples per epoch (`-1` or negative for full validation set). |
| `val_samples` | `int` | `-1` | Alias for `val_samples_per_epoch`. |
| `test_samples` | `int` | `-1` | Cap on test samples during evaluation (`-1` or negative for full test set). |
| `augmentations` | `dict` | - | Data augmentation probabilities (`horizontal_flip`, `vertical_flip`, `random_rotate90`). |

#### `model`
| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `"unet"` | Model architecture family: `"unet"`, `"efficientnet"`, or `"sam3_qlora"`. |
| `in_channels` | `int` | `3` | Input image channels (RGB = 3). |
| `out_channels` | `int` | `1` | Output mask channels (1 for binary foreground mask logits). |
| `features` | `list[int]` | `[64, 128, 256, 512]` | Feature channel dimensions for standard UNet contracting/expanding stages. |
| `bilinear` | `bool` | `true` | Upsampling method for UNet: `true` for bilinear interpolation, `false` for transposed conv. |
| `dropout` | `float` | `0.1` | Dropout rate applied before upsampling/bottleneck layers. |
| `aux_classifier` | `bool` | `false` | Enables auxiliary 3-class classification head (`Real`, `Fully AI`, `Tampered`). |
| `num_classes` | `int` | `3` | Number of classes for the auxiliary classification head. |
| `backbone` | `str` | `"efficientnet_b0"` | Pretrained CNN backbone for EfficientNet (`"efficientnet_b0"` to `"b7"`). |
| `pretrained` | `bool` | `true` | Loads ImageNet pretrained weights for EfficientNet backbones. |
| `sacrifice_of_pixel` | `bool` | `false` | When `true`, uses only bottleneck features routed to a linear layer and zoomed directly. |
| `pretrained_model_name_or_path` | `str` | `"jetjodh/sam3"` | Base Hugging Face model repository or local path for SAM3 foundation models. |
| `load_in_4bit` | `bool` | `true` | Applies 4-bit NormalFloat (NF4) quantization via bitsandbytes for SAM3. |
| `load_in_8bit` | `bool` | `false` | Applies 8-bit quantization via bitsandbytes for SAM3. |
| `lora_r` | `int` | `8` | Low-Rank Adaptation (LoRA) rank dimension for PEFT parameter adapters. |
| `lora_alpha` | `int` | `16` | LoRA alpha scaling hyperparameter. |
| `lora_dropout` | `float` | `0.05` | Dropout probability for LoRA adapter layers. |
| `lora_target_modules` | `list[str]` | `["q_proj", "v_proj"]`| Target attention projection layers to attach LoRA adapters to. |
| `prompt_text` | `str` | `"tampered region"` | Natural language conditioning prompt guiding SAM3 attention queries. |
| `target_size` | `[H, W]` | `[1008, 1008]` | Native ViT patch grid resolution for SAM3 input processing. |

#### `loss`
| Field | Type | Default | Description |
|---|---|---|---|
| `mask_loss_type` | `str` | `"combined"` | Mask loss formulation: `"combined"` (BCE + Dice), `"bce"`, `"dice"`, or `"focal"`. |
| `bce_weight` | `float` | `0.5` | Weight $\alpha$ for Binary Cross-Entropy loss in combined mode. |
| `dice_weight` | `float` | `0.5` | Weight $\beta$ for Soft Dice loss in combined mode. |
| `focal_gamma` | `float` | `2.0` | Focusing parameter $\gamma$ for Focal loss (higher values down-weight easy pixels). |
| `focal_alpha` | `float` | `0.25` | Balance factor $\alpha$ for Focal loss class weighting. |
| `aux_loss_type` | `str` | `"cross_entropy"`| Loss function for auxiliary classifier head. |
| `aux_weight` | `float` | `0.2` | Multi-task loss weight $\lambda_{\mathrm{aux}}$ balancing classification against segmentation. |

#### `training` (Optimization, Checkpointing & Timing)
| Field | Type | Default | Description |
|---|---|---|---|
| `epochs` | `int` | `10` | Maximum number of training epochs. |
| `learning_rate` | `float` | `0.001` | Initial base learning rate for optimizer. |
| `weight_decay` | `float` | `0.0001` | L2 weight regularization penalty. |
| `optimizer` | `str` | `"adamw"` | Optimization algorithm (`"adamw"`, `"adam"`, `"sgd"`, `"adamw8bit"`, `"paged_adamw8bit"`). |
| `use_8bit_optimizer` | `bool` | `false` | Enables 8-bit AdamW optimizer via `bitsandbytes` (reduces optimizer memory by 75%). |
| `check_8bit_compatibility` | `bool` | `true` | Automatically verifies CUDA, GPU compute capability, and `bitsandbytes` compatibility at training time. |
| `fallback_to_16bit` | `bool` | `true` | Automatically falls back to **GPU 16-bit mode (AMP FP16/BF16)** if 8-bit mode is unsupported or fails. |
| `fallback_on_unsupported_8bit` | `bool` | `true` | Alias for `fallback_to_16bit`. |
| `scheduler` | `str` | `"cosine"` | Learning rate schedule (`"cosine"`, `"step"`, `"plateau"`, `"none"`). |
| `warmup_epochs` | `int` | `1` | Number of epochs for linear learning rate warmup. |
| `min_lr` | `float` | `1e-6` | Minimum learning rate floor reached at end of cosine decay. |
| `grad_clip_norm` | `float` | `1.0` | Maximum gradient Euclidean norm threshold to prevent exploding gradients. |
| `amp` | `bool` | `true` | Automatic Mixed Precision (`torch.cuda.amp`) using FP16/BF16 for 2x speedup. |
| `gradient_accumulation_steps` | `int` | `1` | Number of forward passes to accumulate gradients over before optimizer step. |
| `auto_batch_size` | `bool` | `true` | Probes available GPU memory at startup to automatically adjust batch size. |
| `empty_cache_per_epoch` | `bool` | `true` | Flushes `torch.cuda.empty_cache()` at every epoch boundary to prevent memory fragmentation. |
| **`checkpoint_period`** | `int` / `str` | `3600` | **Time-based periodic checkpointing cadence** (see detailed breakdown below). |
| **`checkpoint_steps`** | `int` | `null` | **Step-based periodic checkpointing cadence** (saves every $N$ training steps). |
| **`save_best`** | `bool` | `true` | Saves `checkpoint_best.pt` whenever the validation metric achieves a new optimum. |
| **`save_latest`** | `bool` | `true` | Saves `checkpoint_latest.pt` after every epoch and periodic save trigger. |
| `eval_interval` | `int` | `1` | Frequency in epochs at which validation evaluation is executed. |
| `early_stopping_patience` | `int` | `5` | Epochs without validation metric improvement before halting training early. |
| `early_stopping_metric` | `str` | `"val_iou"` | Target validation metric to monitor (`"val_iou"`, `"val_dice"`, `"val_loss"`). |
| `early_stopping_mode` | `str` | `"max"` | Optimization direction: `"max"` (for IoU/Dice) or `"min"` (for Loss). |
| `debug_mode` | `str` / `bool` | `false` | Automated learnability diagnostic mode via `nn-toolbox` (`"light"`, `"deep"`, or `false`). |

#### `logging`
| Field | Type | Default | Description |
|---|---|---|---|
| `log_interval` | `int` | `20` | Step interval for printing batch loss, memory usage, and throughput to stdout. |
| `log_memory` | `bool` | `true` | Logs allocated and reserved GPU VRAM in megabytes during training. |

#### `post_processing`
| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `true` | Enables post-processing on output binary masks during inference and evaluation. |
| `min_area` | `int` | `64` | Minimum pixel area for connected components; smaller regions are removed as noise. |
| `fill_holes` | `bool` | `true` | Whether to fill enclosed background cavities inside detected tampering regions. |
| `max_hole_size` | `int` | `256` | Maximum pixel area of an enclosed hole to fill. |
| `morphology` | `str` | `"open_close"` | Morphological smoothing algorithm (`"open_close"`, `"open"`, `"close"`). |
| `morph_kernel_size` | `int` | `3` | Structuring element kernel size for morphological operations. |

---

### 2. Checkpoint Timing & Resumption Parameters

Periodic checkpointing ensures that long-running jobs (especially large-scale streaming runs that take hours or days per epoch) save restorable training states at predictable intervals without waiting for full epoch completion:

#### A. Periodic Checkpoint Intervals (`checkpoint_period` & `checkpoint_steps`)
The trainer parses `checkpoint_period` using [`parse_checkpoint_period`](sid_unet/training/trainer.py), and evaluates periodic saving on **every training batch step**:

```yaml
training:
  # Time-based cadence (seconds, duration strings, or hours):
  checkpoint_period: 3600       # 3600 seconds = 1 hour
  # or:
  checkpoint_period: "30m"      # "30m", "1h", "7200s"
  # or:
  checkpoint_period_hours: 1.5  # 90 minutes

  # Step-based cadence (number of global batches/optimizer steps):
  checkpoint_steps: 100         # Save every 100 training steps (CLI: --checkpoint-steps 100)

  # Continuous latest checkpoint maintenance:
  save_latest: true             # Always keeps checkpoint_latest.pt synchronized (default: true)
```

#### B. Checkpoint Files & State Persistence
During training, up to three checkpoints are managed in `outputs/RUN/<stem>/checkpoints/`:
- **`checkpoint_periodic.pt`**: Written whenever elapsed wall-clock time $\ge$ checkpoint\_period or step interval is reached. Contains model weights, optimizer state, LR scheduler state, `GradScaler` state, `epoch`, `global_step`, metrics, and history.
- **`checkpoint_latest.pt`**: Continuously synchronized on periodic saves and written at every epoch boundary. Prioritized first for auto-resumption.
- **`checkpoint_best.pt`**: Updated whenever validation metric improves on `eval_interval` epochs (governed by `early_stopping_metric` and `save_best: true`).

#### C. Cross-Precision Resume Synchronization
- **Resume Format Synchronization**: When resuming a checkpoint from another environment (e.g. 4-bit CUDA NF4 loaded onto CPU float32), `resume_from_checkpoint` automatically synchronizes `checkpoint_latest.pt` in the current machine's native representation, eliminating parameter shape mismatch notices on subsequent runs.

---

### 3. Architecture-Specific Config Examples

#### EfficientNet UNet Config Example:
```yaml
model:
  name: "efficientnet"
  backbone: "efficientnet_b0"
  pretrained: true
  sacrifice_of_pixel: false          # Multi-scale feature skip connections into UNet decoder
  in_channels: 3
  out_channels: 1
  aux_classifier: false
  num_classes: 3
  dropout: 0.1
```

### EfficientNet 'Sacrifice of Pixel' Config Example:
```yaml
model:
  name: "efficientnet"
  backbone: "efficientnet_b0"
  pretrained: true
  sacrifice_of_pixel: true           # Only final bottleneck feature map -> Linear -> Zoom out to pixel size
  in_channels: 3
  out_channels: 1
  aux_classifier: false
  num_classes: 3
  dropout: 0.1
```

### SAM3 + QLoRA Config Example:
```yaml
model:
  name: "sam3_qlora"
  pretrained_model_name_or_path: "jetjodh/sam3"  # Or "facebook/sam3"
  load_in_4bit: true                             # 4-bit NormalFloat quantization via bitsandbytes
  lora_r: 8                                      # LoRA rank dimension
  lora_alpha: 16                                 # LoRA scaling factor
  lora_dropout: 0.05
  lora_target_modules: ["q_proj", "v_proj"]      # Target attention projections
  prompt_text: "tampered region"                 # Conditioning text prompt
  aux_classifier: false
  target_size: [1008, 1008]

data:
  dataset_name: "KhangTruong/BeyondTheBrush"
  streaming: true                                # Streamed dataset loading
  batch_size: 1                                  # Micro-batch 1 to fit inside 12GB VRAM

training:
  gradient_accumulation_steps: 16                # Effective batch size 16
  learning_rate: 0.0002
  amp: false
```

#### 4. Finetuned Diffusion VAE Configuration Example (`configs/experiments/sd_vae_finetune/default.yaml`)
```yaml
model:
  name: "vae_finetune"
  pretrained_model_name_or_path: "runwayml/stable-diffusion-v1-5"
  subfolder: "vae"
  in_channels: 3
  out_channels: 1
  freeze_encoder: false                          # End-to-end VAE fine-tuning
  sample_mode: "sample"                          # 'sample' during train, 'mode' during eval
  scaling_factor: 0.18215
  aux_classifier: true
  num_classes: 3

training:
  learning_rate: 0.0001
  amp: true
```

#### 5. Diffusion Multi-Noise Feature Decoder (`diffusion_diff`) Configuration Example (`configs/experiments/diffusion_diff/default.yaml`)
```yaml
model:
  name: "diffusion_diff"
  pretrained_model_name_or_path: "runwayml/stable-diffusion-v1-5"
  autoencoder_trainable: true                    # VAE autoencoder fine-tuning (trainable by default)
  use_skip_connections: true                    # UNet-style multi-scale encoder skip connections
  timesteps: [100, 250, 500]                     # Perturbation timesteps
  timestep_embed_dim: 32                         # Sinusoidal timestep embedding size
  sigma_embed_dim: 32                            # Sinusoidal noise deviation embedding size
  include_noisy_latents: true
  include_added_noise: true
  include_predicted_noise: true
  include_noise_diff: true
  include_z0: true
  # Configurable Trainable Decoder
  decoder:
    channels: [256, 128, 64, 32]
    upsample_mode: "bilinear"
    norm_layer: "batchnorm"
    activation: "silu"
    dropout: 0.1
    num_res_blocks: 1
  aux_classifier: true
  num_classes: 3

training:
  learning_rate: 0.0003
  amp: true
```

#### 6. Diffusion Multi-Noise Latent Feature Decoder V2 (`diffusion_diff_v2`) Configuration Example (`configs/experiments/diffusion_diff_v2/default.yaml`)
```yaml
model:
  name: "diffusion_diff_v2"
  pretrained_model_name_or_path: "runwayml/stable-diffusion-v1-5"
  freeze_encoder: true                           # VAE encoder frozen by default
  use_encoder_skips: false                       # Encoder-to-trainable-decoder skips disabled by default
  use_perpendicular_skips: true                  # Parallel frozen decoder perpendicular skips enabled
  timesteps: [100, 250, 500]                     # Perturbation timesteps
  timestep_embed_dim: 32                         # Sinusoidal timestep embedding size
  sigma_embed_dim: 32                            # Sinusoidal noise deviation embedding size
  include_noisy_latents: true
  include_added_noise: true
  include_predicted_noise: true
  include_noise_diff: true
  include_z0: true
  # Configurable Trainable Decoder
  decoder:
    channels: [256, 128, 64, 32]
    upsample_mode: "bilinear"
    norm_layer: "batchnorm"
    activation: "silu"
    dropout: 0.1
    num_res_blocks: 1
  aux_classifier: true
  num_classes: 3

training:
  learning_rate: 0.0003
  amp: true
```

---

## Quickstart: How to Run

### 1. Training

#### A. Single Experiment Training
```bash
# Train SAM3 + QLoRA on BeyondTheBrush (streaming mode)
sid-train --config configs/experiments/sam3-qlora/sam3_qlora_beyondthebrush_b1.yaml

# Train standard UNet
sid-train --config configs/default.yaml

# Train Pretrained EfficientNet-B0 with UNet multi-scale skip decoder
sid-train --config configs/experiments/efficientnet/efficientnet_b0_unet.yaml

# Train EfficientNet with Sacrifice of Pixel mode
sid-train --config configs/experiments/efficientnet/efficientnet_b0_sacrifice_of_pixel.yaml

# Train Finetuned Diffusion VAE
sid-train --config configs/experiments/sd_vae_finetune/default.yaml

# Train Diffusion-Diff (Multi-Noise Latent Feature Decoder)
sid-train --config configs/experiments/diffusion_diff/default.yaml

# Train Diffusion-Diff-V2 (Parallel Frozen Decoder & Perpendicular Skips)
sid-train --config configs/experiments/diffusion_diff_v2/default.yaml
```

#### B. Multi-Experiment Suite (Continuous Reporting & Collision Skipping)
```bash
# Run multiple experiments sequentially; already-evaluated combinations are automatically skipped
sid-train --configs \
  configs/experiments/unet_scratch/unet_wide_b32.yaml \
  configs/experiments/efficientnet/efficientnet_b0_unet.yaml \
  configs/experiments/efficientnet/efficientnet_b0_sacrifice_of_pixel.yaml
```

All runs are organized inside a unified `'RUN'` folder named by config (without arbitrary numbering):
```
outputs/RUN/
├── unet_wide_b32/
│   ├── checkpoints/
│   │   ├── checkpoint_best.pt
│   │   └── checkpoint_latest.pt
│   ├── logs/
│   │   └── train_run.log
│   ├── eval_reports/
│   └── effective_config.yaml
├── efficientnet_b0_unet/
│   └── checkpoints/
├── multi_experiment_comparison.md
└── multi_experiment_comparison.json
```

#### C. Automatic Resumption & Hugging Face Hub Resumption
By default, `sid-train` automatically discovers existing checkpoints in the repository and output directories (`outputs/RUN/<stem>/checkpoints/`, `outputs/checkpoints/`, etc.), displaying an on-screen notification before seamlessly resuming training:

```bash
# 1. Automatic checkpoint discovery (enabled by default)
# Finds checkpoint_latest.pt -> checkpoint_periodic.pt -> checkpoint_best.pt and resumes
sid-train --config configs/default.yaml

# 2. Direct Hugging Face Model Repository resumption
# Downloads checkpoint weights from Hugging Face Hub and resumes training
sid-train --config configs/default.yaml --resume-repo KhangTruong/sid-unet
# Or using the hf:// URI scheme:
sid-train --config configs/default.yaml --resume hf://KhangTruong/sid-unet

# 3. Explicit local checkpoint file resumption
sid-train --config configs/default.yaml --resume outputs/RUN/default/checkpoints/checkpoint_latest.pt

# 4. Disable auto-resume to start a fresh training run from epoch 1
sid-train --config configs/default.yaml --no-auto-resume
```

#### D. Runtime Learnability Debugging (`--debug`, `--debug-mode`)
Execute targeted diagnostic experiments before training to verify signal propagation, active gradient flow, update-to-weight displacement ratios, and memorization capacity:

```bash
# Fast non-destructive telemetry (forward signal scale, backward gradient flow, ||Δθ||/||θ||):
sid-train --config configs/default.yaml --debug

# Deep active laboratory (includes tiny-batch memorization & learning rate sweep):
sid-train --config configs/default.yaml --debug --debug-mode deep
```

Reports are automatically generated and displayed in console, with permanent visual artifacts saved to `reports/diagnostics/diagnostic_report.html` and `reports/diagnostics/diagnostic_report.json`.

---

### 2. Evaluation & Benchmarking

#### A. Single Checkpoint Evaluation
```bash
# Evaluate local checkpoint
sid-eval --checkpoint outputs/RUN/unet_wide_b32/checkpoints/checkpoint_best.pt --split test

# Evaluate directly from Hugging Face Hub repository
sid-eval --checkpoint hf://KhangTruong/sid-unet --split test
```

#### B. Multi-Checkpoint Evaluation
```bash
sid-eval --checkpoints \
  outputs/RUN/unet_wide_b32/checkpoints/checkpoint_best.pt \
  outputs/RUN/efficientnet_b0_unet/checkpoints/checkpoint_best.pt \
  hf://KhangTruong/sid-unet \
  --split test
```

#### C. Cross-Evaluation Matrix Benchmarking (`sid-cross-eval`)
```bash
# Cross-evaluate checkpoints across multiple dataset configurations with collision skipping
sid-cross-eval \
  --cross-configs configs/cross-eval/*.yaml \
  --checkpoints "outputs/RUN/*/checkpoints/checkpoint_best.pt" \
  --split test
```

---

### 3. Random Sample Visual Illustration (`sid-illu`)

Sample random examples from datasets and illustrate predictions side-by-side across multiple model checkpoints:

```bash
# Compare multiple checkpoints on multiple dataset configurations
sid-illu \
  --model-ckpts \
    outputs/RUN/unet_wide_b32/checkpoints/checkpoint_best.pt \
    outputs/RUN/efficientnet_b0_unet/checkpoints/checkpoint_best.pt \
  --dataset-configs configs/cross-eval/diffseg30k.yaml configs/cross-eval/casia_v2.0.yaml \
  --num-samples 5 \
  --output-dir outputs/illustrations
```

Outputs:
- **`illustration_<config_name>.png`**: Side-by-side comparison grids displaying Original Image, Ground Truth, each Model's Mask Prediction, Overlay, and Color-Coded Error Map (Green: True Positive, Red: False Positive, Blue: False Negative).
- **`illustration_report.md`**: Consolidated visual Markdown report containing sample metrics (IoU, Dice / F1) for each model and sample.

---

### 4. Inference & Mask Prediction

```bash
sid-predict \
  --checkpoint outputs/RUN/unet_wide_b32/checkpoints/checkpoint_best.pt \
  --image /path/to/test_image.jpg \
  --output_dir predictions \
  --save_overlay
```

---

## Loss Functions & Metrics

### Loss Functions

- **Binary Cross-Entropy Loss with Logits**:

$$
\mathcal{L}_{\mathrm{BCE}}(x, y) = - \frac{1}{N} \sum_{i=1}^N \Bigl[ y_i \log \sigma(x_i) + (1 - y_i) \log \bigl(1 - \sigma(x_i)\bigr) \Bigr]
$$

- **Soft Dice Loss**:

$$
\mathcal{L}_{\mathrm{Dice}}(p, y) = 1 - \frac{2 \sum_{i=1}^N p_i y_i + \epsilon}{\sum_{i=1}^N p_i + \sum_{i=1}^N y_i + \epsilon}
$$

- **Binary Focal Loss**:

$$
\mathcal{L}_{\mathrm{Focal}}(p_t) = - \alpha_t (1 - p_t)^\gamma \log(p_t)
$$

- **Total Multi-Task Loss**:

$$
\mathcal{L}_{\mathrm{Total}} = \alpha \, \mathcal{L}_{\mathrm{BCE}} + \beta \, \mathcal{L}_{\mathrm{Dice}} + \lambda_{\mathrm{aux}} \, \mathcal{L}_{\mathrm{CE}}(\hat{y}_{\mathrm{cls}}, y_{\mathrm{cls}})
$$

---

### Metrics Tracked

- **Intersection over Union (Mean IoU / Jaccard Index)**:

$$
\mathrm{IoU}(P, T) = \frac{|P \cap T|}{|P \cup T|} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP} + \mathrm{FN}}
$$

- **Dice Coefficient / Pixel F1-Score**:

$$
\mathrm{Dice}(P, T) = \frac{2 |P \cap T|}{|P| + |T|} = \frac{2\,\mathrm{TP}}{2\,\mathrm{TP} + \mathrm{FP} + \mathrm{FN}}
$$

- **Area Under ROC Curve (Pixel AUROC)**: Continuous probability ranking metric across all pixels.
- **Pixel Accuracy**:

$$
\mathrm{Acc} = \frac{\mathrm{TP} + \mathrm{TN}}{\mathrm{TP} + \mathrm{TN} + \mathrm{FP} + \mathrm{FN}}
$$

- **Precision, Recall, and Specificity**:

$$
\mathrm{Precision} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}}, \qquad \mathrm{Recall} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}, \qquad \mathrm{Specificity} = \frac{\mathrm{TN}}{\mathrm{TN} + \mathrm{FP}}
$$

---

## Running Tests

Run the complete test suite with `pytest`:

```bash
# Run all unit and integration tests
pytest

# Run tests with detailed verbose output
pytest -v

# Run tests with code coverage report
pytest --cov=sid_unet
```

All tests run cleanly in offline environments without network access.

---

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.
