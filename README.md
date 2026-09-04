# SID-UNet: UNet & Pretrained CNN Backbones for AI-Generated Synthetic Image Masking & Classification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
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
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration System](#configuration-system)
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
- **Continuous Master Reports & Collision Skipping**:
  - **Default Checkpoint Continuation**: Automatically resumes from latest or best checkpoints when available.
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

---

## Architectures & Backbones

### 1. Standard UNet

```
                      Input Image (3, H, W)
                               │
                ┌──────────────▼───────────────┐
                │    DoubleConv (in -> 64)      │───────────────┐ (Skip 1: /1)
                └──────────────┬───────────────┘               │
                ┌──────────────▼───────────────┐               │
                │   Down: MaxPool -> Conv (128) │─────────────┐ │ (Skip 2: /2)
                └──────────────┬───────────────┘             │ │
                ┌──────────────▼───────────────┐             │ │
                │   Down: MaxPool -> Conv (256) │───────────┐ │ │ (Skip 3: /4)
                └──────────────┬───────────────┘           │ │ │
                ┌──────────────▼───────────────┐           │ │ │
                │   Down: MaxPool -> Conv (512) │─────────┐ │ │ │ (Skip 4: /8)
                └──────────────┬───────────────┘         │ │ │ │
                               ▼                         │ │ │ │
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
                │ EfficientNet Stage 0..1 (/2) │───────────────┐ (Skip 1: /2)
                └──────────────┬───────────────┘               │
                ┌──────────────▼───────────────┐               │
                │ EfficientNet Stage 2    (/4) │─────────────┐ │ (Skip 2: /4)
                └──────────────┬───────────────┘             │ │
                ┌──────────────▼───────────────┐             │ │
                │ EfficientNet Stage 3    (/8) │───────────┐ │ │ (Skip 3: /8)
                └──────────────┬───────────────┘           │ │ │
                ┌──────────────▼───────────────┐           │ │ │
                │ EfficientNet Stage 4..5 (/16)│─────────┐ │ │ │ (Skip 4: /16)
                └──────────────┬───────────────┘         │ │ │ │
                               ▼                         │ │ │ │
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

## Installation

Install in editable mode using `pip`:

```bash
# Clone and enter directory
cd /workspace

# Install package and all CLI commands (sid-train, sid-eval, sid-cross-eval, sid-predict, sid-illu)
pip install -e .

# Or with development and testing dependencies:
pip install -e ".[dev]"
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

Configurations are written in standard YAML and can be overridden via CLI with `--override key.nested=value`.

### EfficientNet UNet Config Example:
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

---

## Quickstart: How to Run

### 1. Training

#### A. Single Experiment Training
```bash
# Train standard UNet
sid-train --config configs/default.yaml

# Train Pretrained EfficientNet-B0 with UNet multi-scale skip decoder
sid-train --config configs/experiments/efficientnet/efficientnet_b0_unet.yaml

# Train EfficientNet with Sacrifice of Pixel mode
sid-train --config configs/experiments/efficientnet/efficientnet_b0_sacrifice_of_pixel.yaml
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

---

### 2. Evaluation & Benchmarking

#### A. Single Checkpoint Evaluation
```bash
sid-eval --checkpoint outputs/RUN/unet_wide_b32/checkpoints/checkpoint_best.pt --split test
```

#### B. Multi-Checkpoint Evaluation
```bash
sid-eval --checkpoints \
  outputs/RUN/unet_wide_b32/checkpoints/checkpoint_best.pt \
  outputs/RUN/efficientnet_b0_unet/checkpoints/checkpoint_best.pt \
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
