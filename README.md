# SID-UNet: UNet for AI-Generated Synthetic Image Masking & Classification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Datasets](https://img.shields.io/badge/HuggingFace-Datasets-orange.svg)](https://huggingface.co/datasets/saberzl/SID_Set)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A modular, config-driven PyTorch framework for detecting and segmenting AI-generated or tampered regions in images using a **UNet** architecture with an optional **3-class auxiliary classification head** (`Real`, `Fully AI`, `Partially AI / Inpainting`).

Supports large-scale streaming and local datasets including standard 2-column image/mask datasets like [**KhangTruong/IMD2020**](https://huggingface.co/datasets/KhangTruong/IMD2020) and multi-class datasets like [**saberzl/SID_Set**](https://huggingface.co/datasets/saberzl/SID_Set), featuring native **streaming dataset support** (`streaming = True`), flexible loss functions (BCE, Soft Dice, Focal, Combined), comprehensive metric tracking, rich Markdown/JSON evaluation reports, and a complete unit/integration test suite.

---

## Table of Contents

- [Key Features](#key-features)
- [Dataset Specification](#dataset-specification)
- [Architecture](#architecture)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration System](#configuration-system)
- [Quickstart: How to Run](#quickstart-how-to-run)
  - [1. Training](#1-training)
  - [2. Evaluation & Benchmarking](#2-evaluation--benchmarking)
  - [3. Inference & Mask Prediction](#3-inference--mask-prediction)
- [Loss Functions & Metrics](#loss-functions--metrics)
- [Running Tests](#running-tests)
- [License](#license)

---

## Key Features

- **Streaming-First Dataset Pipeline**: Natively consumes massive Hugging Face datasets with `streaming = True` (or non-streaming if configured), preventing disk storage exhaustion.
- **Universal Dataset Support**:
  - **Standard 2-Column Datasets ([KhangTruong/IMD2020](https://huggingface.co/datasets/KhangTruong/IMD2020))**: Contains `image` and `mask` across all three official subsets: **`train`**, **`validation`**, and **`test`**. Masks are directly normalized and binarized, and classification labels are automatically derived.
  - **3-Class Labeled Datasets ([saberzl/SID_Set](https://huggingface.co/datasets/saberzl/SID_Set))**:
    - **Label `0` (Real/Authentic)**: Target mask is **full black** (all $0$s).
    - **Label `1` (Fully Synthetic)**: Target mask is **full white** (all $1$s).
    - **Label `2` (Partially Synthetic / Tampered)**: Ground truth mask loaded from dataset, normalized, and binarized.
- **Robust Subset & Split Resolution**:
  - Automatically loads and evaluates on explicit `test`, `validation`, or `train` splits.
  - Transparently falls back to `validation` when evaluating datasets where `val == test` (or datasets missing an explicit `test` split).

---

## Dataset Specification

The framework supports multiple dataset formats:

### 1. Common / Regular 2-Column Format ([KhangTruong/IMD2020](https://huggingface.co/datasets/KhangTruong/IMD2020))
- **Subsets Available**: `train`, `validation`, and `test`.
- `image`: RGB image (PIL Image or tensor).
- `mask`: Binary / grayscale segmentation mask for manipulated or inpainted regions.
- When `label` is not provided in the dataset, class indicators are inferred automatically from pixel statistics ($0$: all-zero mask / authentic, $1$: all-one mask / fully synthetic, $2$: mixed mask / tampered).

### 2. Multi-Class Labeled Format ([saberzl/SID_Set](https://huggingface.co/datasets/saberzl/SID_Set))
- `image`: RGB image ($1024 \times 1024$ or variable resolutions).
- `label`: Integer class indicator ($0, 1, 2$).
- `mask`: Segmentation mask for tampered/inpainted regions.
  - When `label == 0`, mask is treated as **all zeros ($0$)**.
  - When `label == 1`, mask is treated as **all ones ($1$)**.
  - When `label == 2`, `mask` is extracted from the sample and thresholded to binary $\{0.0, 1.0\}$.

---

## Architecture

```
                      Input Image (3, H, W)
                               │
               ┌───────────────▼───────────────┐
               │    DoubleConv (in -> 64)      │───────────────┐ (Skip 1)
               └───────────────┬───────────────┘               │
               ┌───────────────▼───────────────┐               │
               │   Down: MaxPool -> Conv (128) │─────────────┐ │ (Skip 2)
               └───────────────┬───────────────┘             │ │
               ┌───────────────▼───────────────┐             │ │
               │   Down: MaxPool -> Conv (256) │───────────┐ │ │ (Skip 3)
               └───────────────┬───────────────┘           │ │ │
               ┌───────────────▼───────────────┐           │ │ │
               │   Down: MaxPool -> Conv (512) │─────────┐ │ │ │ (Skip 4)
               └───────────────┬───────────────┘         │ │ │ │
                               ▼                         │ │ │ │
               ┌───────────────────────────────┐         │ │ │ │
               │     Bottleneck Conv (512)     │         │ │ │ │
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
               ┌───────▼───────────────────────┐
               │   OutConv: 1x1 Conv (-> 1)    │
               └───────┬───────────────────────┘
                       ▼
             Binary Mask Logits (1, H, W)
```

---

## Installation

Install in editable mode using `pip` or `uv`:

```bash
# Clone and enter directory
cd /workspace

# Install dependencies and CLI commands
pip install -e .

# Or with development/testing dependencies:
pip install -e ".[dev]"
```

---

## Project Structure

```
├── configs/
│   ├── default.yaml              # Default configuration
│   ├── train_streaming.yaml      # Config tailored for SID_Set streaming
│   ├── train_non_streaming.yaml  # Config for downloaded / map datasets
│   ├── evaluate.yaml             # Dedicated evaluation settings
│   ├── test_smoke.yaml           # Ultra-fast smoke test config
│   ├── test_quick.yaml           # Quick alternative loss test config
│   └── experiments/              # Scaled models, large batch sizes & specialized losses
│       ├── unet_wide_b32.yaml
│       ├── unet_deep_5stage_b32.yaml
│       ├── unet_large_batch_b64.yaml
│       ├── unet_highres_512_b16.yaml
│       ├── unet_heavy_wide_deep_b32.yaml
│       ├── unet_focal_hard_mining_b32.yaml
│       ├── unet_convtranspose_learned_up_b32.yaml
│       └── unet_non_streaming_b32.yaml
├── sid_unet/
│   ├── __init__.py
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── loader.py             # Streaming & Non-streaming PyTorch DataLoaders
│   │   ├── mask_utils.py         # Label 0/1/2 mask synthesis & normalizers
│   │   └── transforms.py         # Joint augmentations & preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── blocks.py             # DoubleConv, Down, Up, OutConv, Classifier
│   │   └── unet.py               # UNet architecture with auxiliary classifier
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── bce.py                # Binary Cross Entropy with Logits
│   │   ├── dice.py               # Soft Dice Loss
│   │   ├── focal.py              # Binary Focal Loss
│   │   ├── combined.py           # Combined Mask Loss
│   │   └── auxiliary.py          # Composite Total Loss (Mask + Aux Classification)
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── segmentation.py       # IoU, Dice/F1, Pixel Acc, Precision, Recall
│   │   └── classification.py     # Accuracy, Macro F1, Confusion Matrix
│   ├── training/
│   │   ├── __init__.py
│   │   ├── callbacks.py          # CheckpointManager and EarlyStopping
│   │   └── trainer.py            # Complete training, validation, and AMP loop
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py             # YAML loading, deep merge, and CLI overrides
│   │   ├── logger.py             # Console and file logger
│   │   └── report.py             # Markdown, JSON, and ASCII report formatting
│   ├── train.py                  # CLI training entrypoint (single & multi-run)
│   ├── evaluate.py               # CLI evaluation entrypoint
│   └── predict.py                # CLI inference entrypoint
├── tests/
│   ├── test_cli.py               # Single and multi-config CLI tests
│   ├── test_config.py            # Config parsing & override tests
│   ├── test_dataset.py           # Dataset sample & sample-limit tests
│   ├── test_losses.py            # Loss functions unit tests
│   ├── test_mask_utils.py        # Mask synthesis logic tests
│   ├── test_metrics.py           # Metric calculation tests
│   ├── test_models.py            # Forward/backward gradient & shape tests
│   ├── test_report.py            # Single & multi-experiment report tests
│   ├── test_trainer_integration.py # End-to-end integration test
│   └── test_transforms.py        # Preprocessing & augmentation tests
├── pyproject.toml                # Pip package specification
├── .gitignore                    # Ignored artifacts & logs
└── README.md                     # Documentation
```

---

## Configuration System

Configurations are written in standard YAML. Any parameter can be overridden via CLI with dot-notation `--override key.nested=value`.

Example `configs/default.yaml`:
```yaml
project:
  name: "sid_unet_baseline"
  seed: 42
  device: "auto"                    # 'auto', 'cuda', or 'cpu'
  output_dir: "outputs"

data:
  dataset_name: "KhangTruong/IMD2020"
  streaming: false                  # false for downloaded datasets, true for streaming
  image_size: [256, 256]            # Input resolution [H, W]
  batch_size: 16
  num_workers: 2
  train_split: "train"
  val_split: "validation"
  test_split: "test"
  train_samples_per_epoch: -1       # -1 for full dataset until depletion
  val_samples: -1                   # -1 for full validation set
  test_samples: -1                  # -1 for full test set
  evaluate_on_test: true            # Automatically evaluate on test set post-training
  augmentations:
    horizontal_flip: 0.5
    vertical_flip: 0.2
    random_rotate90: 0.5

model:
  name: "unet"
  in_channels: 3
  out_channels: 1
  features: [64, 128, 256, 512]
  bilinear: true
  dropout: 0.1
  aux_classifier: false            # Set false for 2-column image/mask datasets like IMD2020 (set true for 3-class datasets like SID_Set)
  num_classes: 3

loss:
  mask_loss_type: "combined"       # 'bce', 'dice', 'focal', or 'combined'
  bce_weight: 0.5
  dice_weight: 0.5
  aux_loss_type: "cross_entropy"
  aux_weight: 0.2                  # Weight for classification loss

training:
  epochs: 10
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: "adamw"
  scheduler: "cosine"
  amp: true
  auto_batch_size: true            # Enabled by default to automatically search safe batch size and avoid OOM
  save_best: true
  save_latest: false               # Saves only best model checkpoint by default
  early_stopping_patience: 5
  early_stopping_metric: "val_iou"
  early_stopping_mode: "max"
```

---

## Quickstart: How to Run

### 1. Training

#### A. Single Experiment Training
Run training with a configuration file (includes per-epoch validation and post-training test set evaluation):
```bash
# Using installed CLI command:
sid-train --config configs/default.yaml

# Or using Python module:
python -m sid_unet.train --config configs/default.yaml
```

**Fast Smoke Testing:**
Run minimal 1-epoch / small-step test configurations:
```bash
# Ultra-fast smoke test:
sid-train --config configs/test_smoke.yaml

# Quick alternative loss test:
sid-train --config configs/test_quick.yaml
```

#### B. Multi-Experiment Suite (Run Multiple Configs in One Command)
You can pass multiple configuration files to execute a series of experiments sequentially. Each run gets a clean, dedicated output directory (`RUN001`, `RUN002`, `RUN003`, etc.), and a consolidated markdown & JSON comparative benchmark report is generated automatically:
```bash
sid-train --configs configs/test_smoke.yaml configs/test_quick.yaml --output_dir outputs/benchmark_suite
```

**Override parameters via CLI across runs:**
```bash
python -m sid_unet.train \
  --config configs/train_streaming.yaml \
  --override data.batch_size=32 training.learning_rate=5e-4 data.image_size=[256,256]
```

**Full Dataset Passes (Run Until Dataset Depletes):**
Set `data.train_samples_per_epoch: -1` (or `steps_per_epoch: -1`) to stream or iterate through the entire dataset until depletion rather than truncating to fixed epoch sample counts:
```bash
python -m sid_unet.train \
  --config configs/train_streaming.yaml \
  --override data.train_samples_per_epoch=-1 data.val_samples=-1
```

**Resume from a checkpoint:**
```bash
python -m sid_unet.train \
  --config configs/train_streaming.yaml \
  --resume outputs/streaming_run/checkpoints/checkpoint_best.pt
```

#### C. Handling Large Batch Sizes & GPU Memory (OOM Prevention)
When training on high resolutions or constrained GPU memory, multiple built-in mechanisms prevent Out-of-Memory (OOM) failures:

1. **Automatic Batch Sizing (Enabled by Default)**:
   Probes device VRAM before training and scales down the physical batch size while increasing gradient accumulation steps to maintain the target effective batch size. Pass `--no-auto-batch-size` if you wish to disable auto-tuning:
   ```bash
   python -m sid_unet.train --config configs/default.yaml --batch_size 64
   ```

2. **Gradient Accumulation (`--gradient-accumulation-steps`)**:
   Simulate large effective batch sizes with smaller memory footprints:
   ```bash
   # Effective batch size of 64 using micro-batch size 16 accumulated over 4 steps:
   python -m sid_unet.train --config configs/default.yaml --batch_size 16 --gradient-accumulation-steps 4
   ```

3. **Gradient (Activation) Checkpointing (`--gradient-checkpointing`)**:
   Reduces activation memory by ~60-70% during backpropagation, enabling much larger batches or higher image resolutions:
   ```bash
   python -m sid_unet.train --config configs/default.yaml --gradient-checkpointing
   ```

4. **Dynamic OOM Auto-Recovery**:
   If an unexpected OOM occurs during a training or validation step, the Trainer automatically catches the exception, clears the PyTorch allocator cache, splits the batch into smaller micro-batches, and completes the step seamlessly without crashing the job.

---

### 2. Evaluation & Benchmarking

#### A. Single Checkpoint Evaluation
Evaluate a checkpoint against the test or validation split (reports are saved and override in-place in the checkpoint's local run directory, e.g. `outputs/RUN001/eval_reports/`):
```bash
# Evaluate on test set (automatically falls back to validation if test split is not present):
sid-eval --checkpoint outputs/RUN001/checkpoints/checkpoint_best.pt --split test

# Evaluate on specific split with sample limit and custom threshold:
sid-eval \
  --checkpoint outputs/RUN001/checkpoints/checkpoint_best.pt \
  --split validation \
  --samples 500 \
  --threshold 0.5

# Or with config overrides:
python -m sid_unet.evaluate \
  --checkpoint outputs/RUN001/checkpoints/checkpoint_best.pt \
  --split test \
  --override data.batch_size=32 data.streaming=false
```

#### B. Multi-Checkpoint Evaluation (Evaluate Multiple Models at Once)
Pass multiple checkpoints or glob patterns to evaluate a suite of models sequentially. Individual reports override in-place in each model's directory, and a consolidated comparative evaluation report (`multi_checkpoint_evaluation.md` / `.json`) with side-by-side metrics is generated:
```bash
# Evaluate multiple checkpoints explicitly:
sid-eval --checkpoints outputs/RUN001/checkpoints/checkpoint_best.pt outputs/RUN002/checkpoints/checkpoint_best.pt --split test

# Evaluate all best checkpoints across all runs via glob pattern:
sid-eval --checkpoints "outputs/RUN*/checkpoints/checkpoint_best.pt" --split test
```

This outputs a tabulated summary in the terminal and saves:
- `<checkpoint_run_dir>/eval_reports/evaluation_report.md` (overridden in-place)
- `<checkpoint_run_dir>/eval_reports/evaluation_report.json`
- `outputs/multi_checkpoint_evaluation.md` (when multiple checkpoints are evaluated)

Example generated report:
```markdown
# SID-UNet Evaluation Report

### Overall Segmentation & Classification Metrics

| Metric                  |   Value |
|-------------------------|---------|
| Eval Total Loss         |  0.3142 |
| Iou                     |  0.8415 |
| Dice / Pixel F1         |  0.9023 |
| Pixel Auroc             |  0.9610 |
| Pixel Acc               |  0.9412 |
| Precision               |  0.8876 |
| Recall                  |  0.9201 |
| Specificity             |  0.9521 |
| Aux Accuracy            |  0.9125 |
| Aux Macro F1            |  0.9080 |
| Total Evaluated Samples | 1000    |

### Per-Subset Metrics Breakdown

| Subset                            |   Mean IoU |   Dice / F1 |   Pixel Accuracy |   Sample Count |
|-----------------------------------|------------|-------------|------------------|----------------|
| Label 0 (Real / Black Mask)       |     1.0000 |      1.0000 |           0.9850 |            334 |
| Label 1 (Synthetic / White Mask)  |     0.8920 |      0.9410 |           0.9120 |            333 |
| Label 2 (Tampered / Partial Mask) |     0.6325 |      0.7659 |           0.9266 |            333 |
```

---

### 3. Inference & Mask Prediction

Predict binary masks for single images or full directories:

```bash
# Single image prediction
sid-predict \
  --checkpoint outputs/streaming_run/checkpoints/checkpoint_best.pt \
  --image /path/to/test_image.jpg \
  --output_dir predictions \
  --save_overlay

# Directory batch prediction
python -m sid_unet.predict \
  --checkpoint outputs/streaming_run/checkpoints/checkpoint_best.pt \
  --input_dir /path/to/image_folder \
  --output_dir predictions \
  --threshold 0.5 \
  --save_overlay
```

**Inference Outputs:**
- `<image_name>_mask.png`: Binary mask ($0$ = Authentic background, $255$ = AI-generated).
- `<image_name>_overlay.png`: Semi-transparent red highlight over detected AI regions on the original image.

**Python API Usage:**
```python
import torch
from sid_unet.models import UNet

# Load model directly from checkpoint (architecture and weights restored automatically)
model = UNet.from_checkpoint("outputs/checkpoints/checkpoint_best.pt", device="cuda")

# Run inference
# x: (B, 3, H, W) normalized tensor
with torch.no_grad():
    binary_mask = model.predict_mask(x, threshold=0.5)  # (B, 1, H, W)
```

---

## Loss Functions & Metrics

### Loss Functions
- **BCELoss**: Binary Cross-Entropy with Logits:
  $$\mathcal{L}_{\text{BCE}} = - [y \log \sigma(x) + (1 - y) \log (1 - \sigma(x))]$$
- **DiceLoss**: Soft Dice Loss:
  $$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum p_i y_i + \epsilon}{\sum p_i + \sum y_i + \epsilon}$$
- **FocalLoss**: Binary Focal Loss with focusing parameter $\gamma$ and balance $\alpha$:
  $$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
- **Total Loss**:
  $$\mathcal{L}_{\text{Total}} = \alpha \mathcal{L}_{\text{BCE}} + \beta \mathcal{L}_{\text{Dice}} + \lambda_{\text{aux}} \mathcal{L}_{\text{CE}}(\hat{y}_{\text{cls}}, y_{\text{cls}})$$

### Metrics Tracked
- **Intersection over Union (IoU / Jaccard Index)**: $\frac{|P \cap T|}{|P \cup T|}$
- **Pixel F1-Score / Dice Coefficient**: $\frac{2 |P \cap T|}{|P| + |T|}$
- **Area Under ROC Curve (AUROC / ROC-AUC)**: Continuous pixel prediction probability ranking performance
- **Pixel Accuracy**: $\frac{\text{TP} + \text{TN}}{\text{Total Pixels}}$
- **Precision, Recall & Specificity**
- **Auxiliary 3-Class Accuracy, Macro F1, and Confusion Matrix**

---

## Running Tests

Run the comprehensive unit and integration test suite:

```bash
# Run all tests
pytest

# Run tests with detailed verbose output
pytest -v

# Run tests with code coverage report
pytest --cov=sid_unet
```

All tests run cleanly without network dependency using synthetic test batches and mocked data fixtures.

---

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.
