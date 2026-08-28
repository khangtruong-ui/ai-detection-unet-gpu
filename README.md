# SID-UNet: UNet for AI-Generated Synthetic Image Masking & Classification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Datasets](https://img.shields.io/badge/HuggingFace-Datasets-orange.svg)](https://huggingface.co/datasets/saberzl/SID_Set)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular, config-driven PyTorch framework for detecting and segmenting AI-generated regions in images using a **UNet** architecture with an optional **3-class auxiliary classification head** (`Real`, `Fully AI`, `Partially AI / Inpainting`).

Tailored for large-scale datasets such as [**saberzl/SID_Set**](https://huggingface.co/datasets/saberzl/SID_Set), featuring native **streaming dataset support** (`streaming = True`), flexible loss functions (BCE, Soft Dice, Focal, Combined), comprehensive metric tracking, rich Markdown/JSON evaluation reports, TensorBoard logging, and a complete unit/integration test suite.

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
- **Robust Mask Synthesis**:
  - **Label `0` (Real/Authentic)**: Generated target mask is **full black** (all $0$s).
  - **Label `1` (Fully Synthetic)**: Generated target mask is **full white** (all $1$s).
  - **Label `2` (Partially Synthetic / Tampered)**: Ground truth mask loaded from dataset, normalized, and binarized.
- **Modular UNet Backbone**: Configurable depth, channel counts, bilinear/transposed upsampling, and dropout.
- **Auxiliary 3-Class Classifier**: Bottleneck head to jointly classify whole images as `0: Real`, `1: Fully Synthetic`, or `2: Partially Synthetic` alongside pixel-level masking.
- **Flexible Losses**: BCE with Logits, Soft Dice Loss, Binary Focal Loss, and Hybrid/Combined loss weighting.
- **Comprehensive Reporting**: Automatically exports tabulated Markdown summaries, JSON benchmark files, and 3x3 confusion matrices.
- **Production Ready**: Fully pip-installable (`pip install -e .`) with CLI entrypoints (`sid-train`, `sid-eval`, `sid-predict`), automatic mixed precision (AMP), and gradient clipping.

---

## Dataset Specification

The framework operates on [**`saberzl/SID_Set`**](https://huggingface.co/datasets/saberzl/SID_Set) which contains:
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
│   └── evaluate.yaml             # Dedicated evaluation settings
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
│   │   ├── logger.py             # Console, file, and TensorBoard logger
│   │   └── report.py             # Markdown, JSON, and ASCII report formatting
│   ├── train.py                  # CLI training entrypoint
│   ├── evaluate.py               # CLI evaluation entrypoint
│   └── predict.py                # CLI inference entrypoint
├── tests/
│   ├── test_config.py            # Config parsing & override tests
│   ├── test_dataset.py           # Dataset sample processing tests
│   ├── test_losses.py            # Loss functions unit tests
│   ├── test_mask_utils.py        # Mask synthesis logic tests
│   ├── test_metrics.py           # Metric calculation tests
│   ├── test_models.py            # Forward/backward gradient & shape tests
│   ├── test_trainer_integration.py # End-to-end integration test
│   └── test_transforms.py        # Preprocessing & augmentation tests
├── pyproject.toml                # Pip package specification
├── .gitignore                    # Ignored artifacts & logs
└── README.md                     # Documentation
```

---

## Configuration System

Configurations are written in standard YAML. Any parameter can be overridden via CLI with dot-notation `--override key.nested=value`.

Example `configs/train_streaming.yaml`:
```yaml
project:
  name: "sid_unet_streaming"
  seed: 42
  device: "auto"                    # 'auto', 'cuda', or 'cpu'
  output_dir: "outputs/streaming_run"

data:
  dataset_name: "saberzl/SID_Set"
  streaming: true                   # Always true for streaming operation
  image_size: [256, 256]            # Input resolution [H, W]
  batch_size: 16
  num_workers: 2
  train_samples_per_epoch: 2000     # Delineates epoch steps on streaming datasets
  val_samples: 400                  # Max samples for validation pass
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
  aux_classifier: true             # 3-class auxiliary classification head
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
  early_stopping_patience: 5
  early_stopping_metric: "val_iou"
  early_stopping_mode: "max"
```

---

## Quickstart: How to Run

### 1. Training

Run training with default streaming config:
```bash
# Using installed CLI command:
sid-train --config configs/train_streaming.yaml

# Or using Python module:
python -m sid_unet.train --config configs/train_streaming.yaml
```

**Override parameters via CLI:**
```bash
python -m sid_unet.train \
  --config configs/train_streaming.yaml \
  --override data.batch_size=32 training.learning_rate=5e-4 data.image_size=[256,256]
```

**Resume from a checkpoint:**
```bash
python -m sid_unet.train \
  --config configs/train_streaming.yaml \
  --resume outputs/streaming_run/checkpoints/checkpoint_best.pt
```

---

### 2. Evaluation & Benchmarking

Evaluate a saved checkpoint against the validation split and generate reports:
```bash
# Using installed CLI command:
sid-eval --checkpoint outputs/streaming_run/checkpoints/checkpoint_best.pt

# Or using Python module:
python -m sid_unet.evaluate \
  --checkpoint outputs/streaming_run/checkpoints/checkpoint_best.pt \
  --override data.val_samples=1000 --threshold 0.5
```

This outputs a tabulated summary in the terminal and saves:
- `outputs/eval_reports/evaluation_report.md`
- `outputs/eval_reports/evaluation_report.json`

Example generated report:
```markdown
# SID-UNet Evaluation Report

### Overall Segmentation & Classification Metrics

| Metric                  |   Value |
|-------------------------|---------|
| Eval Total Loss         |  0.3142 |
| Iou                     |  0.8415 |
| Dice                    |  0.9023 |
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
- **Dice Coefficient / F1-Score**: $\frac{2 |P \cap T|}{|P| + |T|}$
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

This project is licensed under the MIT License - see the LICENSE file for details.
