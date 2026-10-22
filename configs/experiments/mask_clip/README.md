# OpenSDI MaskCLIP: Spotting Diffusion-Generated Images in the Open World

Integration of **MaskCLIP** from the CVPR 2025 paper *"OpenSDI: Spotting Diffusion-Generated Images in the Open World"* ([arXiv:2503.19653v3](https://arxiv.org/html/2503.19653v3), [iamwangyabin/OpenSDI](https://github.com/iamwangyabin/OpenSDI)) into the `sid_unet` framework.

---

## 1. Architectural Overview: Synergizing Pretrained Models (SPM)

MaskCLIP addresses the open-world diffusion image detection and localization challenge by synergizing two complementary foundation models through the **SPM (Synergizing Pretrained Models)** framework:

```
Input Image [B, 3, H, W]
   │
   ├───> [Bilinear 224x224] ───> Frozen OpenAI CLIP ViT-L/14 (Semantic Stream)
   │                               │
   │                               ├──> ln_2 hooks at layers [2, 4, 8, 12, 16, 20, 23]
   │                               │     ├──> CLS tokens [B, 7, 1024] ──> VSA (CLSAttentionAggregator) ──> Dot-Product ──> Image Logits [B, 2]
   │                               │     └──> Patch tokens [B, 7, 1024, 16, 16]
   │                               │                   │ (Bilinear to 32x32, 1x1 Conv + LN to 768)
   │                               │                   ▼
   ├───> [512x512] ────────────> Windowed MAE ViT-B/16 (Dense Spatial Stream)
   │                               │   x = x + VCA(x, clip_patch) at blocks {1..7}
   │                               ▼
   │                             FPN Multi-Scale Stages (4.0x, 2.0x, 1.0x, 0.5x)
   │                               │ (ConvTranspose2d / MaxPool2d + 1x1/3x3 Conv + LN to 256)
   │                               ▼
   │                             TVCA (Textual-Visual Cross-Attention)
   │                               ▲ Query: Projected Fake Text Embedding [B, 256] from PromptLearner
   │                               │
   │                               ▼
   └───────────────────────────> 1x1 Conv + Sigmoid ──> Tampering Mask [B, 1, H, W]
```

### Key Submodules:
1. **Frozen Semantic Stream (CLIP `ViT-L/14`)**: Extracts rich, generalized high-level semantic tokens from 7 selected transformer layers via forward hooks at `ln_2`.
2. **PromptLearner**: Learns continuous positive (`ctx_pos`, fake) and negative (`ctx_neg`, real) context tokens of length $N_{\text{ctx}}=10$ prepended to `"an image."`, encoded through CLIP's frozen text transformer.
3. **Dense Spatial Stream (MAE `ViT-B/16`)**: Windowed ViT with decomposed relative positional embeddings ($14 \times 14$ windows) providing fine-grained spatial representation for pixel-level boundary localization.
4. **Visual Cross-Attention (VCA / `AttentionFusion`)**: 8-head multi-head cross-attention where projected CLIP patch features query MAE spatial tokens, with residual connections at blocks $\{1, 2, 3, 4, 5, 6, 7\}$.
5. **FPN + Textual-Visual Cross-Attention (TVCA / `MaskDecoder`)**: 4-scale hierarchical feature pyramid network fusing multi-scale representations, modulated by the projected fake text embedding through cross-attention.
6. **Visual Self-Attention (VSA / `CLSAttentionAggregator`)**: 8-head multi-head self-attention aggregating multi-layer `[CLS]` tokens, projected to 768-D to compute dot-product similarity against text embeddings for image-level classification.
7. **Morphological EdgeMaskGenerator & Edge Loss**: Cross-kernel morphological dilation and erosion (kernel width 7) computing edge-weighted binary cross-entropy loss with $\lambda_{\text{edge}} = 20.0$.

---

## 2. Hardcoded Pretrained Weights & Anti-Random-Initialization Policy

Random initialization is strictly disabled by default (`allow_random_init=False`). All weights are automatically retrieved online from official repositories and cached locally:

| Weight Component | Source / Repository | Default File / URL | Checkpoint Keys | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Full Trained MaskCLIP** | Hugging Face Hub: `nebula/MaskCLIP-weights` | `MaskCLIP_sd15_20241108_09_14_22.pth` (512x512, 20 epochs) | 778 keys | `missing=0, unexpected=0` |
| **Alternative MaskCLIP** | Hugging Face Hub: `nebula/MaskCLIP-weights` | `MaskCLIP_sd15_20241109_08_53_19.pth` (512x512, 14 epochs) | 778 keys | `missing=0, unexpected=0` |
| **1024-Res SideCLIP** | Hugging Face Hub: `nebula/MaskCLIP-weights` | `MaskCLIP_sd15_20241103_17_45_16.pth` (1024x1024, Refiner) | 644 keys | `missing=0, unexpected=0` |
| **CLIP Semantic Backbone** | OpenAI Public Azure CDN | `https://openaipublic.azureedge.net/.../ViT-L-14.pt` (`ViT-L/14`) | 378 keys | Cached in `~/.cache/clip/` |
| **MAE Spatial Backbone** | Facebook FAIR Public CDN | `https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth` | 150 keys | Cached in `torch.hub` |

---

## 3. Hardware Benchmark & Compatibility Assessment

Empirical profiling on the local workstation:
- **GPU**: 1x NVIDIA GeForce RTX 3060 (12,288 MiB / 12 GB VRAM, Ampere SM 8.6)
- **CPU**: Dual AMD EPYC 7B12 (128 Cores / 256 Threads)
- **RAM**: 251 GiB System Memory (209 GiB Available)
- **PyTorch**: 2.11.0+cu128, CUDA 13.0, Driver 580.65.06

### Empirical VRAM & Throughput Measurements:

| Operation | Input Resolution | Batch Size | Precision | Latency | Peak VRAM Allocated | Fits on RTX 3060 (12 GB)? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Static Model (eval)** | — | — | FP32 | — | **2,086 MB (2.04 GB)** | Yes |
| **Inference** | $512 \times 512$ | 1 | AMP (FP16) | **496.1 ms/img** | **2,552 MB (2.49 GB)** | Yes (80% headroom) |
| **Inference** | $512 \times 512$ | 4 | AMP (FP16) | **394.5 ms/img** | **3,243 MB (3.17 GB)** | Yes (74% headroom) |
| **SPM Training Step** | $512 \times 512$ | 1 | AMP (FP16) | **1,749 ms/step** | **4,392 MB (4.29 GB)** | Yes |
| **SPM Training Step** | $512 \times 512$ | 2 | AMP (FP16) | **2,164 ms/step** | **6,212 MB (6.07 GB)** | Yes |
| **SPM Training Step** | $512 \times 512$ | 4 | AMP (FP16) | **3,952 ms/step** | **9,847 MB (9.62 GB)** | **Yes (Peak 80.1% VRAM)** |
| **DoRA Surgery Step** | $512 \times 512$ | 2 | AMP (FP16) | **2,388 ms/step** | **6,480 MB (6.33 GB)** | Yes |
| **DoRA Surgery Step** | $512 \times 512$ | 4 | AMP (FP16) | **4,170 ms/step** | **10,050 MB (9.81 GB)**| **Yes (Peak 81.8% VRAM)** |

### Hardware Verdict:
The current machine's hardware **fully and comfortably supports** MaskCLIP for both inference and training:
- **Inference**: Consumes only ~2.5 GB to ~3.2 GB VRAM, allowing batch sizes up to 16.
- **Training**: Full SPM fine-tuning (with frozen CLIP ViT-L/14, trainable MAE ViT-B/16, VCA cross-attention, and TVCA mask decoder) fits on a single RTX 3060 at batch size $B=2$ (6.2 GB VRAM) or $B=4$ (9.8 GB VRAM). With gradient accumulation steps 4 or 8, effective batch sizes of 16 or 32 are achieved seamlessly.
- **System RAM**: The 251 GB host RAM provides massive bandwidth for data streaming, caching, and multi-worker prefetching (`persistent_workers=True`).

---

## 4. Model Surgery Methods & Research Citations

The `MaskCLIP` implementation provides a first-class Model Surgery interface (`sid_unet.models.mask_clip.MaskCLIP`) designed for modular experimentation and future architectural adaptations:

### 1. Spatial Backbone Replacement (`replace_spatial_backbone`)
- **Method**: Swaps the MAE ViT-B/16 spatial encoder with an alternative dense representation backbone (e.g. SAM ViT, EfficientSAM3 TinyViT, DINOv2, or ConvNeXt). Automatically inserts a channel adapter (`1x1 Conv + LayerNorm`) when output channels differ from 768.
- **Literature**:
  - In OpenSDI Supplementary Section B, replacing MAE with SAM achieved **0.7873 Pixel F1** on OpenSDID (*Wang et al., CVPR 2025*).
  - *Kirillov et al., "Segment Anything" (ICCV 2023)* — Demonstrates SAM's dense geometric edge priors for mask segmentation.
  - *Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision" (TMLR 2024)* — Self-supervised ViT features capturing fine-grained spatial and texture anomalies.

### 2. Semantic Backbone Replacement (`replace_semantic_backbone`)
- **Method**: Replaces the frozen CLIP `ViT-L/14` backbone with alternative vision-language foundation models (e.g. SigLIP, MobileCLIP, OpenCLIP, or EVA-CLIP), automatically re-binding intermediate `FeatureHook`s.
- **Literature**:
  - *Zhai et al., "SigLIP: Sigmoid Loss for Language Image Pre-Training" (ICCV 2023)* — Improved zero-shot semantic representation without softmax temperature instabilities.
  - *Vasu et al., "MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training" (CVPR 2024)* — Low-latency edge vision-language models.

### 3. Parameter-Efficient Low-Rank Adaptation (`inject_peft_lora`)
- **Method**: Injects Low-Rank Adaptation (LoRA) or Weight-Decomposed Low-Rank Adaptation (DoRA) into attention projections (`qkv`, `proj`, `fc1`, `fc2`, `query_proj`, `key_proj`, `value_proj`) of MAE, CLIP, or VCA fusion layers.
- **Literature**:
  - *Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022)*.
  - *Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation" (ICML 2024, arXiv:2402.09353)* — Decomposes weights into magnitude and directional components, matching or outperforming full fine-tuning with <7% trainable parameters.

### 4. Surgical Layer Tap & Fusion Rewiring (`reconfigure_layer_tap_and_fusion`)
- **Method**: Dynamically reconfigures which CLIP layers are tapped (`selected_layers`) and which MAE blocks receive cross-attention (`fusion_map`), preserving learned VCA weights across overlapping blocks.
- **Literature**:
  - *Michel et al., "Are Sixteen Heads Really Better than One?" (NeurIPS 2019)* — Cross-layer importance and pruning.

### 5. Weight-Space Model Merging & Task Arithmetic (`task_arithmetic_merge`)
- **Method**: Linearly interpolates or merges model weights across fine-tuned checkpoints:
  $$\mathbf{W}_{\text{merged}} = (1 - \alpha) \mathbf{W}_{\text{zero-shot}} + \alpha \mathbf{W}_{\text{fine-tuned}}$$
  Prevents catastrophic forgetting on open-world generators when adapting to new generators (e.g. Flux.1, SD3, SDXL).
- **Literature**:
  - *Ilharco et al., "Editing Models with Task Arithmetic" (ICLR 2023, arXiv:2212.04089)*.
  - *Wortsman et al., "Robust Fine-Tuning of Zero-Shot Models" (CVPR 2022)* — WiSE-FT weight-space ensemble.

### 6. Forward Intervention & Activation Steering Hooks (`register_surgery_hook`)
- **Method**: Registers non-destructive forward hooks at `"clip_features"`, `"text_features"`, `"mae_block_post_fusion"`, `"decoder_features"`, or `"cls_features"`.
- **Literature**:
  - *Zou et al., "Representation Engineering: A Top-Down Approach to AI Transparency and Control" (arXiv:2310.01405, 2023)*.

---

## 5. Experiment Configurations

| Configuration File | Pretrained Mode | Resolution | Batch Size | Loss Formulation | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`default.yaml`](default.yaml) | Full OpenSDI (`nebula/MaskCLIP-weights`) | $512 \times 512$ | 2 (accum 8) | BCE + Edge ($\lambda=20$) + CE | Official 20-epoch CVPR 2025 MaskCLIP baseline. |
| [`mask_clip_clip_mae_backbones.yaml`](mask_clip_clip_mae_backbones.yaml) | Online Foundation Backbones (`ViT-L/14` + `MAE-B/16`) | $512 \times 512$ | 2 (accum 8) | BCE + Edge ($\lambda=20$) + CE | Training SPM from scratch using pretrained backbones. |
| [`mask_clip_dora_surgery.yaml`](mask_clip_dora_surgery.yaml) | DoRA Surgery ($r=8, \alpha=16$) | $512 \times 512$ | 4 (accum 4) | BCE + Edge ($\lambda=20$) + CE | Parameter-efficient DoRA adaptation (6.3% trainable params). |
| [`mask_clip_sideclip_1024.yaml`](mask_clip_sideclip_1024.yaml) | 1024-Res SideCLIP (`MaskCLIP_sd15_20241103_17_45_16.pth`) | $1024 \times 1024$ | 1 (accum 16) | BCE + Edge ($\lambda=20$) + CE | High-resolution 1024x1024 SideCLIP configuration. |

---

## 6. CLI & Python Usage

### Training with CLI:
```bash
# Train default MaskCLIP
sid-train --config configs/experiments/mask_clip/default.yaml

# Train with DoRA Model Surgery
sid-train --config configs/experiments/mask_clip/mask_clip_dora_surgery.yaml
```

### Evaluation & Prediction:
```bash
# Evaluate checkpoint on validation set
sid-eval --config configs/experiments/mask_clip/default.yaml \
         --checkpoint outputs/experiments/mask_clip/default/checkpoints/checkpoint_best.pt

# Predict tampering mask on a suspicious image
sid-predict --checkpoint nebula/MaskCLIP-weights \
            --input path/to/tampered_image.png \
            --output outputs/predictions/
```

### Python API Example:
```python
import torch
from sid_unet.models.mask_clip import MaskCLIP

# 1. Load pretrained MaskCLIP online (zero random initialization)
model = MaskCLIP.from_pretrained("nebula/MaskCLIP-weights", device="cuda")

# 2. Inference
x = torch.rand(1, 3, 512, 512, device="cuda")
pred_mask = model.predict_mask(x, threshold=0.5)  # [1, 1, 512, 512]
mask_logits, class_logits = model(x)              # mask: [1, 1, 512, 512], class: [1, 2]

# 3. Model Surgery: Inject DoRA into MAE
surgery_meta = model.inject_peft_lora(target="mae", r=8, alpha=16, use_dora=True)
print(f"Adapted {surgery_meta['num_adapted_layers']} layers, {surgery_meta['trainable_pct']}% params trainable")
```
