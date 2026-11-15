# Requirements Specification: FOCAL Critical Error Fixes & Finetuning Strategy 1 (Train/Eval Alignment & Preprocessing Fixes)

**Author**: Antigravity Assistant  
**Date**: November 15, 2026  
**Target Repository**: `khangtruong-ui/ai-detection-unet-gpu`  
**Status**: Proposed Requirements Specification (No Implementation Code)  
**Scope**: `sid_unet/models/focal.py`, `configs/experiments/focal/*.yaml`, `sid_unet/training/trainer.py`, `sid_unet/losses/auxiliary.py`

---

## 1. Executive Summary

An audit of the FOCAL integration (`sid_unet/models/focal.py`, `configs/experiments/focal/focal_hrnet.yaml`, `configs/experiments/focal/focal_vit.yaml`) against the FOCAL paper (*Rethinking Image Forgery Detection via Soft Contrastive Learning and Unsupervised Clustering*, arXiv:2308.09307 / IEEE TDSC 2025, `HighwayWu/FOCAL`) and the existing cross-evaluation investigation (`docs/cross_evaluation_focal_pim_mismatch_investigation.md`) identified three critical errors that prevent valid finetuning under the current configurations.

This document specifies **requirements only** (no code changes) to:
1. Resolve all critical errors in FOCAL's training, preprocessing, and clustering paths.
2. Specify **Strategy 1 (Minimal-Surgery Finetuning Alignment)**—establishing strict parity between training and evaluation paths across two supported finetuning modes (**Mode 1A: Supervised Finetuning** and **Mode 1B: Paper-Aligned Image-Relative Contrastive Finetuning**).

---

## 2. Critical Errors Identified in Current FOCAL Setup

### 2.1 Critical Error 1: Train/Eval Inference Head Disconnect
- **Observed Behavior**:
  - In `sid_unet/models/focal.py` (`FOCAL.forward`, lines 980–988):
    - When `self.training == True`, `FOCAL` routes backbone features `feats` through `self.fc = nn.Conv2d(in_dim, 2, kernel_size=3, padding=1)` and computes `mask_logits = logits_full[:, 1:2] - logits_full[:, 0:1]`, which `Trainer` supervises via `CombinedMaskLoss` (BCE + Dice) in `focal_hrnet.yaml` and `focal_vit.yaml`.
    - When `self.training == False` and `self.use_clustering == True` (the default), `FOCAL.forward()` **completely bypasses `self.fc`** and runs `_cluster_features(feats, orig_h, orig_w)`.
- **Root Failure**:
  - As established in `docs/cross_evaluation_focal_pim_mismatch_investigation.md` and `tests/test_focal_pim_mismatch_audit.py`, `module.fc` in upstream `FOCAL_HRNet_weights.pth` was never trained as a segmentation head (it produces unconditionally positive logit differences across 100% of pixels).
  - Training `self.net` + `self.fc` with global BCE + Dice forces the backbone (`self.net`) to learn a **global cross-image decision boundary**, directly destroying FOCAL's **image-relative** contrastive feature geometry—while `validate()` evaluates unsupervised clustering on those distorted features and ignores the trained `self.fc` weights entirely.

### 2.2 Critical Error 2: Resolution Downscaling & Input Normalization Bottlenecks
- **Observed Behavior**:
  - Both `configs/experiments/focal/focal_hrnet.yaml` and `configs/experiments/focal/focal_vit.yaml` specify `data.image_size: [256, 256]` and omit `data.normalize`.
  - In `sid_unet/dataset/loader.py` (line 2270), `normalize` defaults to `True` (ImageNet mean/std standardization), forcing `FOCAL._ensure_raw_rgb()` (`focal.py`, lines 871–878) to detect out-of-range values and invert standardization on every batch.
  - In `FOCAL.extract_features()` (`focal.py`, lines 884–886), ViT inputs not already `1024×1024` are unconditionally upsampled to `(1024, 1024)` via bilinear interpolation.
- **Root Failure**:
  - **HRNet (`256×256`)**: Downsampling input images to `256×256` destroys high-frequency splicing boundaries, JPEG blocking, and CFA/noise artifacts (upstream FOCAL operates at `1024×1024`) and yields a coarse `64×64` (`H/4`) feature grid.
  - **ViT (`256×256 → 1024×1024`)**: Feeding blurred `256×256` images that are immediately upscaled to `1024×1024` inside `extract_features()` forces the model to pay the full `1024×1024` SAM ViT-Large compute and VRAM cost (`batch_size: 4`) with zero high-frequency detail gain, while still outputting `64×64` (`H/16`) features.
  - **Normalization Round-Trip**: Applying ImageNet mean/std normalization in the dataloader only to clamp and denormalize in `_ensure_raw_rgb()` introduces unnecessary floating-point/clamping overhead and potential precision loss under AMP.

### 2.3 Critical Error 3: Forced-Minority Clustering on Pristine Images
- **Observed Behavior**:
  - `FOCAL._cluster_features()` (`focal.py`, lines 906–953) partitions every image into $k=2$ clusters and deterministically assigns the smaller cluster as tampered (`cond = (c1_count > c0_count)`).
- **Root Failure**:
  - On 100% authentic/pristine images (containing zero tampered pixels), $k=2$ clustering cannot output an all-pristine mask; it is mathematically forced to split background/foreground textures and hallucinate a tampered minority cluster, inflating false positives during finetuning validation and cross-evaluation.

---

## 3. Strategy 1 Requirements: Minimal-Surgery Finetuning Alignment

Strategy 1 resolves all three critical errors while preserving FOCAL's pretrained weights and `Trainer` compatibility. Implementations **must** satisfy Requirements 3.1 through 3.5 below.

### 3.1 Requirement 1: Input Preprocessing & Resolution Alignment
1. **Explicit Raw `[0, 1]` Input Pipeline**:
   - `configs/experiments/focal/focal_hrnet.yaml` and `configs/experiments/focal/focal_vit.yaml` **must** explicitly set `data.normalize: false` so `JointToTensorAndNormalize` delivers raw `[0.0, 1.0]` RGB tensors directly.
   - `FOCAL._ensure_raw_rgb()` **must** remain in place as a defensive fallback guard in case external callers or cross-eval configs pass ImageNet-standardized tensors.
2. **Resolution Policy**:
   - **HRNet Config (`focal_hrnet.yaml`)**: Upgrade `data.image_size` from `[256, 256]` to a forensic-preserving resolution (`[512, 512]` minimum, or `[1024, 1024]` where VRAM permits), with `batch_size` / `gradient_accumulation_steps` tuned accordingly.
   - **ViT Config & Feature Extraction (`focal_vit.yaml`, `FOCAL.extract_features`)**:
     - Eliminate the wasteful `256×256 → 1024×1024` upsampling mismatch.
     - Either configure `data.image_size` directly at the target native resolution (`[512, 512]` or `[1024, 1024]`), **and/or** make ViT input resizing in `FOCAL.extract_features()` configurable (e.g., `force_vit_1024: bool = False` / `native` mode) rather than hardcoded to `1024×1024`, leveraging `ImageEncoderViT`'s existing bicubic `pos_embed` interpolation (`focal.py`, lines 596–600) and `SAMAttention._get_rel_pos` linear interpolation (`focal.py`, lines 462–478).

### 3.2 Requirement 2: Strict Train/Eval Head Parity (Two Supported Strategy 1 Modes)
`FOCAL` **must** enforce that `train()` and `eval()` optimize and evaluate the **same** decision mechanism, configured explicitly via a finetuning mode (e.g., `model.finetune_mode: "supervised" | "contrastive"` or explicit `use_clustering` semantics):

#### Mode 1A: Supervised Finetuning Mode (`use_clustering: false` in both train and eval)
Intended for single-domain or mixed-domain supervised adaptation (e.g., `IMD2020`, `DEAL300K-30K`, inpainting/AI-manipulation masks) where a calibrated pixel mask predictor is desired:
1. **Unified Forward Path**:
   - Both `self.training == True` and `self.training == False` **must** execute the supervised segmentation head (never silently switching to `_cluster_features()` during `eval()`).
2. **Head Upgrade Beyond Bare `3×3` Conv**:
   - Replace or optionally upgrade the bare `self.fc = nn.Conv2d(in_dim, 2, 3, padding=1)` (`focal.py`, line 734) with a lightweight multi-scale / residual upsampling decoder (e.g., `Conv → Norm (GroupNorm/LayerNorm2d) → GELU → Conv → progressive upsample`) so `H/4` (HRNet) and `H/16` (ViT) feature maps do not rely solely on a single linear `3×3` projection + bilinear interpolation.
   - Backward compatibility: Strict loading of upstream checkpoints (`FOCAL_HRNet_weights.pth`, `FOCAL_ViT_weights.pth`) **must** continue to succeed without error when a newly initialized supervised head is attached (i.e., `self.net` weights are strictly verified, while newly added decoder layers are initialized cleanly and exempted from pretrained key-missing failure).
3. **Backbone Protection / Differential Optimization**:
   - Support freezing `self.net` for head warmup (or configuring a lower backbone learning rate relative to the newly initialized supervised head) so random/uncalibrated head gradients in early epochs do not corrupt pretrained contrastive backbone features.
4. **Boundary-Aware Supervision**:
   - Supervised FOCAL configs should leverage `SIDTotalLoss`'s existing `edge_weight > 0` (`EdgeMaskGenerator` in `sid_unet/losses/auxiliary.py`, lines 136–147) alongside BCE + Dice to sharpen manipulation boundaries.

#### Mode 1B: Paper-Aligned Image-Relative Contrastive Finetuning Mode (`use_clustering: true`)
Intended for preserving FOCAL's cross-dataset generalization via Soft Contrastive Learning (SCL / `MyInfoNCE`):
1. **Decouple `self.fc` from Contrastive Training**:
   - When `finetune_mode == "contrastive"` (or `use_clustering == True` with contrastive supervision), `self.fc` **must not** be trained with global BCE + Dice loss, preventing global cross-image classification gradients from collapsing image-relative feature clustering.
2. **Image-by-Image Soft Contrastive Loss (`MyInfoNCE`)**:
   - Supervise L2-normalized dense embeddings `feats_norm = F.normalize(feats, p=2, dim=1)` on a **per-image** basis against downsampled ground-truth masks (`F.interpolate` / `avg_pool2d` to `[H', W']`):
     - Same-label pixels within image $i$ (tampered–tampered, pristine–pristine) act as positive pairs; cross-label pixels within image $i$ (tampered–pristine) act as negative pairs, scaled by temperature $\tau$ (aligned with `cluster_temperature`).
     - Cross-image pixel pairs **must not** be contrasted against each other, preserving FOCAL's relative definition of forgery.
   - **Edge-Case Stability**:
     - Images in a batch that are 100% pristine (no tampered pixels) or 100% tampered (no pristine pixels) contain no valid cross-class pairs and **must** contribute `0.0` contrastive loss safely without producing `NaN` or `Inf`.
     - Subsampling / strided pixel sampling **must** be supported for high-resolution feature maps (e.g., `256×256` spatial tokens at `1024×1024` HRNet `H/4`) to bound pairwise similarity matrix memory to $O(K^2)$ instead of $O((H'W')^2)$.
3. **Trainer & Loss Integration**:
   - Expose the computed contrastive loss via `res["align_loss"]` in `FOCAL.forward()` (or equivalent hook compatible with `SIDTotalLoss` at `sid_unet/losses/auxiliary.py`, lines 113–170) and allow `Trainer._step_batch_train()` (`sid_unet/training/trainer.py`, lines 1061–1064) to forward `gt_masks` to `FOCAL` during training without breaking models that do not accept `gt_masks`.
   - When operating in pure contrastive mode (`use_clustering: true`), `SIDTotalLoss` must allow skipping supervised `mask_loss` on uncalibrated `self.fc` logits (or using differentiable cluster logits) so `self.fc` does not fight the contrastive objective.

### 3.3 Requirement 3: Pristine-Image Safety for Clustering Inference
To resolve Critical Error 3 whenever `_cluster_features()` is active (Mode 1B and zero-shot eval):
1. **Cluster Separability / Pristine Rejection Gate**:
   - `_cluster_features()` **must** expose or apply a cluster-separability signal based on inter-centroid cosine similarity $\langle c_0, c_1 \rangle$ (or distance $d = 1 - \langle c_0, c_1 \rangle$) and/or optional `self.class_head` (`aux_classifier: true`).
   - When $d$ falls below a configurable separability threshold $\delta_{\text{sep}}$ (indicating unimodal / uniform forensic features typical of an unmanipulated image), predicted tampering logits **must** be attenuated/shifted negative so pristine images do not produce forced minority-cluster false alarms.
2. **Evaluation Reporting Consistency**:
   - Validation and cross-evaluation reports **must** clearly document whether `filter_zero_positive` is `true` (tampered-only foreground metrics) or `false` (full-dataset including pristine images), alongside `macro_f1` and `auroc`.

### 3.4 Requirement 4: Zero-Fallback Policy & Backward Compatibility
1. **Strict Pretrained Weight Policy**:
   - Retain `FOCAL`'s zero-fallback policy (`pretrained=False` raises `ValueError`; missing/corrupt weights from `KhangTruong/FOCAL` raise `RuntimeError`).
2. **Existing Tests**:
   - Existing unit tests in `tests/test_focal_pim_mismatch_audit.py` **must** continue to pass (specifically verifying that default zero-shot eval breaks 100% positive saturation, auto-denormalization handles `[0, 1]` and ImageNet inputs, and clustering metrics remain consistent).

---

## 4. Acceptance Criteria (Verification Checklist for Implementation PR)

An implementation PR fulfilling this specification will be considered complete when all of the following pass:
1. **Config & Preprocessing**: `focal_hrnet.yaml` and `focal_vit.yaml` explicitly configure `data.normalize: false` and aligned non-256 resolution without hidden `256→1024` ViT upsampling waste.
2. **Train/Eval Parity**:
   - In Mode 1A (`use_clustering: false`), `train()` and `eval()` execute the exact same supervised head, and validation IoU/Dice improves monotonically without collapsing to clustering mismatch.
   - In Mode 1B (`use_clustering: true`), `self.fc` is excluded from training loss, per-image Soft Contrastive Learning (`MyInfoNCE`) runs cleanly with bounded memory and zero `NaN`/`Inf` on pure-pristine or pure-tampered images, and validation evaluates `_cluster_features()`.
3. **Pristine Image Handling**: Pure-pristine test tensors/batches gated by cluster separability (or `aux_classifier`) no longer force >0% positive hallucinated masks when separability is below threshold.
4. **Regression Suite**: `pytest tests/test_focal_pim_mismatch_audit.py` passes alongside new unit tests covering Mode 1A parity, Mode 1B `MyInfoNCE` gradient flow & edge cases, resolution/normalization alignment, and pristine-image rejection.

---

## 5. Out of Scope (Deferred to Later Strategy PRs)
The following higher-complexity model surgeries are intentionally excluded from Strategy 1 and reserved for follow-up specifications:
- **Strategy 2**: PEFT (`SurgicalLoRALinear` / DoRA) injection into `FOCALViTNet` attention/MLP blocks.
- **Strategy 3**: Dual-backbone fusion (`HRNet` + `FOCALViTNet` L2 concatenation / `ZeroGatedFiLM` learned fusion).
- **Strategy 4**: End-to-end differentiable soft-EM clustering head.
- **Strategy 5**: Multi-scale HRNet stage-4 branch fusion / intermediate ViT global block tapping & `ForensicVectorMemory` integration for generative AI / diffusion forgery detection.
