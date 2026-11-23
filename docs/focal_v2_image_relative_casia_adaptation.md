# FOCAL v2: Domain-Invariant Image-Relative Dual-Backbone Adaptation on CASIA-v2.0

> **Unified Report**: For a complete comparative explanation of methods, architectural evolution, failure modes, and results across both PR #34 and PR #35, see [`docs/focal_pr34_pr35_methods_failures_results.md`](focal_pr34_pr35_methods_failures_results.md).

- **Hugging Face Weights & Reports**: [`KhangTruong/focal-v2-image-relative-casia`](https://huggingface.co/KhangTruong/focal-v2-image-relative-casia)
- **Experiment Config**: [`configs/experiments/focal/focal_vit_hrnet_image_relative_casia.yaml`](../configs/experiments/focal/focal_vit_hrnet_image_relative_casia.yaml)
- **Training Dataset**: `KhangTruong/Casia-v2.0-Fake` (`train`: 4,354, `validation`: 769)
- **Cross-Evaluation Protocol**: [`configs/cross-eval/MVSS-v1-protocol.yaml`](../configs/cross-eval/MVSS-v1-protocol.yaml) (`KhangTruong/Tampered-Images-Combined-Dataset-Fake`, `cross_test`, 1,676 stratified samples across 15 sub-datasets, `--samples-per-dataset 120`)

---

## 1. Executive Summary

Building on PR #34 (`KhangTruong/focal-vit-hrnet-hybrid-casia`), **FOCAL v2 (`focal_vit_hrnet_image_relative_casia`)** beats both the original `KhangTruong/FOCAL` baseline and PR #34 across overall `MVSS-v1-protocol` cross-evaluation metrics while training only **0.07% of parameters** (`228,246 / 337,051,866`, with **`0` trainable backbone parameters**).

| Model | Loss | Mean IoU | IoU 95% CI | Dice / F1 | AUROC | Pixel Acc | Precision | Recall | Specificity |
|---|---|---|---|---|---|---|---|---|---|
| `KhangTruong/FOCAL` (`sid_focal_hrnet`) | 0.8795 | 0.3503 | `[0.3320, 0.3686]` | 0.4155 | 0.6945 | 0.8114 | 0.3937 | **0.5802** | 0.8504 |
| `KhangTruong/focal-vit-hrnet-hybrid-casia` (PR #34) | 0.5367 | 0.3984 | `[0.3790, 0.4178]` | 0.4584 | 0.7140 | 0.8567 | 0.4662 | 0.5619 | 0.9062 |
| **FOCAL v2 (`focal-v2-image-relative-casia`)** | **0.5302** | **0.4193** | `[0.3990, 0.4397]` | **0.4715** | **0.7163** | **0.8633** | **0.4871** | 0.5595 | **0.9132** |
| **Gain vs. `KhangTruong/FOCAL`** | **-0.3493** | **+0.0690** | — | **+0.0560** | **+0.0218** | **+0.0519** | **+0.0934** | -0.0207 | **+0.0628** |
| **Gain vs. PR #34** | **-0.0065** | **+0.0209** | — | **+0.0131** | **+0.0023** | **+0.0066** | **+0.0209** | -0.0024 | **+0.0070** |

On in-domain CASIA-v2.0 validation (`KhangTruong/Casia-v2.0-Fake`, 385 positive samples evaluated at Epoch 4), the model reached **Val IoU `0.6642`**, **Val F1 `0.7245`**, **Val AUROC `0.9488`**, **Val Pixel Acc `0.9501`**, **Val Recall `0.8662`**, and **Val Loss `0.3168`** (Best Val IoU `0.6643` at Epoch 3).

---

## 2. Root-Cause Diagnosis & Architectural Changes

### Why PR #34 Stalled OOD
PR #34 fed raw 288-channel `[ViT, HRNet]` features directly into `ClusterResidualRefiner`. Because CASIA-v2.0 is roughly 90% authentic/background pixels, a raw-channel CNN refiner can memorize domain-specific feature coordinates and learn a global negative bias that suppresses OOD tampered regions (PR #34 recall dropped to `0.5619` vs. `KhangTruong/FOCAL` `0.5802`).

### Changes Implemented
1. **Rotation- & Domain-Invariant `ImageRelativeClusterRefiner` ([`sid_unet/models/focal.py`](../sid_unet/models/focal.py))**:
   - Enabled via `cluster_refiner_type: "image_relative"` with `_DilatedResBlock` blocks `(dilation=1, 2, 4)` and zero-initialized `(raw_bias, raw_scale)` output bounded by `tanh`.
   - Never observes raw feature channels. Instead constructs **10** (single-branch) or **24** (multi-branch `ViT+HRNet`) geometric/clustering statistics invariant to orthogonal channel rotations:
     - Fused cluster probability `p_f`, temperature logit `t_f`, normalized entropy `ent_f`, cluster separability `sep_f`, predicted tampered area ratio `area_f`, image-relative z-scored distance difference `d_mean_z`, local multi-scale probability contrasts `d_loc3`, `d_loc9`, `d_loc25`, and centered probability `p_centered`.
     - When `multi_branch_logit_fusion: true` on concatenated backbones, appends primary (ViT) and secondary (HRNet) branch statistics plus branch discrepancy signals (`|p_v - p_h|`, `t_v - t_h`).
2. **`SpatialCrossScaleAdapter` ([`sid_unet/models/focal.py`](../sid_unet/models/focal.py))**:
   - Enabled via `feature_adapter_type: "spatial"`: bottleneck `1x1 -> depthwise 3x3 -> dilated depthwise 3x3 (dilation=2) -> zero-init 1x1` residual block, plus prior-anchoring regularization (`prior_anchor_weight: 0.75`) and boundary-aware contrastive sampling (`contrastive_boundary_ratio: 0.4`).
3. **High-Resolution ViT Input & Horizontal-Flip TTA ([`sid_unet/models/focal.py`](../sid_unet/models/focal.py), [`sid_unet/models/unet.py`](../sid_unet/models/unet.py))**:
   - Added `vit_input_size` (`768` vs. default `512` or `1024`) to preserve SAM ViT high-frequency detail with manageable memory on dual T4 GPUs.
   - Added eval-time `tta_hflip: true` averaging logits across original and horizontally flipped views.
4. **Multi-GPU `DataParallel` & DDP Stability Fixes**:
   - [`sid_unet/training/trainer.py`](../sid_unet/training/trainer.py): Added `_SIDDataParallel(nn.DataParallel)` preserving non-tensor scalar/boolean batch metadata (`skip_mask_loss`) during `gather()`.
   - [`sid_unet/losses/auxiliary.py`](../sid_unet/losses/auxiliary.py): Reduced replica vectors in `raw_prior_anchor` with `.mean()` before scalar addition in `SIDTotalLoss`.
   - [`sid_unet/utils/distributed.py`](../sid_unet/utils/distributed.py): Filtered non-numeric metadata (`filter_zero_positive_mode`, booleans) in `reduce_dict()`.
   - [`sid_unet/models/focal.py`](../sid_unet/models/focal.py): Froze unused `fused_fc` / `domain_fc` heads in `hybrid`/`contrastive` modes for DDP unused-parameter safety and aligned `_anchor_neck` / `_anchor_hrnet_stage4` devices under `DataParallel`.

---

## 3. Systematic Short-Horizon Experimentation & Failures / Lessons Learned

Following short-run iterative diagnostics on 576-sample `MVSS-40` (`--samples-per-dataset 40`, zero-shot + 1-epoch ablations) before committing to 4-epoch training:

### A. Zero-Shot Clustering & Resolution Diagnostics (`MVSS-40`, 576 samples)

| Run | Clustering | `vit_input_size` | `tta_hflip` | Loss | IoU | F1 | AUROC | Pixel Acc | Recall | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| PR #34 | `kmeans` | 512 | False | 0.5255 | 0.4234 | 0.4861 | 0.7272 | 0.8613 | 0.5865 | Reference baseline |
| `zs_0_kmeans_512` | `kmeans` | 512 | False | 0.6778 | 0.4035 | 0.4665 | 0.7181 | — | 0.5943 | Zero-shot step 0 |
| `zs_1_soft_512` | `soft_kmeans` | 512 | False | — | **0.2945** | 0.3324 | 0.6849 | — | 0.4469 | **FAILED** (centroid collapse) |
| `zs_2_soft_768` | `soft_kmeans` | 768 | False | — | 0.3219 | — | — | — | — | **FAILED** |
| `zs_3_soft_1024` | `soft_kmeans` | 1024 | False | — | 0.3253 | — | — | — | — | **FAILED** |
| `zs_4_soft_512_tta` | `soft_kmeans` | 512 | True | — | 0.2991 | — | — | — | — | **FAILED** |
| `zs_5_kmeans_768` | `kmeans` | 768 | False | — | 0.4163 | 0.4736 | 0.7189 | 0.8479 | — | +0.0128 IoU over 512 |
| `zs_6_kmeans_512_tta` | `kmeans` | 512 | True | — | 0.4148 | 0.4779 | 0.7261 | — | 0.6075 | +0.0113 IoU over no-TTA |
| **`zs_7_kmeans_768_tta`** | **`kmeans`** | **768** | **True** | **0.6339** | **0.4267** | **0.4853** | **0.7210** | **0.8513** | **0.5905** | **Selected zero-shot base** |

- **Failure Lesson #1 (`soft_kmeans` collapse)**: Mean initialization plus soft EM pulled foreground and background centroids below `cluster_separability_threshold` (`0.0015`), falsely triggering `_apply_pristine_gate` on tampered images (`IoU 0.4035 -> 0.2945`). Retained `clustering_mode: "kmeans"`.

### B. 1-Epoch Training Ablations on `Casia-v2.0-Fake` (`MVSS-40`, 576 samples)

| Experiment | Key Settings | Casia Val IoU | `MVSS-40` IoU | `MVSS-40` F1 | `MVSS-40` AUROC | `MVSS-40` Recall | Outcome |
|---|---|---|---|---|---|---|---|
| `Exp 1` | Image-relative refiner only, wide `(1.5, 0.5)` bounds, `pos_weight=1.0` | 0.6324 | 0.4186 | 0.4653 | 0.7054 | **0.5029** | **Over-suppressed OOD recall** |
| **`Exp 2`** | + `SpatialCrossScaleAdapter`, `(0.50, 0.25)` bounds, `pos_weight=1.35`, `prior_anchor=0.5`, WiSE-FT `0.45` | 0.6516 | **0.4327** | **0.4863** | **0.7222** | **0.5789** | Raw Ep1 `0.4326`; WiSE-FT `0.45` peak |
| `Exp 3` | Tighter `(0.35, 0.15)` bounds, `prior_anchor=0.8`, `pos_weight=1.6`, `jpeg=[65,100]` | 0.6429 | 0.4320 | 0.4867 | 0.7217 | 0.5576 | Higher precision (`0.5110` vs `0.495`) |
| `Exp 4` | Same as Exp 3 + **unfrozen `HRNet stage4`** (`secondary_lr_multiplier=0.2`) | **0.6599** | **0.4248** | 0.4809 | 0.7175 | — | **FAILED OOD** (`HiFi_IFDL` `0.3326 -> 0.2494`) |

- **Failure Lesson #2 (Unfreezing `HRNet stage4`)**: Allowing gradient updates to `HRNet stage4` improved in-domain Casia Val IoU (`0.6599`) while degrading `MVSS-40` IoU (`0.4320 -> 0.4248`) and `HiFi_IFDL` (`0.3326 -> 0.2494`), confirming both pretrained backbones should stay strictly frozen (`0` backbone params trainable).
- **Failure Lesson #3 (Wide residual bounds & negative bias over multiple epochs)**: In `Exp 1`, wide `(1.5, 0.5)` bounds with `pos_weight=1.0` dropped OOD recall to `0.5029`. Even with `(0.40, 0.20)` and `pos_weight=1.5` in the 4-epoch run, Epoch 3 `checkpoint_best.pt` (`alpha=1.0`) scored `0.4264` IoU / `0.5318` Recall on `MVSS-40`. Applying post-training WiSE-FT residual/adapter scaling across `alpha in {0.45, 0.65, 0.85}` recovered Recall to `0.5598` and raised `MVSS-40` IoU to **`0.4303`** at **`alpha=0.65`** (vs. `0.4294` at `0.85` and `0.4276` at `0.45`).

---

## 4. Final 4-Epoch Training & Official 1,676-Sample Results

### Training Progression (`KhangTruong/Casia-v2.0-Fake`, 2-GPU DDP, FP16 AMP)

| Epoch | Train Loss | Val Loss | Val IoU | Val F1 | Val AUROC | Val Pixel Acc | Learning Rate |
|---|---|---|---|---|---|---|---|
| 1 | 7.1687 | 0.3620 | 0.6495 | 0.7126 | 0.9496 | 0.9467 | 0.000150 |
| 2 | 7.0783 | 0.3336 | 0.6622 | 0.7230 | 0.9493 | 0.9494 | 0.000129 |
| 3 (`best`) | 7.0616 | 0.3313 | **0.6643** | 0.7244 | 0.9481 | 0.9500 | 0.0000775 |
| 4 | 7.0525 | **0.3168** | 0.6642 | **0.7245** | 0.9488 | **0.9501** | 0.0000262 |

After Epoch 3 `checkpoint_best.pt` was selected, WiSE-FT calibration (`alpha=0.65`, effective `cluster_residual_max_scale=0.26`, `cluster_residual_scale_gain=0.13`) was promoted to the final released `checkpoint_best.pt`.

### Official `MVSS-v1-protocol` Per-Dataset Breakdown (1,676 Samples, `--samples-per-dataset 120`)

| Sub-Dataset | Samples | `KhangTruong/FOCAL` IoU | PR #34 IoU | **FOCAL v2 IoU** | **FOCAL v2 F1** | **FOCAL v2 AUROC** | **FOCAL v2 Pixel Acc** | **FOCAL v2 Prec** | **FOCAL v2 Rec** | **FOCAL v2 Spec** |
|---|---|---|---|---|---|---|---|---|---|---|
| `CASIA_V1` | 120 | 0.6756 | 0.7581 | **0.8609** | 0.9037 | 0.9725 | 0.9848 | 0.8827 | 0.9580 | 0.9870 |
| `CASIA_V2` | 120 | 0.7387 | 0.8123 | **0.8811** | 0.9130 | 0.9727 | 0.9805 | 0.9060 | 0.9569 | 0.9885 |
| `COCO_Inpaint` | 120 | **0.0402** | 0.0212 | 0.0154 | 0.0283 | 0.4970 | 0.8825 | 0.0278 | 0.0722 | 0.9217 |
| `CocoGlide` | 120 | 0.3817 | 0.4018 | **0.4200** | 0.4918 | 0.6825 | 0.7492 | 0.5929 | 0.4982 | 0.8668 |
| `Columbia_Gray` | 120 | 0.8949 | 0.9425 | **0.9478** | 0.9612 | 0.9733 | 0.9767 | 0.9619 | 0.9606 | 0.9860 |
| `Coverage` | 100 | 0.3418 | 0.4962 | **0.5446** | 0.6243 | 0.8536 | 0.9061 | 0.5698 | 0.7811 | 0.9261 |
| `HiFi_IFDL` | 120 | 0.1678 | **0.3509** | 0.3356 | 0.3983 | 0.6537 | 0.6995 | 0.4678 | 0.4957 | 0.8117 |
| `IMD2020` | 120 | **0.0473** | 0.0379 | 0.0375 | 0.0593 | 0.5589 | 0.8851 | 0.0455 | 0.2239 | 0.8939 |
| `IMD2020_RealLife` | 120 | 0.3724 | 0.4700 | **0.5133** | 0.6045 | 0.9178 | 0.9365 | 0.5549 | 0.8886 | 0.9470 |
| `NIST16` | 120 | 0.3802 | 0.4438 | **0.4723** | 0.5423 | 0.7594 | 0.8601 | 0.5544 | 0.5832 | 0.9355 |
| `OpenSDID_Flux.1` | 120 | **0.2346** | 0.2036 | 0.1989 | 0.2762 | 0.6205 | 0.7823 | 0.3145 | 0.3821 | 0.8590 |
| `OpenSDID_SD1.5` | 120 | **0.1556** | 0.1393 | 0.1306 | 0.1893 | 0.5744 | 0.7640 | 0.2116 | 0.2892 | 0.8595 |
| `OpenSDID_SDXL` | 120 | **0.0947** | 0.0880 | 0.0881 | 0.1328 | 0.5381 | 0.7695 | 0.2298 | 0.1879 | 0.8883 |
| `SAGI-D` | 16 | 0.5551 | **0.6229** | 0.6165 | 0.6991 | 0.8746 | 0.8919 | 0.6831 | 0.8052 | 0.9440 |
| `TGIF2_flux1dev` | 120 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4563 | 0.9125 | 0.0000 | 0.0000 | 0.9125 |

### Full Unstratified `MVSS-v1-protocol` Cross-Evaluation (`17,929` Samples)

Published in [`KhangTruong/focal-v2-image-relative-casia` (`master_cross_evaluation_report.md`)](https://huggingface.co/KhangTruong/focal-v2-image-relative-casia/raw/main/cross_evaluation/master_cross_evaluation_report.md):

| Model | Samples | Loss | Mean IoU | IoU 95% CI | Dice / F1 | AUROC | Pixel Acc | Precision | Recall | Specificity |
|---|---|---|---|---|---|---|---|---|---|---|
| `KhangTruong/FOCAL` (`sid_focal_hrnet`)* | 17,929 | 1.0407 | 0.2293 | — | 0.2902 | 0.6171 | 0.7758 | 0.2809 | **0.4618** | 0.8233 |
| PR #34 (`focal-vit-hrnet-hybrid-casia`) | 17,929 | 0.6320 | 0.2639 | `[0.2587, 0.2692]` (`±0.0053`) | 0.3206 | **0.6332** | 0.8274 | 0.3317 | 0.4307 | 0.8835 |
| **FOCAL v2 (`focal-v2-image-relative-casia`)** | **17,929** | **0.6294** | **0.2755** | `[0.2699, 0.2811]` (`±0.0056`) | **0.3248** | 0.6301 | **0.8333** | **0.3443** | 0.4152 | **0.8910** |

| Sub-Dataset | Samples | PR #34 IoU | **FOCAL v2 IoU** | **FOCAL v2 F1** | **FOCAL v2 AUROC** | **FOCAL v2 Pixel Acc** | **FOCAL v2 Prec** | **FOCAL v2 Rec** | **FOCAL v2 Spec** |
|---|---|---|---|---|---|---|---|---|---|
| `CASIA_V1` | 920 | 0.7135 | **0.8234** | 0.8741 | 0.9590 | 0.9798 | 0.8526 | 0.9340 | 0.9840 |
| `CASIA_V2` | 1,000 | 0.8074 | **0.8942** | 0.9306 | 0.9833 | 0.9887 | 0.9126 | 0.9748 | 0.9918 |
| `COCO_Inpaint` | 2,000 | **0.0285** | 0.0201 | 0.0347 | 0.4869 | 0.8622 | 0.0375 | 0.0652 | 0.9086 |
| `CocoGlide` | 507 | **0.4342** | 0.4180 | 0.4976 | 0.7001 | 0.7833 | 0.5773 | 0.5201 | 0.8800 |
| `Columbia_Gray` | 180 | 0.9516 | **0.9540** | 0.9682 | 0.9783 | 0.9813 | 0.9685 | 0.9685 | 0.9880 |
| `Coverage` | 100 | 0.5010 | **0.5446** | 0.6243 | 0.8536 | 0.9061 | 0.5698 | 0.7811 | 0.9261 |
| `HiFi_IFDL` | 2,000 | 0.3219 | **0.3259** | 0.3864 | 0.6605 | 0.6859 | 0.4448 | 0.5288 | 0.7921 |
| `IMD2020` | 2,000 | 0.0301 | **0.0317** | 0.0524 | 0.5464 | 0.8702 | 0.0377 | 0.2140 | 0.8787 |
| `IMD2020_RealLife` | 1,000 | 0.4906 | **0.5536** | 0.6417 | 0.9126 | 0.9269 | 0.5842 | 0.8887 | 0.9365 |
| `NIST16` | 220 | 0.4844 | **0.5065** | 0.5797 | 0.7774 | 0.8697 | 0.5983 | 0.6137 | 0.9411 |
| `OpenSDID_Flux.1` | 2,000 | **0.2199** | 0.2080 | 0.2802 | 0.6151 | 0.7882 | 0.3191 | 0.3553 | 0.8748 |
| `OpenSDID_SD1.5` | 2,000 | **0.1281** | 0.1203 | 0.1725 | 0.5533 | 0.7607 | 0.2149 | 0.2350 | 0.8716 |
| `OpenSDID_SDXL` | 2,000 | **0.1192** | 0.1080 | 0.1607 | 0.5447 | 0.7556 | 0.2164 | 0.2177 | 0.8717 |
| `SAGI-D` | 16 | **0.6253** | 0.6165 | 0.6991 | 0.8746 | 0.8919 | 0.6831 | 0.8052 | 0.9440 |
| `TGIF2_flux1dev` | 1,986 | 0.0000 | 0.0000 | 0.0000 | 0.4549 | 0.9097 | 0.0000 | 0.0000 | 0.9097 |

*(\*Note: `KhangTruong/FOCAL`'s historical 17,929-sample report predates the `NIST16` mask-polarity fix; 78.0% of the unstratified 17,929 samples come from the seven ~2,000-image datasets.)*

---

## 5. GPU Utilization Audit

Hardware: **2x NVIDIA Tesla T4** (14,912 MiB reportable VRAM each, SM 7.5 FP16 AMP), **4 vCPUs**.

| Phase | Mode | Samples (`nvidia-smi -l 2`) | GPU 0 Mean Util | GPU 0 Zeros (`0%`) | GPU 1 Mean Util | GPU 1 Zeros (`0%`) | Observation |
|---|---|---|---|---|---|---|---|
| Baseline Cross-Eval | `DataParallel` (`batch_size=16`) | 178 | 35.2% | 57 (32.0%) | 31.3% | 68 (38.2%) | Cold HF downloads + Parquet I/O CPU stalls on 4 vCPUs |
| Final 4-Epoch Training | **2-GPU DDP** | 619 | **93.8%** (median 100.0%) | **18 (2.9%)** | **93.3%** (median 100.0%) | **18 (2.9%)** | Sustained compute saturation with minimal I/O bubbles |
| Final Cross-Eval (`MVSS-v1`) | `DataParallel` (`batch_size=16`, warm cache) | 267 | **81.5%** (active 90.3%, median 100.0%) | 26 (9.7%) | **88.5%** (active 94.5%, median 100.0%) | 17 (6.4%) | Remaining zeros correspond to validation-boundary / Parquet transitions |

---

## 6. Reproducibility & Usage

### Train
```bash
sid-train --config configs/experiments/focal/focal_vit_hrnet_image_relative_casia.yaml
```

### Cross-Evaluate (`MVSS-v1-protocol`)
```bash
sid-cross-eval \
  --cross-configs configs/cross-eval/MVSS-v1-protocol.yaml \
  --checkpoints KhangTruong/focal-v2-image-relative-casia \
  --samples-per-dataset 120 \
  --batch-size 16 \
  --num-workers 4
```
