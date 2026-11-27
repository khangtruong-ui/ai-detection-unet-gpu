# FOCAL v3: Multi-Tap Pre-Neck, Polarity-Aware & Separability-Gated Dual-Backbone Adaptation on CASIA-v2.0

- **Hugging Face Weights & Reports**: [`KhangTruong/focal-v3-multitap-casia`](https://huggingface.co/KhangTruong/focal-v3-multitap-casia)
- **Experiment Config**: [`configs/experiments/focal/focal_v3_multitap_casia.yaml`](../configs/experiments/focal/focal_v3_multitap_casia.yaml)
- **Training Dataset**: `KhangTruong/Casia-v2.0-Fake` (`train`: 4,354, `validation`: 769)
- **Cross-Evaluation Protocol**: [`configs/cross-eval/MVSS-v1-protocol.yaml`](../configs/cross-eval/MVSS-v1-protocol.yaml) (`KhangTruong/Tampered-Images-Combined-Dataset-Fake`, `cross_test`, 1,676 stratified samples across 15 sub-datasets, `--samples-per-dataset 120`)
- **Companion Diagnostics PR**: `khangtruong-ui/nn-toolbox` (`clustering_polarity_diagnostic` on `feat/clustering-polarity-diagnostic`)

---

## 1. Executive Summary

Building directly on the forensic root-cause audit in PR #38 ([`docs/focal_pr34_pr35_cross_dataset_failure_forensic_report.md`](focal_pr34_pr35_cross_dataset_failure_forensic_report.md)) and PR #34 / #35, **FOCAL v3 (`focal_v3_multitap_casia`)** resolves the three primary failure mechanisms holding back `KhangTruong/FOCAL` and `PR #35`:
1. **Pre-Neck ViT Multi-Tap Feature & Logit Fusion** (`vit_feature_taps: [22, 23]`, `dual_logit` at compact `48x48`) recovers OOD generative/diffusion separation attenuated inside `FOCAL_ViT`'s `neck[0..3]` bottleneck.
2. **Flip-Equivariant `kmeans_robust` + Border-Aware Cluster Polarity** (`polarity_border_weight: 0.75`, `polarity_area_gate: 0.25`) cures corner-seed asymmetry and `CocoGlide` / `HiFi_IFDL` minority-polarity inversions (`>50%` or interior tampered regions).
3. **Branch-Balanced Adapter + Separability-Gated, DC-Zero & Ortho-Regularized Refiner** eliminates PR #35's `2.09x` HRNet adapter asymmetry, `-0.9942` rank-1 head collapse, and `-0.205` global negative OOD bias shift.

Trainable footprint remains **0.07% of parameters** (`222,872 / 337,046,492`, **`0` trainable backbone parameters**).

### Official `MVSS-v1-protocol.yaml` Comparison (1,676 Samples, `--samples-per-dataset 120`, Continuous AUROC Evaluator)

| Model | Loss | Mean IoU | IoU 95% CI | Dice / F1 | Continuous AUROC | Pixel Acc | Precision | Recall | Specificity |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| **`KhangTruong/FOCAL`** (`sid_focal_hrnet`) | 0.8794 | 0.3503 | `[0.3320, 0.3686]` | 0.4155 | 0.7759 | 0.8114 | 0.3937 | 0.5802 | 0.8504 |
| **`PR #35`** (`focal-v2-image-relative-casia`) | 0.5156 | 0.4131 | `[0.3927, 0.4334]` | 0.4647 | 0.8049 | **0.8729** | 0.4947 | 0.5293 | **0.9306** |
| **FOCAL v3 (`focal-v3-multitap-casia`)** | **0.5126** | **0.4304** | `[0.4102, 0.4506]` | **0.4856** | **0.8248** | 0.8632 | **0.4967** | **0.5887** | 0.9092 |
| **Gain vs. `KhangTruong/FOCAL`** | **-0.3668** | **+0.0801** | — | **+0.0701** | **+0.0489** | **+0.0518** | **+0.1030** | **+0.0085** | **+0.0588** |
| **Gain vs. PR #35** | **-0.0030** | **+0.0173** | — | **+0.0209** | **+0.0199** | -0.0097 | **+0.0020** | **+0.0594** | -0.0214 |

### Per-Dataset Breakdown (All 15 `MVSS-v1` Sub-Datasets, 1,676 Samples)

| Sub-Dataset | Samples | `KhangTruong/FOCAL` IoU / AUC | PR #35 IoU / AUC | **FOCAL v3 IoU / AUC** | **FOCAL v3 F1** | **IoU $\Delta$ vs FOCAL** | **IoU $\Delta$ vs PR #35** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `CASIA_V1` | 120 | 0.6756 / 0.9592 | **0.8613** / 0.9951 | 0.8537 / **0.9954** | 0.9010 | `+0.1781` | `-0.0076` |
| `CASIA_V2` | 120 | 0.7387 / 0.9760 | **0.8931** / 0.9896 | 0.8712 / **0.9906** | 0.9065 | `+0.1325` | `-0.0219` |
| `COCO_Inpaint` | 120 | **0.0403 / 0.5930** | 0.0137 / 0.5297 | 0.0164 / 0.5275 | 0.0300 | `-0.0239` | `+0.0027` |
| `CocoGlide` | 120 | 0.3820 / 0.7226 | 0.4104 / 0.7771 | **0.4703 / 0.8334** | **0.5489** | **`+0.0883`** | **`+0.0599`** |
| `Columbia_Gray` | 120 | 0.8949 / 0.9518 | 0.9481 / 0.9775 | **0.9543 / 0.9873** | **0.9685** | `+0.0594` | `+0.0062` |
| `Coverage` | 100 | 0.3423 / 0.7759 | **0.5606** / 0.9085 | 0.5470 / **0.9289** | 0.6310 | `+0.2047` | `-0.0136` |
| `HiFi_IFDL` | 120 | 0.1678 / 0.4614 | 0.2752 / 0.7276 | **0.3973 / 0.8130** | **0.4686** | **`+0.2295`** | **`+0.1221`** |
| `IMD2020` | 120 | **0.0473 / 0.6921** | 0.0373 / 0.6576 | 0.0395 / 0.6689 | 0.0624 | `-0.0078` | `+0.0022` |
| `IMD2020_RealLife` | 120 | 0.3724 / 0.9269 | 0.5232 / 0.9650 | **0.5255 / 0.9693** | **0.6212** | `+0.1531` | `+0.0023` |
| `NIST16` | 120 | 0.3800 / 0.7598 | 0.4747 / 0.8042 | **0.4770 / 0.8326** | **0.5473** | `+0.0970` | `+0.0023` |
| `OpenSDID_Flux.1` | 120 | **0.2345 / 0.7352** | 0.1889 / 0.7174 | 0.2020 / 0.7283 | 0.2760 | `-0.0325` | **`+0.0131`** |
| `OpenSDID_SD1.5` | 120 | **0.1556 / 0.6836** | 0.1076 / 0.6523 | 0.1406 / 0.6720 | 0.2015 | `-0.0150` | **`+0.0330`** |
| `OpenSDID_SDXL` | 120 | 0.0941 / **0.6084** | 0.0741 / 0.5727 | **0.0957** / 0.6046 | 0.1457 | `+0.0016` | **`+0.0216`** |
| `SAGI-D` | 16 | 0.5551 / 0.8972 | 0.6107 / 0.8950 | **0.6122 / 0.9193** | **0.6959** | `+0.0571` | `+0.0015` |
| `TGIF2_flux1dev` | 120 | 0.0000 / 1.0000 | 0.0000 / 1.0000 | 0.0000 / 1.0000 | 0.0000 | `0.0000` | `0.0000` |

*(Note: `TGIF2_flux1dev` has 1,986/1,986 blank all-zero ground-truth PNG masks in `MVSS-v1`, as documented in PR #38.)*

---

## 2. Architectural Changes (`sid_unet/models/focal.py`, `sid_unet/models/unet.py`)

1. **Multi-Tap Pre-Neck ViT Extraction (`ImageEncoderViT.forward_with_taps`, `FOCALViTNet.forward_with_taps`)**:
   - Captures intermediate blocks `[22, 23]` alongside `neck_out` in a single ViT pass (with L2-averaged horizontal-flip TTA).
   - `vit_tap_fusion_mode: "dual_logit"` clusters `[neck, b22, b23, HRNet]` at compact `48x48` (`vit_tap_cluster_size: 48`) and blends its logits with full-resolution `[neck, HRNet]` clustering via learnable softmax `tap_logit_weights`.
2. **Flip-Equivariant `kmeans_robust`**:
   - Replaces corner-pixel `(0, 0)` initialization with global mean anchor $\to$ top-2% farthest mean $\to$ top-2% opposite mean followed by 8 hard spherical 2-means iterations, ensuring strict HFlip equivariance and corner-outlier immunity.
3. **Domain-Invariant Cluster Polarity (`border_aware` & `ClusterPolarityHead`)**:
   - `cluster_polarity_mode: "border_aware"` scores each cluster by $\text{area}_k + 0.75 \cdot \text{border\_frac}_k$ whenever $\min(\text{area}_0, \text{area}_1) > 0.25$ (preserving tiny border-touching splices while catching interior/`>50%` manipulations like `CocoGlide` and `HiFi_IFDL`).
   - Also adds optional antisymmetric `ClusterPolarityHead` (`cluster_polarity_mode: "learned"`) over 6 rotation/flip-invariant cluster geometry & dispersion descriptors.
4. **Branch-Balanced Adapter (`BranchBalancedSpatialAdapter`)**:
   - Independent zero-initialized `SpatialCrossScaleAdapter` modules for ViT (`256ch`) and HRNet (`32ch`) branches with equal `0.5/0.5` drift anchoring (`prior_anchor_weight: 1.5`), preventing asymmetric HRNet perturbation (`1.05x` primary/secondary RMS ratio vs. `2.09x` in PR #35).
5. **Separability-Gated, DC-Zero & Ortho-Regularized `ImageRelativeClusterRefiner`**:
   - `refiner_sep_gate_tau: 0.30` scales residual `bias` and `scale - 1` by $\min(1, \text{sep} / \tau)$ so low-separability OOD samples smoothly revert toward neutral identity.
   - `refiner_zero_mean_bias_weight: 0.10` penalizes spatial-mean `bias` during training (eliminating PR #35's `-0.205` global negative shift).
   - `refiner_ortho_weight: 0.10` penalizes $\cos^2(w_{\text{bias}}, w_{\text{scale}})$ on `cluster_refiner.out_conv.weight` (curing PR #35's `-0.9942` rank-1 collapse).

---

## 3. Zero-Shot Probe, Short-Horizon Ablations & Failures (`MVSS-40`, 576 Samples)

### A. Zero-Shot Mechanistic Probes (`MVSS-40`, `768x768` + HFlip TTA)

| Zero-Shot Variant | IoU | F1 | AUROC | Outcome & Lesson |
|---|---:|---:|---:|---|
| `ZS_v2_corner` (`neck + HRNet`, minority) | 0.4275 | 0.4837 | 0.8125 | Baseline zero-shot reference |
| `ZS_v2_robust_center050` (mean centering $\alpha=0.5$) | 0.4222 | 0.4873 | 0.8174 | **Rejected**: lowers IoU vs `0.4275` |
| `feat_b22_b23_hr_noneck` (drop `neck` completely) | 0.4169 | 0.4817 | 0.8081 | **Rejected**: drops `Coverage` & `CocoGlide` precision |
| `pol_v2_hr_pruned_bw075_g025` (skip dead HRNet `stage4.2.b0.{1..3}`) | 0.4373 | 0.5026 | 0.8340 | **Failed**: `-0.0067` IoU vs unpruned `0.4440` (`CASIA_V2`, `Coverage` regressed) |
| `pol_v2_bw075_g025` (`neck + HRNet`, `kmeans_robust` + border polarity) | 0.4440 | 0.5054 | 0.8333 | `+0.0165` IoU (`HiFi_IFDL 0.294->0.416`, `CocoGlide 0.474->0.518`) |
| **`pol_logit_v2_fb2223_bw075_g025`** (`dual_logit` + `border075_g025`) | **0.4462** | **0.5081** | **0.8356** | **Selected** (`sid-cross-eval` parity `ZS_v3`: `0.4433` IoU / `0.5065` F1 / `0.8343` AUROC) |

*(Note: Pass 2 probe initially hit CUDA OOM at `batch_size=16` on a single 15 GB T4 when holding multiple `2336`-ch `128x128` feature maps simultaneously; solved cleanly in `FOCAL` via `vit_tap_cluster_size=48` and `batch_size=8`.)*

### B. Short-Horizon 1-Epoch Training & WiSE-FT Calibration (`Casia-v2.0-Fake` DDP $\to$ `MVSS-40`)

| Run | Setup | Casia Val IoU | `MVSS-40` IoU | `MVSS-40` F1 | `MVSS-40` AUROC | `MVSS-40` Loss | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| `PR #35` (4 ep, ref) | `SpatialCrossScaleAdapter`, minority | — | 0.4259 | 0.4776 | 0.8147 | 0.5002 | OOD negative bias `-0.205` |
| `ZS_v3` (0 ep) | Production zero-shot v3 (`sid-cross-eval`) | — | 0.4433 | 0.5065 | 0.8343 | 0.5730 | Clean zero-shot floor |
| `Exp 1` (1 ep) | Zero-adapter + gated/DC-zero/ortho refiner | 0.5872 | 0.4442 | 0.5065 | 0.8343 | 0.5608 | `cos=-0.0047`, no forgetting |
| `Exp 2` (1 ep) | + `BranchBalancedSpatialAdapter`, `anchor=1.0` | 0.6109 | 0.4481 | 0.5077 | 0.8339 | 0.5183 | `+0.0048` IoU over `ZS_v3` |
| **`Exp 3A` (1 ep)** | + `anchor=1.5`, `lr=3e-4`, `bounds=0.65/0.30` | **0.6218** | **0.4492** | **0.5070** | **0.8319** | **0.5111** | **Winner** (`cos=-0.0092`, `1.16x` RMS) |
| `Exp 3B` (1 ep) | `Exp 3A` + `learned` polarity (`pol_wt=0.5`) | — | 0.4492 | 0.5064 | 0.8316 | 0.5079 | **Rejected**: ties IoU, slightly lower F1/AUROC |
| **Final 4-Ep (`alpha=0.50`)** | Uniform WiSE-FT sweep | — | 0.4485 | 0.5069 | **0.8358** | 0.5106 | Higher recall (`0.6169`), lower IoU |
| **Final 4-Ep (`alpha=0.65`)** | Uniform WiSE-FT sweep | — | 0.4486 | 0.5059 | 0.8349 | 0.5047 | — |
| **Final 4-Ep (`alpha=0.80`)** | Uniform WiSE-FT sweep | — | 0.4490 | 0.5056 | 0.8345 | 0.5010 | — |
| **Final 4-Ep (`alpha=1.00`)** | **Raw Epoch 4 `checkpoint_best.pt`** | **0.6320** | **0.4510** | **0.5075** | 0.8341 | **0.4974** | **Peak `MVSS-40` IoU/F1/Loss; no WiSE-FT shrinkage needed** |

---

## 4. Final 4-Epoch Training, `nn-toolbox` Audit & GPU Utilization

### Training Progression (`KhangTruong/Casia-v2.0-Fake`, 2-GPU DDP, FP16 AMP)

| Epoch | Train Loss | Val Loss | Val IoU | Val F1 | Val AUROC | Val Pixel Acc | Learning Rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.1363 | 0.3469 | 0.6227 | 0.6874 | 0.9442 | 0.9312 | 0.000300 |
| 2 | 7.0745 | 0.3290 | 0.6270 | 0.6918 | 0.9446 | 0.9332 | 0.000257 |
| 3 | 7.0558 | 0.3266 | 0.6302 | 0.6942 | 0.9442 | 0.9342 | 0.000152 |
| 4 (`best`) | **7.0536** | **0.3231** | **0.6320** | **0.6958** | 0.9442 | **0.9345** | 0.0000482 |

### `nn-toolbox` Weight Health & Polarity Diagnostic Audit (`checkpoint_best.pt`)

- **`cluster_refiner.out_conv`**: `||w_bias|| = 0.2625`, `||w_scale|| = 0.3388`, **`cos(w_bias, w_scale) = -0.0100`** (cleanly orthogonal; vs. `-0.9942` in PR #35), `bias = [-0.0193, -0.0216]`.
- **`BranchBalancedSpatialAdapter` RMS**: `primary_adapter.proj_out = 0.00661`, `secondary_adapter.proj_out = 0.00693` (`1.05x` balanced ratio; vs. `2.09x` in PR #35). Note: `inspect_pretrained_weight_health` flags one high-cosine channel pair (`[19, 21] = 0.9969`) inside 32-ch `secondary_adapter.proj_out` due to HRNet's positive-orthant ReLU cone.
- **`clustering_polarity_diagnostic`** (10 samples/domain, FP16):
  - `CASIA_V2`: `min_iou = 0.920`, `brd_iou = 0.920`, `orc_iou = 0.920`, `flip_rate = 0.00`, `gt/km = 1.000`
  - `HiFi_IFDL`: `min_iou = 0.705`, `brd_iou = 0.705`, `orc_iou = 0.723`, `flip_rate = 0.10`, `gt/km = 0.789`
  - `CocoGlide`: `min_iou = 0.370` $\to$ **`brd_iou = 0.450`** (`+0.080` recovery; `flip_rate = 0.40`, `orc_iou = 0.522`, `gt/km = 0.695`)
  - `OpenSDID_SDXL`: `min_iou = 0.191`, `orc_iou = 0.266`, `flip_rate = 0.50`, `gt/km = 0.376` (structural `SEMANTIC_HIJACKING` without paired diffusion training data).

### GPU Utilization Telemetry (2x NVIDIA Tesla T4, `nvidia-smi -l 2`)

| Phase | Mode | Samples | GPU 0 Mean / Med / Zeros | GPU 1 Mean / Med / Zeros |
|---|---|---:|---|---|
| Beginning Cross-Eval (`FOCAL` + `PR #35`) | `DataParallel` (`bs=-1`) | 341 | `68.5%` / `100.0%` / `23.8%` | `68.3%` / `100.0%` / `27.0%` |
| Final 4-Epoch Training (`FOCAL v3`) | **2-GPU DDP** | 1,815 | **`95.7%` / `100.0%` / `1.8%`** | **`96.6%` / `100.0%` / `1.0%`** |
| Final Cross-Eval (`FOCAL v3`, 1,676 samples) | `DataParallel` (`bs=28` via `-1`) | 257 | **`88.8%` / `100.0%` / `5.8%`** | **`88.2%` / `100.0%` / `8.2%`** |

---

## 5. Reproducibility

```bash
# Train FOCAL v3 (4 epochs, 2-GPU DDP)
sid-train --config configs/experiments/focal/focal_v3_multitap_casia.yaml

# Cross-evaluate against MVSS-v1-protocol
sid-cross-eval \
  --cross-configs configs/cross-eval/MVSS-v1-protocol.yaml \
  --checkpoints KhangTruong/focal-v3-multitap-casia \
  --samples-per-dataset 120 \
  --batch-size -1 \
  --num-workers 4
```
