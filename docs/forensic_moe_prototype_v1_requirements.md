# Requirements & Architecture Specification: Lightweight 3-Expert `ForensicMoE` Prototype v1

**Author**: Antigravity Assistant  
**Date**: November 18, 2026  
**Target Repository**: `khangtruong-ui/ai-detection-unet-gpu`  
**Status**: Proposed Specification & PR Plan (No Implementation Code Yet)  
**Target Evaluation Protocol**: [`configs/cross-eval/MVSS-v1-protocol.yaml`](../configs/cross-eval/MVSS-v1-protocol.yaml) (`KhangTruong/Tampered-Images-Combined-Dataset-Fake`, `cross_test`)

---

## 1. Executive Summary & Scope

Single-backbone forensic models suffer severe catastrophic forgetting when jointly fine-tuned across incompatible domains (e.g., `CASIAv2.0` splicing vs. `IMD2020` synthetic inpainting, as documented in [`docs/focal_domain_split_and_dual_backbone.md`](focal_domain_split_and_dual_backbone.md)) and collapse on unseen generator families during cross-evaluation ([`docs/cross_evaluation_focal_pim_mismatch_investigation.md`](cross_evaluation_focal_pim_mismatch_investigation.md)).

To cover the non-facial, general-object subsets of `MVSS-v1-protocol.yaml` without heavy foundation backbones (excluding 2B–13B LMMs, `MaskCLIP` ~418M, `SAM ViT-L/H`, and `SegFormer-B5`), this specification defines **`ForensicMoE` Prototype v1**:
- **3 frozen, open-weight, lightweight specialist experts** (~187.7M frozen total, pure PyTorch without `mamba-ssm`):
  1. **Expert 1B — `SparseViT`** (AAAI 2025, `50.3M` params, `46.2G` FLOPs @ `512×512`)
  2. **Expert 2 — `TruFor (Noiseprint++)`** (CVPR 2023, `68.7M` params, `236.5G` FLOPs @ `512×512`)
  3. **Expert 3 — `TruFor-DiffusionPrint`** (CVPRW 2026, `~68.7M` params, `~240G` FLOPs @ `512×512`)
- **1 tiny trainable calibration & gating head** (**~0.4M–1.2M trainable parameters**, <0.7% of ensemble size), trained with all three experts locked in `.eval()` mode so zero forgetting occurs.

### Prototype v1 Scope Boundaries
- **In Scope (15,709 images across 13 sub-datasets):**
  - *Classical Splicing / Copy-Move / Real-Life (3,200):* `CASIA_V1` (920), `CASIA_V2` (1,000), `Columbia_Gray` (180), `Coverage` (100), `IMD2020_RealLife` (1,000)
  - *Legacy / Synthetic Inpainting (4,000):* `IMD2020` (2,000), `COCO_Inpaint` (2,000)
  - *Diffusion & Flow-Matching Inpainting / Splicing (8,509):* `CocoGlide` (507), `OpenSDID_SD1.5` (2,000), `OpenSDID_SDXL` (2,000), `OpenSDID_Flux.1` (2,000), `TGIF2_flux1dev` (1,986), `SAGI-D` (16)
- **Explicitly Deferred for Later Iterations:**
  - `HiFi_IFDL` (2,000 samples — facial-domain `FaShifter`/`STGAN`, out of scope for v1).
  - `NIST16` (220 samples — deferred; requires mask polarity normalization per [`docs/cross_evaluation_focal_pim_mismatch_investigation.md`](cross_evaluation_focal_pim_mismatch_investigation.md#L197-L203)).

---

## 2. Frozen Expert Pool & Open-Weight Sources

| Slot | Model | Architecture | Params / FLOPs (@512) | Open Weights Source & License |
|---|---|---|---|---|
| **Expert 1B** | **SparseViT** | Sparse-coding UniFormer + Lightweight Feature Fusion (`LFF`) | **50.3M** / **46.2G** | [`scu-zjz/SparseViT`](https://github.com/scu-zjz/SparseViT) (`SparseViT_Mul`, Google Drive checkpoint) |
| **Expert 2** | **TruFor (`NP++`)** | `SegFormer MiT-B2` + `Noiseprint++` (1-ch DnCNN) + CMX + `MLPDecoder` + `conf_head` + `det_head` | **68.7M** / **236.5G** | [`grip-unina/TruFor`](https://github.com/grip-unina/TruFor) |
| **Expert 3** | **TruFor-DiffusionPrint** | `SegFormer MiT-B2` + `DiffusionPrint` (**64-ch** MoCo DnCNN) + CMX + `MLPDecoder` + `conf_head` + `det_head` | **~68.7M** / **~240G** | [`giakoupg/diffusionprint`](https://huggingface.co/giakoupg/diffusionprint) (`trufor_diffusionprint_tgif.pth`, CC-BY-4.0) |

---

## 3. Engineering Integration Plan

### 3.1 Proposed Repository Modules
Following existing repository conventions ([`sid_unet/models/msccnet.py`](../sid_unet/models/msccnet.py), [`sid_unet/models/omnifd.py`](../sid_unet/models/omnifd.py), [`sid_unet/models/noise_rag.py`](../sid_unet/models/noise_rag.py)):

1. **`sid_unet/models/sparsevit.py`**:
   - Pure-PyTorch implementation of `SparseViT` (`depth=[5, 8, 20, 7]`, `embed_dim=[64, 128, 320, 512]`, `s_blocks3=[8, 4, 2, 1]`, `s_blocks4=[2, 1]`, `img_size=512`) and `LFF` decoder (`SparseViT_Mul`).
   - Enforces `allow_random_init: bool = False` by default.
2. **`sid_unet/models/trufor.py`**:
   - Unified `TruFor` (`detconfcmx`) builder parameterized by `np_out_channels: int`:
     - `np_out_channels=1` for **Expert 2 (`Noiseprint++`)**, reusing or aligning with `NoisePrintPlusPlus` in [`sid_unet/models/noise_rag.py`](../sid_unet/models/noise_rag.py).
     - `np_out_channels=64` for **Expert 3 (`DiffusionPrint`)**, loading `trufor_diffusionprint_tgif.pth` (`detconfcmxdiffusionprint_tgif2sagid-bal-all`).
   - Returns standardized `return_dict=True` outputs: `mask_logits` `[B, 1, H, W]` (`logit_fg - logit_bg`), `conf_map` `[B, 1, H, W]`, and `det_logit` `[B, 1]`.
3. **`sid_unet/models/forensic_moe.py`**:
   - `ForensicMoE` orchestrating frozen Experts `1B + 2 + 3` and the tiny trainable calibration & gating head.
   - Registered in [`sid_unet/models/unet.py`](../sid_unet/models/unet.py) (`build_model()` and `UNet.from_checkpoint()`) and exported in [`sid_unet/models/__init__.py`](../sid_unet/models/__init__.py).
4. **Configs & Tests**:
   - `configs/experiments/moe/forensic_moe_v1.yaml`
   - Unit tests validating zero-random-init policy, input preprocessing invariants, frozen-expert `requires_grad=False`, and trainable parameter budget (< 1.5M).

### 3.2 Critical Preprocessing & Input Scale Invariant (`[0, 1]` vs. ImageNet)
Upstream source inspection reveals an essential preprocessing split:
- **`TruFor` & `TruFor-DiffusionPrint`** (`builder_np_conf.py`): Expect **raw `[0.0, 1.0]` float RGB** at the top of `forward()`. Both `Noiseprint++` and `DiffusionPrint` (`self.dncnn(rgb)`) operate directly on `[0, 1]` pixels, and `self.prepro(rgb)` applies ImageNet mean/std internally *only* before the `mit_b2` RGB encoder.
- **`SparseViT`** (`utils/datasets.py`): Expects **ImageNet-standardized RGB** (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) at `512 × 512`.

**`ForensicMoE` Contract:**
- Training and cross-eval configs standardize on `data.image_size: [512, 512]` and `data.normalize: false` (raw `[0.0, 1.0]` tensors).
- `ForensicMoE` applies an `_ensure_raw_rgb()` defensive guard (matching [`sid_unet/models/focal.py`](../sid_unet/models/focal.py)), passes `[0.0, 1.0]` RGB directly to Experts 2 and 3, and applies ImageNet standardization only before Expert 1B (`SparseViT`).

### 3.3 Offline Feature/Logit Caching for Fast Gate Training
Because all three experts (~187.7M params) remain **strictly frozen** in `.eval()` mode (`torch.no_grad()`):
- Running a single forward pass over a small balanced calibration dataset can cache `[logit_1, logit_2, logit_3, conf_2, conf_3, det_2, det_3]` + lightweight high-pass summary features to disk.
- Training the `~0.4M–1.2M` gate then requires negligible GPU memory and completes in minutes.

---

## 4. Lightweight Trainable Gate Design (`~0.4M–1.2M` Params)

### 4.1 Why Naive Averaging & Heavy Gates Both Fail
1. **Logit Calibration Mismatch:** `SparseViT` outputs single-channel BCE logits, whereas `TruFor` and `TruFor-DP` output 2-channel CE logits accompanied by learned reliability maps (`conf_2`, `conf_3`). Naive probability averaging lets an out-of-domain expert's false positives dilute an in-domain expert.
2. **Semantic Overfitting Hazard:** Training a multi-million-parameter RGB CNN router on a compact calibration split causes the router to memorize object categories or dataset brightness rather than manipulation cues.

### 4.2 Two-Level Forensic-Signal-Only Gate (~0.4M–1.2M Trainable Params)
1. **Per-Expert Affine Logit Calibration (`6` trainable scalars):**
   $$z'_k = \gamma_k \cdot z_k + \beta_k, \quad k \in \{1\text{B}, 2, 3\}$$
2. **Image-Level Domain Prior Router $\alpha_{\text{img}}(x) \in \Delta^2$ (`~0.25M–0.45M` params):**
   - Global forensic summary vector (~20 dims): `det_2`, `det_3`, plus global `[mean, std, max, p95, frac(>0.5)]` of `[z'_1, z'_2, z'_3, conf_2, conf_3]`.
   - Tiny high-frequency spectral stem: fixed SRM + DCT high-pass residual channels (`3` ch) $\to$ 3-layer depthwise-separable Conv2d + GlobalAvgPool $\to$ `64`-dim vector.
   - 2-layer MLP (`84 -> 64 -> 3`) outputting $\alpha_{\text{img}}(x) = \text{softmax}(g_{\text{img}} / \tau)$.
3. **Pixel-Level Spatial Residual Gate $w_k(x, y) \in \Delta^2$ (`~0.15M–0.75M` params):**
   - Operates at `128 × 128` (`H/4 × W/4`) strictly on **11 forensic channels** (no raw RGB pixels):
     - Calibrated logits `[z'_1, z'_2, z'_3]` (3 ch)
     - TruFor reliability maps `[conf_2, conf_3]` (2 ch)
     - Local binary entropy $H(\sigma(z'_k))$ per expert (3 ch)
     - Broadcast image-level prior `alpha_img` (3 ch)
   - Compact 3-block depthwise-separable residual FCN (`11 -> 48 -> 48 -> 3`, **zero-initialized output conv**) bilinearly upsampled to `512 × 512`:
     $$w(x, y) = \text{softmax}_k\big(\log(\alpha_{\text{img}, k} + \epsilon) + \Delta g_k(x, y)\big)$$
   - **Final Mixture Prediction (log-odds space):**
     $$z_{\text{MoE}}(x, y) = \sum_{k=1}^{3} w_k(x, y) \cdot z'_k(x, y)$$

### 4.3 Gate Training Objective & Anti-Collapse Recipe
- **Balanced Domain Sampling:** Equal per-batch sampling across Classical (`CASIAv2` / `TampCOCO` / `IMD2020_RealLife`), Diffusion (`TGIF2` / `OpenSDID`), and Synthetic Inpainting (`KhangTruong/IMD2020`), with JPEG quality and scale augmentations.
- **Loss:** `0.5 * BCE + 0.5 * Dice` + load-balancing / entropy regularizer to prevent routing collapse + optional soft per-sample oracle supervision toward $\arg\max_k \text{IoU}_k$.

---

## 5. Expected Results & Metric Quality Audit (`MVSS-v1-protocol`)

> **Metric Quality Caveats:**
> - `SparseViT` and `TruFor` papers mix fixed (`0.5`) vs. oracle (`best`) F1 thresholds across tables, and `SparseViT` omits authentic-image false-alarm testing (marked `NR` in `paper-v3.pdf` Table V).
> - `TruFor-DiffusionPrint` reports F1 (`0.5` threshold) on `TGIF-FR` (`SD2.1`, `SDXL`, `Flux`) only.
> - **Omission = Failure Risk:** Datasets omitted in a model's original paper empirically correspond to severe cross-domain drops (confirmed in `OpenSDI` Table 12 and `SAGI-D` Table 2).

| `MVSS-v1-protocol` Subset (Samples) | Expert 1B: `SparseViT` (50.3M) | Expert 2: `TruFor NP++` (68.7M) | Expert 3: `TruFor-DP` (68.7M) | Expected MoE v1 Result & Risk Assessment |
|---|---|---|---|---|
| **`CASIA_V1` (920)** | **0.819–0.827 F1** *(fixed)* | **0.737 F1** *(0.822 best)* | *Unreported* *(likely weak)* | **Strong (`~0.78–0.84 F1`)** — Gate favors SparseViT + TruFor NP++ |
| **`CASIA_V2` (1,000)** | *(in CAT train)* | *(in CAT train)* | *Unreported* | **Strong (`>0.80 F1`)** — In-domain for Experts 1B & 2 |
| **`Columbia_Gray` (180)** | **0.958–0.959 F1** | **0.798–0.859 F1** *(0.914 best)* | *Unreported* | **Very Strong (`~0.90–0.96 F1`)** — SparseViT leads |
| **`Coverage` (100)** | **0.512–0.513 F1** | **0.523–0.600 F1** *(0.735 best)* | *Unreported* | **Moderate–Good (`~0.55–0.62 F1`)** — TruFor NP++ + SparseViT |
| **`IMD2020_RealLife` (1,000)** | *(in CAT train)* | *(in CAT train)* | *Unreported* | **Moderate–Good** *(aligned with CAT-Net's IMD2020 real-life split; `Wild` cross-eval is `0.501` SparseViT / `0.612` TruFor)* |
| **`CocoGlide` (507)** | **0.386 F1** *(ForMa Table I)* | **0.523 F1** *(0.720 best)* | *Unreported* *(GLIDE unseen)* | **Moderate (`~0.48–0.55 F1`)** — Anchored by TruFor NP++ |
| **`TGIF2_flux1dev` (1,986)** | *Unreported* *(likely collapses)* | **0.73 F1** *(`TGIF2-SP` flux1dev Sem)* | **0.83 sem / 0.74 rand F1** *(`TGIF-FR` flux1dev)* | **Strong (`~0.73–0.83 F1`)** — Major gain over `sid_focal_hrnet` (`0.003 F1`) |
| **`SAGI-D` (16)** | *Unreported* *(likely fails)* | **Fails zero-shot** (`0.12–0.20 IoU`) | **In training mix** (`tgif2sagid-bal-all`) | **Good (N=16)** — Covered by Expert 3 |
| **`OpenSDID_SDXL` (2,000)** | *Unreported* *(likely <0.25)* | **Fails** (`0.104 F1`) | **0.67 sem / 0.73 rand F1** *(on `TGIF-FR` SDXL)* | **Medium / Unverified Transfer (`~0.35–0.65 F1`)** — Depends on TGIF $\to$ OpenSDID preprocessing transfer |
| **`OpenSDID_Flux.1` (2,000)** | *Unreported* *(likely <0.15)* | **Fails** (`0.146 F1`) | **0.83–0.89 sem / 0.74–0.85 rand** *(on `TGIF-FR` Flux)* | **Medium–High / Unverified Transfer** — Strongest lightweight Flux.1 candidate |
| **`OpenSDID_SD1.5` (2,000)** | *Unreported* *(likely <0.20)* | **Fails** (`0.107 F1`) | *SD1.5 unreported* (`SD 2.1`: `0.53 sem / 0.25 rand`) | **Weak–Medium (`~0.25–0.50 F1`)** — Known v1 gap (SD1.5 $\ne$ SD2.1) |
| **`IMD2020` (2,000, inpainting)** | *Unreported* *(MVSS-protocol avg `0.24–0.33`)* | *Protocol-MVSS: `0.322 F1`* | *Unreported* | **Weak (`~0.20–0.35 F1`)** — Known v1 gap (synthetic inpainting) |
| **`COCO_Inpaint` (2,000)** | *Unreported* | *Unreported* | *Unreported* | **Weak (`<0.30 F1`)** — Known v1 gap |
| *(`HiFi_IFDL` 2,000)* | *Excluded (face)* | *Excluded (face)* | *Excluded (face)* | *Out of scope for Prototype v1* |
| *(`NIST16` 220)* | *`0.383–0.384 F1`* | *`0.399–0.470 F1`* | *Unreported* | *Deferred (requires GT polarity fix)* |

---

## 6. Verification & Next-Step Checklist (GPU Machine)

- [ ] Implement `SparseViT`, shared `TruFor` (`np_out_channels=1` & `64`), and `ForensicMoE` with unit tests verifying `[0, 1]` vs. ImageNet preprocessing routing and `< 1.5M` trainable parameters.
- [ ] Run individual zero-shot baselines of Experts 1B, 2, and 3 on [`configs/cross-eval/MVSS-v1-protocol.yaml`](../configs/cross-eval/MVSS-v1-protocol.yaml) using `sid-cross-eval` to measure exact `OpenSDID` transfer before gate training.
- [ ] Cache frozen expert outputs on a balanced calibration split and train the tiny `ForensicMoE` gate.
- [ ] Evaluate full `ForensicMoE` per-sub-dataset breakdown and verify zero regression on `CASIA_V1`/`Columbia_Gray`/`TGIF2_flux1dev`.
- [ ] In **v2** (if `IMD2020` / `COCO_Inpaint` / `OpenSDID_SD1.5` remain weak), add a 4th frozen LoRA/SegFormer-B2 specialist trained strictly on `OpenSDID + IMD2020` without touching Experts 1B, 2, or 3.
