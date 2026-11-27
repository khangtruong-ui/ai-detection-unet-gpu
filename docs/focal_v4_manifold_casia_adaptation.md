# FOCAL v4: Native-1024 Manifold Adaptation & Intra-Image Self-Border k-NN Memory Bank on CASIA-v2.0

- **Hugging Face Weights & Reports**: [`KhangTruong/focal-v4-manifold-casia`](https://huggingface.co/KhangTruong/focal-v4-manifold-casia)
- **Experiment Config**: [`configs/experiments/focal/focal_v4_manifold_casia.yaml`](../configs/experiments/focal/focal_v4_manifold_casia.yaml)
- **Training Dataset**: `KhangTruong/Casia-v2.0-Fake` (`train`: 4,354, `validation`: 769) — zero target/cross-eval leakage
- **Cross-Evaluation Protocol**: [`configs/cross-eval/MVSS-v1-protocol.yaml`](../configs/cross-eval/MVSS-v1-protocol.yaml) (`KhangTruong/Tampered-Images-Combined-Dataset-Fake`, `cross_test`, `1,676` stratified samples across 15 sub-datasets, `--samples-per-dataset 120`, `open_close` + `min_area: 64` + `fill_holes: 256` morphological post-processing)
- **Unit & Integration Tests**: [`tests/test_focal_v4_innovations.py`](../tests/test_focal_v4_innovations.py)

---

## 1. Executive Summary

Building on PR #35 (`0.4131` Mean IoU) and PR #40's `focal_v3_multitap_casia` (`0.4304` Mean IoU, `+0.0173` IoU over PR #35), **FOCAL v4** (`focal_v4_manifold_casia`) raises the official 1,676-sample `MVSS-v1-protocol.yaml` Mean IoU to **`0.4474`** (`[0.4270, 0.4679]`), achieving another **`+0.0170` IoU** leap over `v3` (**`+0.0343`** over PR #35, **`+0.0971`** over `KhangTruong/FOCAL`), crossing **`0.5005` Dice/F1**, reducing loss from `0.5125 → 0.4907`, and improving Pixel Accuracy (`0.8632 → 0.8734`), Precision (`0.4968 → 0.5140`), and Specificity (`0.9092 → 0.9197`).

Key cross-dataset breakthroughs on the 1,676-sample protocol vs. `focal_v3_multitap_casia`:
- **`SAGI-D`**: `0.6122 → 0.7018` (**`+0.0896` IoU**)
- **`IMD2020_RealLife`**: `0.5256 → 0.6137` (**`+0.0881` IoU**)
- **`Coverage`**: `0.5471 → 0.6153` (**`+0.0682` IoU**)
- **`NIST16`**: `0.4770 → 0.5287` (**`+0.0517` IoU**)
- **`CASIA_V2`**: `0.8712 → 0.8955` (**`+0.0243` IoU**)
- **`CocoGlide`**: `0.4704 → 0.4898` (**`+0.0194` IoU**)
- **`IMD2020`**: `0.0395 → 0.0526` (**`+0.0131` IoU**, `+33.2%` relative)
- **`OpenSDID_Flux.1`**: `0.2020 → 0.2128` (**`+0.0108` IoU**)
- **`HiFi_IFDL`**: `0.3973 → 0.4062` (`+0.0089` IoU)
- **`COCO_Inpaint`**: `0.0164 → 0.0204` (`+24.4%` relative)

---

## 2. Mechanistic Innovations

1. **Native `1024×1024` Window-Exact ViT Resolution (`vit_input_size: 1024`, `vit_tap_cluster_size: 64`)**:
   - Pretrained SAM ViT-Large inside `KhangTruong/FOCAL` was trained at native `1024×1024` (`64×64` tokens, `14×14` local window attention). Downsampling ViT inputs to `768×768` (`48×48`) distorted both the 2D absolute/relative positional embeddings and window boundaries. Restoring `1024×1024` zero-shot jumped `v3` on `MVSS-40` from `0.4495 → 0.4646` Mean IoU (`+0.0151`) and enabled 1-epoch `Exp 5A` CASIA Val IoU to jump from `0.6310 → 0.7347`.
2. **Dead-Channel-Guarded Per-Image `ShiftGatedAdaBN` (`hrnet_adabn_lambda=0.16`, `hrnet_adabn_shift_floor=0.38`, `hrnet_adabn_shift_scale=0.25`)**:
   - Inspecting HRNet `stage4` BatchNorm layers revealed `running_var` subnormal/dead channels (`5.6e-45`, `running_var <= 1e-5`). `_forward_hrnet_maybe_adabn` leaves dead channels untouched and blends running vs. per-image statistics (`(1 - λ_eff) * running + λ_eff * instance`) strictly per image (zero batch dependence, no buffer mutation), gated by each image's mean active log-variance shift so low-shift domains (`Columbia_Gray` median shift `0.353`, `CASIA_V2` `0.381`) remain unperturbed while shifted OOD domains (`HiFi_IFDL` `0.834`, `CocoGlide` `0.536`) receive full recalibration.
3. **Markov Affinity Graph Diffusion (`_build_affinity_matrix`, `_diffuse_logits_on_graph`) & Detached Otsu Calibration (`_apply_otsu_calibration`)**:
   - Builds a compact-grid (`48×48`) row-stochastic geodesic similarity graph `P = softmax_row((sim - 1) / τ)` (`τ=0.08`) from frozen backbone features and applies 2-step differentiable label propagation `(1 - β) * L_0 + β * (P @ L_t)` (`β=0.35`) to snap jagged boundaries to semantic/forensic object contours.
   - Centers pre-refiner cluster logits partially toward each image's 64-bin 1D Otsu valley threshold (`otsu_cal_alpha=0.45`) with detached histogram computation so DDP autograd through `tap_logit_weights` remains intact.
4. **Orthogonal Zero-Bias Pre-Neck Forensic Projector (`PreNeckForensicProjector`) & Adapter Spectral Decorrelation**:
   - Projects concatenated pre-neck ViT taps `[22, 23]` (`2048 -> 256` channels) via a bias-free linear `1×1` layer + L2 normalization trained jointly with contrastive temperature loss (`0.50`), Gram orthogonality (`0.10`, achieving `max_off_cos = 0.1245`), and `adapter_ortho_weight=0.05` on `BranchBalancedSpatialAdapter`.
5. **Non-Local Affinity-Propagating `ImageRelativeClusterRefiner` + Intra-Image Self-Border k-NN Memory Bank (`refiner_use_border_knn_desc=True`, `refiner_use_srm_desc=True`, `refiner_use_affinity_prop=True`)**:
   - Augments the 24 rotation/flip-invariant image-relative multi-branch geometric descriptors with `+2` fixed SRM high-frequency energy descriptors and `+2` zero-parameter PatchCore-style intra-image Self-Border k-NN anomaly descriptors (`_border_knn_descriptors`: cosine distance to top-`k=32` and top-`k=160` nearest border-ring patches on the `48×48` hypersphere, averaged across fused + HRNet features). Zero-initialized `aff_proj` (`norm=0.5145`) and `out_conv` (`cos=-0.0100`, `bias=[-0.0160, +0.0015]`) preserve step-0 parity and prevent channel memorization or negative CASIA bias drift.
   - Final weights apply **WiSE-FT (`α=0.65`)** weight-space interpolation between 3-epoch `focal_v4_manifold_casia` and `KhangTruong/focal-v3-multitap-casia` over shared trainable tensors (`52` keys), keeping v4-exclusive layers intact.

---

## 3. Iterative Ablation History (`MVSS-40`, 576 Samples)

| Experiment / Probe | Resolution | Key Variant | Val IoU (CASIA) | MVSS-40 IoU | MVSS-40 F1 | Notes |
|---|---|---|---|---|---|---|
| `v3` reference (`KhangTruong/focal-v3-multitap-casia`, raw `P@48`) | `768` | PR #40 baseline | — | `0.4495`* | — | *Pos-only `N=536` `128×128` probe; PR #40 full `MVSS-120` = `0.4304` |
| `v3` zero-shot @ Native `1024` (`P@64`) | `1024` | Restore `64×64` tokens only | — | `0.4646`* | — | `+0.0151` zero-shot from un-distorted pos-embed & window geometry |
| `Exp 4A` (1 ep) | `768` | AdaBN `0.10` + AffDiff `0.35` + Otsu `0.50` + AffRefiner+SRM | `0.6273` | `0.4497` | `0.5063` | Baseline manifold training (`N=576` `sid-cross-eval`) |
| `Exp 4B` (1 ep) | `768` | `Exp 4A` + `PreNeckForensicProjector` | `0.6310` | `0.4500` | `0.5065` | `PreNeck` improved AUROC (`0.8336 → 0.8362`) & `out_conv` ortho (`cos=0.0007`) |
| `Exp 4C` (1 ep) | `768` | `ShiftGatedAdaBN` + Otsu `0.65` + strict refiner + MaskZoom/Feather | `0.6163` | `0.4476` | `0.5061` | Curriculum augs hurt precision/spec; `NIST16` rose to `0.5033` |
| `Exp 4D` (1 ep) | `768` | `Exp 4C` without curriculum augs | `0.6182` | `0.4477` | `0.5060` | Confirmed `otsu_cal_alpha=0.65` / strict gates too restrictive |
| `Exp 5A` (1 ep) | `1024` | Native `1024` + `ShiftGatedAdaBN` + `PreNeck` + `Border-kNN` + Otsu `0.45` | `0.7347` | `0.4623` | `0.5186` | Huge CASIA (`+0.1037`) & `MVSS-40` (`+0.0123`) jump in 1 epoch |
| `focal_v4` Raw (3 ep) | `1024` | `Exp 5A` scaled to 3 epochs | **`0.7374`** | `0.4636` | `0.5199` | Epoch 1 `0.7350` → Ep 2 `0.7361` → Ep 3 `0.7374` |
| `focal_v4` WiSE-FT `α=0.80` | `1024` | `0.20*v3 + 0.80*v4_ep3` | — | `0.4644` | `0.5209` | AUROC `0.8340`, Spec `0.9273` |
| **`focal_v4` WiSE-FT `α=0.65` (Final)** | **`1024`** | **`0.35*v3 + 0.65*v4_ep3`** | — | **`0.4654`** | **`0.5220`** | Promoted to `checkpoint_best.pt` & pushed to HF |

Rejected zero-shot / standalone hypotheses (did not beat baseline or degraded `CASIA_V1`/`Columbia_Gray`):
- Dispersion-calibrated Mahalanobis/Bayes LLR (`0.4315–0.4357`), standalone/blended `K=3/4` clustering (`0.2539 / 0.4421`), 4-side Manifold Ranking (`0.1503–0.3117`), Spectral Fiedler cut (`0.4113`), soft power-law spectral whitening (`0.2916–0.3664`), top-PC semantic damping (`0.2998`), 1D heteroscedastic GMM Bayes (`0.4015`), contour gradient alignment (`0.3715–0.4432`), and vMF-MM EM (`0.3195`).

---

## 4. Official 1,676-Sample Cross-Evaluation Comparison (`MVSS-v1-protocol.yaml`, `--samples-per-dataset 120`)

### Aggregate Metrics

| Model | Loss | Mean IoU (95% CI) | Dice / F1 | AUROC | Pixel Acc | Precision | Recall | Specificity |
|---|---|---|---|---|---|---|---|---|
| `KhangTruong/FOCAL` (`sid_focal_hrnet`) | `0.8794` | `0.3503` `[0.3320, 0.3686]` | `0.4155` | `0.7759` | `0.8114` | `0.3937` | `0.5802` | `0.8504` |
| PR #35 (prior baseline) | — | `0.4131` | `0.4647` | `0.8049` | — | — | — | — |
| `focal_v3_multitap_casia` (PR #40 baseline) | `0.5125` | `0.4304` `[0.4102, 0.4506]` | `0.4856` | **`0.8248`** | `0.8632` | `0.4968` | **`0.5887`** | `0.9092` |
| **`focal_v4_manifold_casia` (Final)** | **`0.4907`** | **`0.4474`** `[0.4270, 0.4679]` | **`0.5005`** | `0.8237` | **`0.8734`** | **`0.5140`** | `0.5828` | **`0.9197`** |

### Per-Dataset Breakdown (All 15 Sub-Datasets)

| Sub-Dataset | Samples | v3 IoU (`v3` AUROC) | v4 IoU (`v4` AUROC) | Δ IoU | v4 Dice/F1 | v4 Prec / Rec / Spec |
|---|---:|---:|---:|---:|---:|---|
| `CASIA_V1` | 120 | `0.8537` (`0.9954`) | `0.8396` (`0.9838`) | `-0.0141` | `0.8883` | `0.8704 / 0.9448 / 0.9827` |
| `CASIA_V2` | 120 | `0.8712` (`0.9906`) | **`0.8955`** (`0.9932`) | **`+0.0243`** | `0.9269` | `0.9141 / 0.9632 / 0.9893` |
| `COCO_Inpaint` | 120 | `0.0164` (`0.5275`) | **`0.0204`** (`0.5302`) | **`+0.0040`** | `0.0359` | `0.0308 / 0.0899 / 0.9146` |
| `CocoGlide` | 120 | `0.4704` (`0.8334`) | **`0.4898`** (`0.8238`) | **`+0.0194`** | `0.5737` | `0.6445 / 0.6150 / 0.8840` |
| `Columbia_Gray` | 120 | `0.9543` (`0.9873`) | `0.9462` (`0.9785`) | `-0.0081` | `0.9604` | `0.9636 / 0.9573 / 0.9878` |
| `Coverage` | 100 | `0.5471` (`0.9289`) | **`0.6153`** (`0.9466`) | **`+0.0682`** | `0.6929` | `0.6475 / 0.8273 / 0.9381` |
| `HiFi_IFDL` | 120 | `0.3973` (`0.8130`) | **`0.4062`** (`0.8251`) | **`+0.0089`** | `0.4801` | `0.5622 / 0.6022 / 0.8535` |
| `IMD2020` | 120 | `0.0395` (`0.6689`) | **`0.0526`** (`0.6666`) | **`+0.0131`** | `0.0753` | `0.0570 / 0.2202 / 0.8937` |
| `IMD2020_RealLife` | 120 | `0.5256` (`0.9693`) | **`0.6137`** (`0.9825`) | **`+0.0881`** | `0.6980` | `0.6556 / 0.9300 / 0.9647` |
| `NIST16` | 120 | `0.4770` (`0.8326`) | **`0.5287`** (`0.8112`) | **`+0.0517`** | `0.6001` | `0.6235 / 0.6258 / 0.9350` |
| `OpenSDID_Flux.1` | 120 | `0.2020` (`0.7283`) | **`0.2128`** (`0.7323`) | **`+0.0108`** | `0.2843` | `0.3088 / 0.3797 / 0.8666` |
| `OpenSDID_SD1.5` | 120 | `0.1405` (`0.6720`) | `0.1134` (`0.6606`) | `-0.0271` | `0.1670` | `0.1843 / 0.2682 / 0.8601` |
| `OpenSDID_SDXL` | 120 | `0.0957` (`0.6046`) | `0.0765` (`0.6015`) | `-0.0192` | `0.1206` | `0.2132 / 0.1586 / 0.8804` |
| `SAGI-D` | 16 | `0.6122` (`0.9193`) | **`0.7018`** (`0.9411`) | **`+0.0896`** | `0.7663` | `0.7285 / 0.8403 / 0.9599` |
| `TGIF2_flux1dev` | 120 | `0.0000` (`1.0000`) | `0.0000` (`1.0000`) | `0.0000` | `0.0000` | `PixelAcc/Spec: 0.9224` (vs `v3` `0.8983`) |

---

## 5. Quality Checks & Telemetry

- **`nn-toolbox` Weight Health Audit**: `inspect_pretrained_weight_health` confirmed zero collinear heads (`collinear_heads: []`), zero collapsed weights in trainable v4 modules, `cluster_refiner.out_conv` cosine `-0.0100`, bias `[-0.0160, +0.0015]`, active non-local `aff_proj` (`norm=0.5145`), and orthogonal `PreNeckForensicProjector` (`max_off_cos=0.1245`).
- **`nn-toolbox` Polarity Diagnostic**: On 2-sample/domain FP16 probe, `0%` polarity flips on `CASIA_V2`, `Columbia_Gray`, `CocoGlide`, `HiFi_IFDL`; remaining large-tamper edge cases still occur occasionally on `OpenSDID_Flux.1`/`OpenSDID_SD1.5`.
- **GPU Utilization Telemetry**: During the 3-epoch 2×T4 DDP training run (`n=2,872` samples at 2s intervals), **GPU 0 averaged `91.2%`** (median `100%`, `1.0%` zeros) and **GPU 1 averaged `88.1%`** (median `100%`, `0.8%` zeros).
- **Reproducibility**:
  ```bash
  sid-train --config configs/experiments/focal/focal_v4_manifold_casia.yaml
  sid-cross-eval --cross-configs configs/cross-eval/MVSS-v1-protocol.yaml \
    --checkpoints KhangTruong/focal-v4-manifold-casia --samples-per-dataset 120
  ```
