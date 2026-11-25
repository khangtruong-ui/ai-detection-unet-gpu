# Deep Forensic Audit: Why Pretrained FOCAL Backbones & PR #34 / #35 Adaptations Fail on Cross-Domain Benchmarks (`MVSS-v1-protocol`)

- **Audit Target**: Pull Requests `#34` ([`KhangTruong/focal-vit-hrnet-hybrid-casia`](https://huggingface.co/KhangTruong/focal-vit-hrnet-hybrid-casia)), `#35` ([`KhangTruong/focal-v2-image-relative-casia`](https://huggingface.co/KhangTruong/focal-v2-image-relative-casia)), and `#36` against pretrained baseline [`KhangTruong/FOCAL`](https://huggingface.co/KhangTruong/FOCAL) (`FOCAL_ViT_weights.pth`, `FOCAL_HRNet_weights.pth`).
- **Evaluation Benchmark**: [`configs/cross-eval/MVSS-v1-protocol.yaml`](../configs/cross-eval/MVSS-v1-protocol.yaml) (`KhangTruong/Tampered-Images-Combined-Dataset-Fake`, `cross_test`, `17,929` images across `15` sub-datasets).
- **Companion Diagnostics PR**: [`khangtruong-ui/nn-toolbox#1`](https://github.com/khangtruong-ui/nn-toolbox/pull/1) (`inspect_pretrained_weight_health`, `batchnorm_domain_shift_test`, `layer_representation_discrepancy_test`, dict-output & diagnostic metric bug fixes).
- **Primary Evidence Basis**: Complete 17,929-sample Hugging Face evaluation JSON reports, 36-shard Parquet pixel/metadata scan, exact CPU weight/SVD/BatchNorm dissection, 4 controlled GPU mechanistic probes (`287`-image causal ablation across 15 sub-datasets, `100`-image strict split-half cross-validated layer-by-layer probing + 292-layer BN covariate shift, `280`-image zero-shot intermediate-depth ViT clustering intervention, and `512`/`768`/`1024` input resolution audit).

---

## 1. Executive Summary & Ranked Findings Matrix

Although PR #34 and PR #35 successfully raised in-domain CASIA accuracy (`CASIA_V2` IoU `0.8074` in PR #34 $\to$ `0.8942` in PR #35) without catastrophic forgetting on traditional splicing (`Columbia_Gray` IoU `0.9540`), both models—as well as the underlying frozen `KhangTruong/FOCAL` weights—suffer severe cross-domain degradation on modern generative inpainting and face-swap benchmarks (`COCO_Inpaint` IoU `0.0201`, `OpenSDID_SDXL` `0.1080`, `OpenSDID_SD1.5` `0.1203`, `OpenSDID_Flux.1` `0.2080`, `HiFi_IFDL` `0.3259`, and `TGIF2_flux1dev` `0.0000`).

Closer weight-space, layer-wise, and statistical dissection shows that **the cross-domain ceiling is NOT primarily caused by adaptation overfitting, nor is all information missing from the backbones**. Rather, it stems from a concrete chain of **four architectural/weight pathologies** and **three benchmark/dataset artifacts**:

| Rank | Component / Scope | Discovered Pathology | Hard Empirical Evidence |
|---:|---|---|---|
| **1** | **Pretrained `FOCAL_ViT_weights.pth`** (`blocks[23]` + `neck[0..3]`) | **Last-Layer Splicing Bottleneck & Bias-Heavy Cone Compression**: Intermediate blocks (`blk_17`–`blk_22`) separate OOD diffusion well, but `blk_23` + `neck[0..3]` amplify CASIA splicing (`GTsep/KMsep = 0.984`) while crushing OOD diffusion (`COCO_Inpaint GT/KM: 0.465 -> 0.034`, `IMD2020: 0.807 -> 0.170`). | Zero-shot clustering **before** the neck (`blk_23`) improves 7-dataset OOD diffusion IoU by **+19.7% relative** (`0.152 -> 0.182`; `HiFi_IFDL +0.099` vs `neck_out`!). `neck[0]` has `cond=66.68, sv1/sv2=2.18`; `neck[3]` has negative gammas (`min=-0.0550`) & huge bias (`||b||=1.2687`), driving OOD `cone_cos` from `0.57` up to `0.868`! |
| **2** | **Pretrained `FOCAL_ViT_weights.pth`** (`blocks[0].attn.proj`, `blocks[23]`) | **Early Spectral Conditioning & Block-23 Dispersion**: `blocks.0.attn.proj` has condition number `57,998.3` (`sv1/sv2=2.38`); `blocks.23.attn.qkv` norm doubles (`51.45 -> 104.74`) while `mlp.lin2` effective rank drops `0.870 -> 0.469` and activation `std` explodes `7.14 -> 59.69` entering `neck[0]`. | Live `nn_toolbox.inspect_pretrained_weight_health` scan flags `blocks.0.attn.proj`, `blocks.{13..16}.mlp.lin1`, and `blocks.23.norm1` (`bias_scale_ratio=0.90, min_weight=-0.0013`). |
| **3** | **Dataset Corruption (`MVSS-v1`)** | **`TGIF2_flux1dev` Has 100% Blank All-Zero PNG Masks**: All `1,986 / 1,986` images (`11.1%` of MVSS-v1) have zero positive ground-truth PNG pixels despite Parquet metadata claiming `mask_ratio = 0.1500` and `label = 1`. | Across shards `20..24`, `all_zero_png = 1,986 / 1,986` (`min=0, max=0`), guaranteeing `IoU = 0.0000`, `F1 = 0.0000`, and `Recall = 0.0000` for **any** detector. |
| **4** | **Evaluation Pipeline Bug (`sid_unet`)** | **Post-Processing Binary AUROC Collapse (`-0.11` Drop & Model Ranking Reversal)**: `post_seg_tracker` received hard `{0, 1}` binary masks and overwrote headline `auroc` with balanced accuracy `(rec + spec)/2` (plus a single-class fast-path inconsistency on `n_pos=0`). | On all 17,929 samples: **PR #34** `0.7399 -> 0.6332` (`-0.1067`), **PR #35** `0.7431 -> 0.6301` (`-0.1130`), falsely reversing PR #35 above PR #34 in headline AUROC! On `TGIF2`, single-class fast-path dropped AUROC from `1.0000` to `0.4549`. *(Fixed in this PR).* |
| **5** | **Clustering Formulation** | **Polarity Inversion (`minority-is-tampered`) & Semantic Hijacking**: On `CocoGlide`, `HiFi_IFDL`, `OpenSDID_SD1.5/SDXL`, raw feature Fisher AUROC is **`0.927–0.977`**, yet unsupervised 2-means collapses to `0.61–0.77` AUROC (`0.07–0.38` IoU). | Fixing polarity alone jumps `CocoGlide` AUROC **`0.633 -> 0.895` (`+0.262`)** / IoU **`0.377 -> 0.619` (`+0.242`)** and `HiFi_IFDL` AUROC **`0.767 -> 0.928` (`+0.161`)**. Meanwhile, `COCO_Inpaint`/`SD1.5` have `GTsep/KMsep = 0.25–0.27` (`ViT~HRNet` IoU `0.09–0.28`), causing semantic variance hijacking. |
| **6** | **Pretrained `FOCAL_HRNet_weights.pth`** | **Final-Stage Residual Collapse & 292-Layer BN Covariate Shift**: `stage4.2.branches.0.{1, 2, 3}` have `conv` norms $\sim 10^{-6}$–$10^{-5}$ and `bn2` running stats at the IEEE-754 float32 subnormal floor **`5.605e-45`** (`16/292` BN layers have dead channels). `stage4` BN variances shift heavily OOD. | Blocks `1..3` contribute `0.000` CV AUROC gain (`0.994 -> 0.994` CASIA, `0.814 -> 0.797` COCO) while constant `bn2.bias` inflates `cone_cos` (`0.745 -> 0.848`). `\|log_var_ratio\|` jumps from `0.212` (`CASIA_V2`) to `0.588` (`HiFi_IFDL`) and `0.392` (`COCO_Inpaint`). |
| **7** | **PR #35 `ImageRelativeClusterRefiner`** | **Rank-1 Anti-Correlated Head Collapse (`cos = -0.9942`) & Constant Negative OOD Bias**: Despite channel-rotation invariance, it memorized CASIA's high `sep_f` (`0.49–0.53`), outputting a nearly uniform negative shift `-0.203` to `-0.206` (`scale 1.102–1.103`) on all OOD datasets. | Degrades OOD diffusion IoU (`PR35_base -> PR35_fin`: `HiFi_IFDL 0.390 -> 0.371`, `OpenSDID_Flux 0.306 -> 0.298`, `SDXL 0.116 -> 0.110`, `COCO_Inpaint 0.013 -> 0.011`). Exacerbated by `cluster_refiner_l2_weight` shrinking `GroupNorm` gammas toward `0` (`0.91–0.93`). *(L2 bug fixed in this PR).* |
| **8** | **PR #35 `SpatialCrossScaleAdapter`** | **Rank-2 CASIA-Biased HRNet-Heavy Perturbation (`2.09x` HRNet RMS Ratio)**: Over-perturbs frozen HRNet channels (`erank = 56.1/128`) and underperforms zero-shot dual-backbone `768+TTA` on OOD diffusion & face-swap. | Feature drift `1 - cos(F_ad, F_bb)` is **`1.75x` larger** on `HiFi_IFDL` (`0.0410`) and `1.66x` on `CocoGlide` (`0.0388`) than `CASIA_V2` (`0.0234`). Zero-shot `ZS_768TTA` beats `PR35_ad` on `COCO_Inpaint` (`0.062 vs 0.016`), `HiFi_IFDL` (`0.479 vs 0.391`), `SDXL` (`0.154 vs 0.118`), and `SAGI-D` (`0.667 vs 0.600`). |
| **9** | **Dead Pristine Gate & HFlip Corner Seed** | **`cluster_separability_threshold = 0.0015` Never Triggers (`0 / 287`)**, and hard `kmeans` seeds from corner pixel `(0, 0)`, breaking strict horizontal-flip equivariance. | Minimum observed `sep35_ad` across 287 images (even blank-GT `TGIF2` mean `0.327`, min `0.067`) is **`0.023` (`15x` above threshold)**. `soft_kmeans` fails due to `cone_cos >= 0.65` + high `em_temp=0.15`, not merely separability threshold. |
| **10** | **Dataset Spatial / Area Skew** | **`COCO_Inpaint` 60.9% Mod-8 Padding Size Mismatch & `IMD2020` Tiny-Patch Skew**: `COCO_Inpaint` images are padded to multiples of 8 while masks keep unpadded dimensions (`dw, dh in [0, 7]`); `IMD2020` in `MVSS-v1` is strictly tiny patches (`mean mask_ratio = 0.0129`). | In `COCO_Inpaint` shard 4, `66 / 101` (`65.3%`) sampled rows had mismatched image vs mask dimensions (100% of padded images divisible by 8) stretched via nearest-neighbor resize in `mask_utils.py`. |

---

## 2. Audit Scope, Weight Integrity & Methodology

### 2.1 Verified Checkpoints & Frozen-Backbone Bit Identity

Five published checkpoints were pulled directly from Hugging Face Hub and compared parameter-by-parameter in FP32 on CPU:

1. `KhangTruong/FOCAL/FOCAL_ViT_weights.pth` (SAM ViT-Large + `neck[0..3]`)
2. `KhangTruong/FOCAL/FOCAL_HRNet_weights.pth` (HRNet-W32)
3. `KhangTruong/focal-vit-hrnet-hybrid-casia/checkpoints/checkpoint_best.pt` (**PR #34**)
4. `KhangTruong/focal-v2-image-relative-casia/checkpoints/checkpoint_best.pt` (**PR #35 best**, WiSE-FT `alpha=0.65`)
5. `KhangTruong/focal-v2-image-relative-casia/checkpoints/checkpoint_latest.pt` (**PR #35 Ep4 raw**, `alpha=1.0`)

Key structural equality measurements:
- `ViT` matched parameters (`347 / 347` tensors): `max |diff(FOCAL_ViT, PR #35 best)| = 0.000e+00`.
- `HRNet` matched parameters (`1,752 / 1,752` tensors): `max |diff(FOCAL_HRNet, PR #35 best)| = 0.000e+00`.
- Cross-run backbone identity: `max |diff(PR #34, PR #35)| = 0.000e+00` across both `net.*` (`ViT`) and `secondary_net.*` (`HRNet`).

Because both backbones are bitwise identical to upstream `FOCAL` across PR #34 and PR #35, **every intrinsic ViT and HRNet anomaly reported below is inherited verbatim from the upstream pretrained weights**.

### 2.2 Experimental Probes Executed

| Probe | Sample Scope | Purpose & Statistical Safeguards |
|---|---|---|
| **Parquet Dataset Scan** | All `36` shards (`17,929` rows) | Columnar PyArrow metadata scan + decoded PNG ground-truth verification on `TGIF2_flux1dev` (`1,986` rows) and `COCO_Inpaint`. |
| **Probe 1 (GPU)** | `287` images (`20` per sub-dataset across 14 datasets, `7` in `SAGI-D`) | Causal ablation across `10` pipeline variants (`HRNet`, `ViT512`, `ViT768`, `ZS_512`, `ZS_768TTA`, `PR34_ad`, `PR34_fin`, `PR35_ad`, `PR35_base`, `PR35_fin`), oracle linear separability, polarity/semantic decomposition, and refiner telemetry. |
| **Probe 2 (GPU)** | `100` images (`10` each across `10` datasets) | Layer-by-layer dissection across `15` ViT depths + `9` HRNet stages using **strict spatial split-half cross-validated centroid AUROC** (fit centroids on even pixel indices, evaluate AUROC on odd indices — zero in-sample leakage), global attention entropy (`Blocks 5, 11, 17, 23`), and live empirical shift across all `292` HRNet `BatchNorm2d` layers. |
| **Probe 3 / `nn-toolbox`** | Full checkpoint tensors (`337.1M` params) | SVD spectra, effective rank, condition number, BN subnormal detection, and output-head collinearity (`khangtruong-ui/nn-toolbox#1`). |
| **Probe 4 (GPU)** | `280` images (`20` × `14` tampered datasets) | Direct causal zero-shot `kmeans` clustering at intermediate ViT depths (`blk_14`, `blk_17`, `blk_20`, `blk_22`, `blk_23`, `neck_out`, `hr_out`, `blk17+hr`) + `512` vs `768` vs `1024` ViT resolution ablation (`10` images × `5` datasets). |

---

## 3. Smoking Gun #1: Pretrained `FOCAL_ViT_weights.pth` Final-Stage Splicing Bottleneck

```
Input Image [3 x H x W]
  |
  v  patch_embed (erank=0.10, cone_cos=0.95)
blocks[0]    attn.proj cond = 57,998.3 (sv1/sv2 = 2.38), norm1.min_w = -0.6118
  |
  v  blocks[1..20] (OOD & splicing forensic separation steadily builds!)
blk_20/22    OOD COCO GT/KM = 0.33-0.35 | CV AUROC: CAS=1.000, HiFi=0.969, SDXL=0.968, COCO=0.943
  |
  v  blocks[23] (FINAL GLOBAL BLOCK)
             qkv.norm DOUBLES (51.45 -> 104.74) | mlp.lin2 erank DROPS (0.870 -> 0.469)
             norm1.bias_ratio = 0.90 (min_w = -0.0013) | token std jumps 7.14 -> 15.11 (CASIA)
  |
  v  neck[0..3] (THE OOD BOTTLENECK)
             neck[0] 1x1 conv [256, 1024]: cond=66.68, sv1/sv2=2.18, act_std = 59.69 (CASIA)
             neck[1] LN2d: weights compressed to mean 0.0508
             neck[3] LN2d: negative affine scales (min=-0.0550) + large bias (||b||=1.2687)
  |
  v  Resulting neck_3_out [256 x 48 x 48]:
             - CASIA_V2:  GT/KM = 0.984, cone_cos = 0.651, cv_auroc = 1.000
             - COCO_Inp:  GT/KM drops 0.465 -> 0.034 (-93%), cone_cos = 0.767, cv_auroc drops 0.943 -> 0.859
             - IMD2020:   GT/KM drops 0.807 -> 0.170 (-79%)
             - HiFi_IFDL: GT/KM drops 0.777 -> 0.321 (-59%)
```

### 3.1 Layer-by-Layer Split-Half CV AUROC & Forensic-to-Semantic Energy Ratio (`GTsep / KMsep`)

On any L2-normalized spatial map $\hat{F}_l \in \mathbb{R}^{N \times d}$ from layer $l$, let $\mu_+, \mu_-$ be unit ground-truth tampered and pristine centroids ($\mathrm{GT}_{\mathrm{sep}} = 1 - \mu_+ \cdot \mu_-$), and let $c_0, c_1$ be unsupervised 2-means cluster centroids ($\mathrm{KM}_{\mathrm{sep}} = 1 - c_0 \cdot c_1$). The ratio $\mathrm{GT}_{\mathrm{sep}} / \mathrm{KM}_{\mathrm{sep}}$ measures how much of the dominant two-cluster feature variance is aligned with true tampering vs. arbitrary background semantics.

**Table 3.1A — ViT-Large (`768x768`) Strict Split-Half Cross-Validated Centroid AUROC Across 15 Depths (`N = 10` per dataset):**

| Layer | `CASIA_V2` | `Columbia` | `IMD_Real` | `CocoGlide` | `HiFi_IFDL` | `SD_Flux.1` | `SDXL` | `COCO_Inp` | `IMD2020` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `patch_embed` | `0.874` | `0.836` | `0.757` | `0.791` | `0.833` | `0.737` | `0.753` | `0.737` | `0.770` |
| `blk_0` | `0.926` | `0.939` | `0.833` | `0.852` | `0.877` | `0.857` | `0.881` | `0.841` | `0.885` |
| `blk_3` | `0.954` | `0.947` | `0.875` | `0.883` | `0.864` | `0.901` | `0.880` | `0.837` | `0.909` |
| `blk_5 (G)` | `0.980` | `0.974` | `0.923` | `0.910` | `0.873` | `0.922` | `0.914` | `0.877` | `0.932` |
| `blk_8` | `0.994` | `0.985` | `0.954` | `0.928` | `0.893` | `0.937` | `0.937` | `0.911` | `0.955` |
| `blk_11 (G)` | `0.998` | `0.994` | `0.978` | `0.946` | `0.930` | `0.950` | `0.955` | `0.926` | `0.974` |
| `blk_14` | `0.999` | `0.997` | `0.987` | `0.954` | `0.957` | `0.955` | `0.960` | `0.935` | `0.980` |
| `blk_17 (G)` | `0.999` | `1.000` | `0.994` | `0.962` | **`0.970`** | **`0.957`** | `0.965` | `0.939` | `0.984` |
| **`blk_20`** | **`1.000`** | **`1.000`** | **`0.996`** | `0.964` | `0.969` | `0.954` | **`0.968`** | **`0.943`** | **`0.985`** |
| `blk_22` | `1.000` | `1.000` | `0.996` | `0.963` | `0.968` | `0.949` | `0.966` | `0.942` | `0.982` |
| `blk_23 (G)` | `1.000` | `1.000` | `0.996` | `0.962` | `0.959` | `0.928` | `0.957` | `0.917` | `0.968` |
| `neck_0_conv1` | `1.000` | `0.997` | `0.992` | `0.950` | `0.943` | **`0.890`** | **`0.908`** | **`0.811`** | **`0.904`** |
| `neck_1_ln1` | `1.000` | `1.000` | `0.995` | `0.961` | `0.955` | `0.912` | `0.934` | `0.848` | `0.935` |
| `neck_2_conv3` | `1.000` | `1.000` | `0.990` | `0.956` | `0.935` | `0.916` | `0.939` | `0.826` | `0.938` |
| `neck_3_out` | `1.000` | `1.000` | `0.996` | **`0.969`** | `0.947` | `0.934` | `0.947` | **`0.859`** | `0.954` |

**Table 3.1B — ViT-Large Layer-by-Layer Forensic-to-Semantic Energy Ratio (`GTsep / KMsep`), Effective Rank, Cone Concentration (`cone_cos = ||E[f_norm]||^2`), and Feature `std`:**

| Layer | `GT/KM` `CASIA` | `GT/KM` `IMD_Real` | `GT/KM` `CocoGlide` | `GT/KM` `HiFi` | `GT/KM` `SDXL` | `GT/KM` `COCO_Inp` | `GT/KM` `IMD2020` | `erank` `CAS / SDXL / COCO` | `cone_cos` `CAS / SDXL / COCO` | `std` `CAS / COCO` |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `patch_embed` | `0.645` | `0.419` | `0.343` | `0.273` | `0.323` | `0.166` | `0.197` | `0.099 / 0.103 / 0.153` | `0.950 / 0.956 / 0.934` | `0.665 / 0.655` |
| `blk_0` | `0.704` | `0.388` | `0.437` | `0.317` | `0.491` | `0.216` | `0.298` | `0.210 / 0.222 / 0.270` | `0.729 / 0.749 / 0.673` | `2.153 / 2.180` |
| `blk_5 (G)` | `0.816` | `0.597` | `0.518` | `0.358` | `0.583` | `0.319` | `0.488` | `0.490 / 0.495 / 0.542` | `0.642 / 0.619 / 0.552` | `3.151 / 3.243` |
| `blk_11 (G)` | `1.061` | `0.828` | `0.820` | `0.569` | `0.721` | `0.463` | `0.698` | `0.568 / 0.570 / 0.599` | `0.512 / 0.495 / 0.439` | `4.372 / 4.441` |
| `blk_14` | **`1.150`** | `0.990` | `0.868` | `0.654` | `0.738` | **`0.465`** | `0.761` | `0.571 / 0.572 / 0.602` | `0.540 / 0.532 / 0.473` | `5.020 / 5.044` |
| `blk_17 (G)` | `1.035` | `1.095` | `0.877` | `0.730` | **`0.769`** | `0.444` | **`0.807`** | `0.563 / 0.569 / 0.601` | `0.564 / 0.574 / 0.521` | `5.737 / 5.677` |
| `blk_20` | `1.008` | **`1.098`** | **`0.937`** | `0.774` | `0.721` | `0.346` | `0.799` | `0.549 / 0.568 / 0.602` | `0.538 / 0.593 / 0.548` | `6.421 / 6.295` |
| `blk_22` | `1.010` | `1.048` | `0.905` | **`0.777`** | `0.697` | `0.331` | `0.728` | `0.536 / 0.564 / 0.598` | `0.533 / 0.608 / 0.564` | `7.139 / 6.895` |
| `blk_23 (G)` | `1.003` | `1.051` | `0.866` | `0.694` | `0.534` | **`0.162`** | **`0.408`** | **`0.388`** `/ 0.524 / 0.564` | `0.483 / 0.652 / 0.616` | **`15.106`** `/ 8.926` |
| `neck_0_conv1` | `0.986` | `0.935` | `0.546` | **`0.226`** | **`0.272`** | **`0.015`** | **`0.090`** | **`0.229`** `/ 0.499 / 0.528` | `0.555 /` **`0.863 / 0.868`** | **`59.688`** `/ 34.989` |
| `neck_1_ln1` | `0.998` | `0.978` | `0.627` | `0.451` | `0.363` | `0.068` | `0.247` | `0.280 / 0.534 / 0.562` | `0.433 / 0.611 / 0.579` | `0.035 / 0.026` |
| `neck_2_conv3` | `0.997` | `0.963` | `0.520` | `0.353` | `0.326` | `0.032` | `0.159` | `0.156 / 0.454 / 0.488` | `0.371 / 0.695 / 0.700` | `0.186 / 0.117` |
| `neck_3_out` | `0.984` | `0.956` | `0.589` | `0.321` | `0.292` | **`0.034`** | **`0.170`** | `0.219 / 0.483 / 0.510` | `0.651 /` **`0.777 / 0.767`** | `0.127 / 0.142` |

Look carefully at the inflection between **`blk_20` / `blk_22`**, **`blk_23 (G)`**, and **`neck_0` $\to$ `neck_3`**:
1. Across `blk_0` through `blk_20`, OOD discriminability monotonically improves everywhere, peaking at `0.943` on `COCO_Inpaint`, `0.968` on `OpenSDID_SDXL`, and `0.985` on `IMD2020`, with `GT/KM` healthy at `0.70–0.80` on `HiFi_IFDL`, `OpenSDID_SDXL`, and `IMD2020`, and `0.35–0.47` on `COCO_Inpaint`.
2. Inside `blk_23` and `neck_0_conv1`, an abrupt **splicing-only projection** takes place: on `CASIA_V2`, activation standard deviation spikes from `7.139 -> 15.106 -> 59.688` while effective rank contracts (`0.536 -> 0.388 -> 0.229`), concentrating nearly all energy into traditional splicing eigenvectors (`GT/KM` locks near `1.000`). On OOD images lacking strong CASIA splicing signatures, `neck_0` collapses `COCO_Inpaint` CV AUROC `0.942 -> 0.811`, `OpenSDID_SDXL` `0.966 -> 0.908`, and `IMD2020` `0.982 -> 0.904`, while crashing `GT/KM` to **`0.015`** (`COCO_Inpaint`), **`0.090`** (`IMD2020`), and **`0.226`** (`HiFi_IFDL`).
3. Across every OOD dataset except traditional `Columbia_Gray`/`IMD2020_RealLife`, `neck[3]` compresses unit tokens into a narrow positive cone: `cone_cos = 0.767–0.777` (meaning any two random pixels have average cosine similarity $\ge 0.77$!), leaving almost no angular margin for 2-means clustering.

### 3.2 Weight-Space & Attention Entropy Dissection of ViT-Large

Why does this abrupt bottleneck happen between `blk_22` and `neck[3]`? Inspecting the weights directly reveals five concrete signatures:

| Weight Layer | Norms & Spectral Properties | Diagnostic Anomaly |
|---|---|---|
| `blocks.0.attn.proj` | `proj=31.40`, `erank=0.587`, `sv1/sv2=2.38`, **`cond=57,998.3`** | Extreme early ill-conditioning in `proj`; `blocks.0.norm1` has `min_w = -0.6118`. |
| `blocks.{5,11,14,17}` | `qkv=48.9–76.0`, `mlp2=54.5–87.4`, `mlp2 erank=0.864–0.890`, `norm1_g=0.81–1.13` | Clean, high-rank, well-conditioned intermediate feature extractors. |
| `blocks.{20,21,22}` | `qkv=51.5–59.3`, `mlp2 erank=0.806 -> 0.678`, `norm1_g=0.817 -> 0.575` | Gradual rank compression toward final block. |
| **`blocks.23` (Final)** | `qkv=104.74` (**2.14x** `blk_17`), `mlp2 erank=0.469`, `norm1.min_w=-0.0013`, `bias_ratio=0.90` | Massive late query-key/value scaling (`+114%`) and low-rank MLP collapse. |
| **`neck[0]` (`conv1x1`)** | `[256, 1024]`, `norm=18.953`, `sv=[4.31, 1.98, 1.73, ..., 0.065]`, **`sv1/sv2=2.18, cond=66.68`** | Single top singular vector (`4.31`) dominates `1024 -> 256` dimensionality reduction by $>2.18\times$, amplifying CASIA splicing direction (`std=59.69`) and attenuating orthogonal OOD signals. |
| **`neck[1]` (`LN2d`)** | `weight mean=0.0508`, `std=0.0087`, `range=[0.0077, 0.0701]`, `bias norm=0.1353` | Severe scale compression ($0.05\times$ unity) following `neck[0]`'s amplitude blow-up. |
| **`neck[2]` (`conv3x3`)** | `[256, 2304]`, `norm=26.512`, `sv=[4.27..0.24]`, `cond=17.81`, `erank=235.3/256 (0.919)` | Spatial conv recovers some local texture (`0.811 -> 0.859` COCO AUROC), but cannot undo `neck[0]` subspace loss. |
| **`neck[3]` (`LN2d`)** | `weight mean=0.1466`, `range=[-0.0550, +0.2251]`, **`bias norm=1.2687`** | **Negative affine scales** paired with large **`1.269`** static bias offset shifts every output vector into a narrow bias-dominated cone (`cone_cos = 0.777`). |

Meanwhile, global attention entropy measurements across all four global attention blocks (`5, 11, 17, 23`) **rule out token-sink / attention-map collapse**:

- **Block 5** (`ent/log(HW)` / `max_w`): `CASIA_V2 0.633 (0.19)` | `Columbia 0.630 (0.19)` | `CocoGlide 0.622 (0.20)` | `HiFi 0.659 (0.18)` | `SDXL 0.636 (0.19)` | `COCO 0.632 (0.19)` | `TGIF2 0.664 (0.21)`
- **Block 11**: `CASIA_V2 0.468 (0.28)` | `Columbia 0.467 (0.27)` | `CocoGlide 0.458 (0.29)` | `HiFi 0.496 (0.24)` | `SDXL 0.458 (0.29)` | `COCO 0.446 (0.30)` | `TGIF2 0.532 (0.22)`
- **Block 17**: `CASIA_V2 0.536 (0.19)` | `Columbia 0.532 (0.19)` | `CocoGlide 0.539 (0.20)` | `HiFi 0.569 (0.17)` | `SDXL 0.554 (0.18)` | `COCO 0.547 (0.20)` | `TGIF2 0.588 (0.17)`
- **Block 23**: `CASIA_V2 0.531 (0.20)` | `Columbia 0.547 (0.18)` | `CocoGlide 0.537 (0.21)` | `HiFi 0.545 (0.18)` | `SDXL 0.539 (0.21)` | `COCO 0.514 (0.23)` | `TGIF2 0.540 (0.23)`

Attention maps remain diffuse across domains; the OOD degradation is caused strictly by the `blocks[23] -> neck[0..3]` channel projection and bias cone shift.

### 3.3 Direct Causal Proof: Unsupervised Zero-Shot Clustering BEFORE `neck` vs. `neck_out`

Running FOCAL's exact unsupervised clustering (`kmeans`, PR #35's active spherical 2-means mode, **zero fine-tuning**) directly on intermediate ViT blocks across all `14` tampered datasets (`N = 280` images) proves that the neck actively throws away OOD localization performance:

**Table 3.3 — Zero-Shot Unsupervised 2-Means Continuous IoU / AUROC by ViT Depth (`768x768`, no adapter/refiner):**

| Dataset | `blk_14` IoU / AUC | `blk_17` IoU / AUC | `blk_20` IoU / AUC | `blk_22` IoU / AUC | **`blk_23` IoU / AUC** | **`neck_out` IoU / AUC** | `hr_out` IoU / AUC | `blk17+hr` IoU / AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `CASIA_V1` | `0.832 / 0.957` | `0.941 / 0.999` | `0.967 / 1.000` | `0.972 / 1.000` | `0.969 / 1.000` | `0.966 / 1.000` | `0.965 / 1.000` | `0.966 / 1.000` |
| `CASIA_V2` | `0.571 / 0.797` | `0.785 / 0.907` | `0.904 / 0.996` | `0.930 / 1.000` | **`0.944`** `/ 1.000` | `0.939 / 1.000` | `0.898 / 0.999` | `0.897 / 0.999` |
| `Columbia_Gray` | `0.698 / 0.922` | `0.946 / 1.000` | `0.968 / 1.000` | `0.968 / 1.000` | `0.965 / 1.000` | `0.964 / 1.000` | `0.964 / 1.000` | `0.965 / 1.000` |
| `Coverage` | `0.117 / 0.565` | `0.164 / 0.708` | `0.304 / 0.889` | `0.328 / 0.912` | `0.436 / 0.928` | **`0.515`** `/ 0.923` | `0.300 / 0.765` | `0.376 / 0.867` |
| `IMD2020_RealLife` | `0.118 / 0.625` | `0.329 / 0.813` | `0.523 / 0.873` | `0.529 / 0.877` | `0.671 / 0.929` | **`0.712`** `/ 0.923` | `0.484 / 0.921` | `0.492 / 0.925` |
| `NIST16` | `0.304 / 0.699` | `0.370 / 0.681` | `0.530 / 0.761` | **`0.602`** `/ 0.828` | `0.583 / 0.822` | `0.537 / 0.815` | `0.515 / 0.882` | `0.465 / 0.793` |
| `SAGI-D` | `0.367 / 0.756` | `0.431 / 0.825` | `0.488 / 0.827` | **`0.543 / 0.867`** | `0.466 / 0.872` | `0.362 / 0.813` | **`0.556 / 0.897`** | `0.547 / 0.889` |
| `CocoGlide` | `0.141 / 0.363` | `0.178 / 0.398` | `0.242 / 0.442` | `0.245 / 0.485` | `0.285 / 0.639` | **`0.352 / 0.723`** | `0.246 / 0.537` | `0.247 / 0.498` |
| `HiFi_IFDL` | `0.158 / 0.420` | `0.163 / 0.407` | `0.360 / 0.755` | **`0.394 / 0.818`** | `0.375 / 0.815` | `0.276 / 0.786` | `0.343 / 0.667` | **`0.398`** `/ 0.764` |
| `OpenSDID_Flux.1` | `0.277 / 0.682` | `0.298 / 0.710` | `0.290 / 0.693` | `0.271 / 0.673` | **`0.331`** `/ 0.749` | `0.249 / 0.764` | `0.283 / 0.787` | `0.285 /` **`0.789`** |
| `OpenSDID_SD1.5` | `0.107 / 0.578` | `0.108 / 0.616` | **`0.121`** `/ 0.653` | `0.120 / 0.644` | `0.101 /` **`0.668`** | `0.061 / 0.614` | `0.075 / 0.585` | `0.071 / 0.608` |
| `OpenSDID_SDXL` | **`0.189 / 0.692`** | `0.148 / 0.588` | `0.155 / 0.645` | `0.147 / 0.619` | `0.127 / 0.613` | `0.087 / 0.632` | `0.122 / 0.592` | `0.126 / 0.610` |
| `COCO_Inpaint` | **`0.055`** `/ 0.519` | `0.048 / 0.538` | `0.033 / 0.499` | `0.026 / 0.461` | `0.037 / 0.459` | `0.014 / 0.464` | `0.075 / 0.640` | **`0.077 / 0.675`** |
| `IMD2020` | `0.017 / 0.578` | `0.017 / 0.582` | `0.017 / 0.578` | `0.015 / 0.573` | `0.017 / 0.557` | `0.028 / 0.527` | **`0.039 / 0.658`** | `0.022 / 0.622` |
| **MEAN (14 Pos)** | `0.282 / 0.654` | `0.352 / 0.698` | `0.422 / 0.758` | `0.435 / 0.768` | **`0.450 / 0.789`** | `0.433 / 0.784` | `0.419 / 0.781` | `0.424 / 0.788` |
| **MEAN (7 OOD-Diff)** | `0.135 / 0.547` | `0.137 / 0.548` | `0.174 / 0.609` | `0.174 / 0.610` | **`0.182 / 0.643`** | `0.152 / 0.644` | `0.169 / 0.638` | `0.175 /` **`0.652`** |

Look at those deltas vs. `neck_out`:
- **`blk_23` vs `neck_out`**: `+0.017` over all 14 datasets and **`+0.030` (`+19.7%` relative)** over the 7 OOD diffusion/inpainting subsets (`HiFi_IFDL +0.099`, `OpenSDID_Flux.1 +0.082`, `OpenSDID_SD1.5 +0.040`, `OpenSDID_SDXL +0.040`, `SAGI-D +0.104`).
- Even `blk_22` beats `neck_out` on `HiFi_IFDL` (`0.394` vs `0.276`, **+0.118**), `SAGI-D` (`0.543` vs `0.362`, **+0.181**), `NIST16` (`0.602` vs `0.537`), and `OpenSDID_SDXL` (`0.147` vs `0.087`), while `blk_14` doubles `OpenSDID_SDXL` (`0.189` vs `0.087`) and nearly quadruples `COCO_Inpaint` (`0.055` vs `0.014`).
- Only `Coverage`, `IMD2020_RealLife`, and `CocoGlide` benefit from the post-neck layer (`0.436 -> 0.515`, `0.671 -> 0.712`, `0.285 -> 0.352`), proving that a **multi-tap fusion across pre-neck (`blk_17`/`blk_22`/`blk_23`) and HRNet** preserves both domains rather than forcing everything through `neck[0..3]`.

### 3.4 First-Principles Explanation of PR #35 Failure Lesson #1 (`soft_kmeans` Collapse)

In PR #35's diagnostic table (`zs_1_soft_512` through `zs_4_soft_512_tta`), `soft_kmeans` collapsed overall MVSS-40 IoU from `0.4035` (`kmeans`) down to `0.2945`. PR #35 hypothesized that soft EM pulled centroids below `cluster_separability_threshold` (`0.0015`), tripping `_apply_pristine_gate`. Probe 1 + Probe 2 reveal a deeper geometric cause:
- In PR #35's own code (`focal.py` lines 2862–2887), when `cone_cos` reaches `0.73–0.78` (`neck[3]` + `HRNet`), typical cosine similarities between a pixel and $c_0$ vs. $c_1$ differ by only $\Delta s \approx 0.03\text{--}0.08$.
- Dividing by `em_temp = 0.15` in the first 5 EM steps gives softmax logits separated by $\Delta s / 0.15 \approx 0.2\text{--}0.5$, yielding soft assignment weights $w \in [0.38, 0.62]$ rather than sharp clusters!
- Averaging $0.4 \cdot \text{foreground} + 0.6 \cdot \text{background}$ pulls both EM centroids directly toward the global mean, washing out minority regions and collapsing `sep_f` even without needing `sep < 0.0015`!

### 3.5 ViT Input Resolution (`512` vs `768` vs `1024`) Trade-off

Because `FOCAL_ViT_weights.pth` has `64x64` positional embeddings (`pos_embed`) and `rel_pos_h/w` trained for SAM ViT (`1024x1024`), changing `vit_input_size` interpolates `pos_embed` and relative attention biases:

| ViT Size (`neck_out`, `N=10`) | `CASIA_V2` IoU / AUC | `OpenSDID_Flux.1` IoU / AUC | `OpenSDID_SDXL` IoU / AUC | `COCO_Inpaint` IoU / AUC | `IMD2020` IoU / AUC |
|---:|---|---|---|---|---|
| `512` (`32x32` tokens) | `0.861 / 0.976` | `0.283 / 0.804` | `0.144 / 0.759` | **`0.093 / 0.571`** | `0.042 / 0.498` |
| `768` (`48x48` tokens, PR #35) | `0.939 / 0.999` | **`0.289 / 0.827`** | `0.136 / 0.746` | `0.022 / 0.386` | `0.022 / 0.453` |
| `1024` (`64x64` native) | **`0.966 / 1.000`** | `0.181 / 0.729` | **`0.151`** `/ 0.729` | `0.020 / 0.414` | **`0.044 / 0.535`** |

Increasing ViT size from `512 -> 768 -> 1024` boosts in-domain `CASIA_V2` monotonically (`0.861 -> 0.939 -> 0.966`) as high-frequency splicing boundaries resolve more cleanly, but **drops `COCO_Inpaint` (`0.093 -> 0.022`) and `OpenSDID_Flux.1` (`0.283 @ 512 / 0.289 @ 768 -> 0.181 @ 1024`)** because `neck[0..3]`'s bias cone dominates finer grids on low-contrast diffusion textures.

---

## 4. Smoking Gun #2: Pretrained `FOCAL_HRNet_weights.pth` Residual Collapse & 292-Layer BN Shift

### 4.1 Complete Residual Collapse in `stage4.2.branches.0.{1, 2, 3}` & Dead BN Channels

In `FOCAL_HRNet_weights.pth` (frozen across `FOCAL`, PR #34, and PR #35), **`16 / 292`** `BatchNorm2d` layers contain dead channels (`running_var < 1e-6`), and all **3 final high-resolution residual blocks (`stage4.2.branches.0.{1, 2, 3}`)** collapsed to IEEE-754 float32 subnormals during upstream contrastive training:

| Block in `stage4.2.branches.0` | `conv1.weight` Norm | `conv2.weight` Norm | `bn1.weight` Norm | `bn2.weight` Norm | `bn2.bias` Norm `[min, max]` | `bn2.running_mean` Max Abs | `bn2.running_var` Max |
|---|---:|---:|---:|---:|---|---:|---:|
| `blk 0` (active) | `6.92e-01` | `8.35e-01` | `2.22e-01` | `7.42e-01` | `0.3242 [-0.0973, +0.0963]` | `1.073e-02` | `1.860e-04` |
| **`blk 1` (DEAD)** | **`7.41e-05`** | **`5.20e-05`** | **`2.94e-06`** | `1.21e-03` | `0.3315 [-0.0973, +0.0972]` | **`5.605e-45`** | **`5.605e-45`** |
| **`blk 2` (DEAD)** | **`5.60e-06`** | **`4.33e-06`** | **`2.59e-07`** | `1.14e-04` | `0.3315 [-0.0973, +0.0972]` | **`5.605e-45`** | **`5.605e-45`** |
| **`blk 3` (DEAD)** | **`9.76e-05`** | **`6.48e-05`** | **`4.51e-06`** | `1.18e-03` | `0.3315 [-0.0973, +0.0972]` | **`5.605e-45`** | **`5.605e-45`** |

Notice that `5.605193857299268e-45` ($2^{-147}$) is literally at the IEEE-754 float32 subnormal floor. Across blocks 1–3, **every convolution weight shrunk to $\le 10^{-4}$** while all 32 `bn2.bias` entries stayed locked near `[-0.0973, +0.0972]`. Because HRNet `BasicBlock` computes $\mathrm{ReLU}(\mathrm{BN}_2(\mathrm{Conv}_2(\dots)) + x)$, blocks 1–3 execute almost pure identity + constant bias shift!

Additional dead BN channels (`running_var < 1e-6`) exist in:
- `bn1`: **`19 / 64`** channels dead (`min = 6.33e-16`)
- `stage4.2.branches.0.0.bn1`: **`14 / 32`** channels dead (`min = 5.19e-14`)
- `stage4.0.branches.0.0.bn1`: **`8 / 32`** channels dead (`min = 2.47e-10`)
- `stage4.1.branches.0.2.bn1`: **`3 / 32`** channels dead (`min = 3.30e-11`)
- Plus `layer1.0.bn3` and `layer1.0.downsample.1` in Stage 1 (`16/256` and `15/256`).

### 4.2 Stage-by-Stage Empirical Consequences & Positive-Orthant ReLU Cone

Because HRNet ends with `ReLU(out + residual)`, its output embedding is constrained strictly to $\mathbb{R}_{\ge 0}^{32}$ (in contrast to zero-mean signed embeddings). Probe 2 confirms:

| HRNet Stage | `CASIA_V2` CV AUC | `HiFi` CV AUC | `SDXL` CV AUC | `COCO_Inp` CV AUC | `GT/KM` `CAS / SDXL / COCO` | `cone_cos` `CAS / COCO` | `erank` `CAS / COCO` | `zero_frac` |
|---|---:|---:|---:|---:|---|---|---|---|
| `stem_bn2` | `0.908` | `0.858` | `0.814` | `0.751` | `0.178 / 0.052 / 0.047` | `0.734 / 0.667` | `0.648 / 0.699` | `0.271` |
| `layer1` | `0.933` | `0.858` | `0.855` | `0.795` | `0.437 / 0.151 / 0.116` | `0.395 / 0.348` | `0.714 / 0.719` | `0.491` |
| `stage2_b0` | `0.856` | `0.683` | `0.808` | `0.717` | `0.374 / 0.244 / 0.141` | `0.793 / 0.745` | `0.904 / 0.893` | `0.035` |
| `stage3_b0` | `0.786` | `0.710` | `0.734` | `0.704` | `0.153 / 0.144 / 0.057` | `0.833 / 0.790` | `0.882 / 0.893` | `0.355` |
| `stage4_m0_b0` | `0.967` | `0.867` | `0.808` | `0.764` | `0.499 / 0.331 / 0.079` | `0.895 / 0.872` | `0.810 / 0.819` | `0.422` |
| `stage4_m1_b0` | `0.994` | `0.939` | `0.827` | `0.814` | `1.048 / 0.290 / 0.032` | `0.783 / 0.854` | `0.587 / 0.715` | `0.313` |
| `stage4_m2_b0_blk0` | `0.994` | `0.938` | `0.822` | `0.805` | `1.040 / 0.297 / 0.032` | `0.745 / 0.837` | `0.555 / 0.706` | `0.292` |
| `stage4_m2_b0_blk3` | `0.994` | `0.939` | `0.824` | `0.797` | `1.068 / 0.278 / 0.027` | **`0.848 / 0.899`** | `0.565 / 0.698` | `0.242` |
| `stage4_out` | **`0.999`** | **`0.977`** | **`0.928`** | **`0.862`** | `1.019 / 0.397 / 0.162` | `0.627 / 0.735` | `0.399 / 0.680` | `0.399` |

Notice two things:
1. `stage4_m2_b0_blk0 -> blk3` produces **zero improvement** in CV AUROC (`0.994 -> 0.994` CASIA, `0.805 -> 0.797` COCO), **strictly worsens** `cone_cos` (`0.745 -> 0.848` CASIA, `0.837 -> 0.899` COCO) due to constant `bn2.bias`, and only multi-branch cross-resolution fusion (`fuse_layers` entering `stage4_out`) rescues `stage4_out` (`COCO_Inpaint CV AUROC 0.797 -> 0.862`, `OpenSDID_SDXL 0.824 -> 0.928`)!
2. Across ALL `292` `BatchNorm2d` layers (evaluating strictly on active channels `running_var > 1e-6`), live batch statistics diverge significantly on OOD generative datasets — especially in lower-resolution HRNet fusion branches (`branches.3.*.bn2`) and `layer1.3`:

| Dataset | `BN mean_z` (Mean / P90 / Max) | `BN \|log(var_batch / running_var)\|` (Mean / P90 / Max) |
|---|---|---|
| `CASIA_V2` (in-domain ref) | `0.194 / 0.317 / 0.795` | `0.212 / 0.415 / 1.063` |
| `OpenSDID_SDXL` | `0.191 / 0.281 / 0.623` | `0.257 / 0.420 / 0.996` |
| `CocoGlide` | `0.241 / 0.403 / 0.721` | `0.281 / 0.544 / 1.967` |
| `Columbia_Gray` | `0.269 / 0.399 / 0.579` | `0.286 / 0.482 / 0.978` |
| `IMD2020` | `0.215 / 0.318 / 0.558` | `0.306 / 0.561 / 1.405` |
| `IMD2020_RealLife` | `0.254 / 0.373 / 0.571` | `0.317 / 0.464 / 0.775` |
| `OpenSDID_Flux.1` | `0.232 / 0.349 / 0.566` | `0.338 / 0.558 / 1.209` |
| `COCO_Inpaint` | `0.277 / 0.411 / 0.620` | **`0.392 / 0.707 / 1.812`** |
| `TGIF2_flux1dev` | **`0.300`** `/ 0.436 / 0.795` | **`0.545 / 0.899 / 2.116`** |
| **`HiFi_IFDL`** | **`0.355 / 0.594 / 1.142`** | **`0.588 / 1.006 / 2.370`** |

Top shifted HRNet BN layers (`OpenSDID_SDXL + COCO_Inpaint` vs. `CASIA_V2`): `stage4.2.branches.3.1.bn2` (`lv 0.48 -> 1.31`, delta `+0.78`), `stage4.1.branches.3.2.bn2` (`0.48 -> 1.16`, `+0.65`), `layer1.3.bn3` (`+0.61`), `stage3.3.branches.1.2.bn2` (`+0.52`), and `stage4.2.branches.3.0.bn2` (`+0.51`). Notice that **1/8-scale semantic branch 3** shows $>2.5\times$ variance ratio shift on diffusion images under frozen CASIA running statistics.

---

## 5. Adapted Module Pathologies in PR #34 & PR #35

PR #34 and PR #35 both improved in-domain CASIA metrics cleanly. However, deep telemetry explains **why those gains failed to transfer (and even regressed) on OOD diffusion and face-swap subsets**.

### 5.1 `ImageRelativeClusterRefiner` (PR #35): Rank-1 Head Collapse & Separability Memorization

Although [`test_descriptor_exact_channel_rotation_invariance`](../tests/test_focal_v2_image_relative.py) proves `ImageRelativeClusterRefiner` is mathematically invariant to arbitrary orthogonal channel rotations, **invariance to channel rotations does not imply invariance to cluster separability magnitude (`sep_f`) or margin scale (`c_base`)**:

1. **Rank-1 Anti-Correlated Output Head (`cos = -0.9942`)**:
   - Inspecting `cluster_refiner.out_conv.weight` (`[2, 24]`): `||w_bias|| = 0.3230`, `||w_scale|| = 0.3259`, and **`cos(w_bias, w_scale) = -0.994179`**, with static biases `[-0.06367, +0.04946]`.
   - Thus `raw_scale ≈ -raw_bias` at every spatial pixel—the 2-channel residual affine head collapsed onto a single 1D scalar direction (high negative `raw_bias` paired with slightly $>1$ positive `raw_scale`).
2. **CASIA Separability Memorization $\to$ Constant Negative OOD Shift (`-0.203` to `-0.206`)**:
   - In-domain `CASIA_V1` / `CASIA_V2` have `sep_f = 0.49–0.53` and large median `|c_base| = 5.4–5.8`. Here the refiner applies mild positive/calibrated biases (`+0.035` on `CASIA_V1`, `-0.053` on `CASIA_V2`, scales `0.980` and `1.025`).
   - On **every OOD dataset** (where `sep_f = 0.08–0.30` and median `|c_base| = 1.0–3.3`), the refiner fails to recognize weak separability as uncertain and instead emits an almost spatially constant global suppression **`bias = -0.203 to -0.206`** and **`scale = 1.102–1.103`** (`Probe 1 Table B` below!).
   - Directly comparing `PR35_base -> PR35_fin` (Isolating ONLY `cluster_refiner` while holding ViT+HRNet+Adapter+BranchWeights fixed): it improves `IMD2020_RealLife` (`0.722 -> 0.733`), `Coverage` (`0.464 -> 0.479`), and `SAGI-D` (`0.605 -> 0.616`), but **degrades IoU on every generative/face-swap subset**: `HiFi_IFDL` (`0.390 -> 0.371`), `OpenSDID_Flux.1` (`0.306 -> 0.298`), `OpenSDID_SD1.5` (`0.083 -> 0.074`), `OpenSDID_SDXL` (`0.116 -> 0.110`), and `COCO_Inpaint` (`0.013 -> 0.011`).
3. **`cluster_refiner_l2_weight` Bug Shrinks `GroupNorm` Gammas Toward 0**:
   - `sid_unet/models/focal.py` previously applied `p.float().pow(2).mean()` across **all** `cluster_refiner.parameters()`, pulling 1D `GroupNorm` scale weights (`gamma`, initialized at `1.0`) down toward zero (`0.9522 -> 0.9098` across blocks `0..2` in PR #35; `0.9578 -> 0.9466` in PR #34). *(Fixed in this PR by anchoring normalization `weight` parameters around `1.0`).*

### 5.2 `SpatialCrossScaleAdapter` (PR #35): Rank-2 CASIA Mode & `2.09x` HRNet Asymmetry

Inspecting `feature_adapter.proj_out.weight` (`[288, 128]`) in PR #35:
- **Spectral concentration**: top-6 normalized singular values `[1.0000, 0.8253, 0.4268, 0.4051, 0.2906, 0.2614]` (`erank = 56.1 / 128`), showing adaptation is dominated by a rank-2 CASIA subspace.
- **2.09x HRNet channel RMS imbalance**: ViT target channels `[0:256]` have RMS `0.00225`, whereas HRNet channels `[256:288]` receive **`0.00470` (`2.09x` higher per-channel perturbation)**, explaining why the adapter causes `1.75x` larger cosine drift `1 - cos(F_ad, F_bb)` on OOD `HiFi_IFDL` (`0.0410`) and `CocoGlide` (`0.0388`) than `CASIA_V2` (`0.0234`).
- **Causal consequence (`ZS_768TTA -> PR35_ad`)**: On OOD diffusion and face-swap subsets, PR #35's trained adapter **lowers** IoU relative to zero-shot dual-backbone concatenation (`ZS_768TTA`):
  - `COCO_Inpaint`: `0.062 -> 0.016` (`-74%` drop; AUROC `0.622 -> 0.462`)
  - `HiFi_IFDL`: `0.479 -> 0.391` (`-0.088`; AUROC `0.871 -> 0.765`)
  - `OpenSDID_SDXL`: `0.154 -> 0.118` (`-0.036`; AUROC `0.670 -> 0.637`)
  - `SAGI-D`: `0.667 -> 0.600` (`-0.067`)

### 5.3 Under-Utilized Fusion Scalars & Exact WiSE-FT (`alpha=0.65`) Audit

- `fusion_module.fusion_scale_logits`: `[+0.00125, -0.00125]` (effective concat weights `[1.0012, 0.9988]` — essentially inert).
- `branch_logit_weights`: `[+0.0475, +0.0425]` (`0.5 * tanh(...) = [+0.0237, +0.0213]` — `< 2.5%` contribution to `c_base`).
- **WiSE-FT (`alpha=0.65`) implementation**: Comparing `checkpoint_best.pt` vs `checkpoint_latest.pt` shows `proj_out` scaled by `0.6406`, `branch_logit_weights` by `0.6141`, and `fusion_scale_logits` by `0.5658` (and config refiner caps scaled from `0.40/0.20` to `0.26/0.13`), while `cluster_refiner.out_conv` remained almost unscaled at **`0.9551`**!

---

## 6. Mathematical Decomposition of Unsupervised Clustering Failure Modes

### 6.1 Full 10-Stage Causal Ablation Across All 15 `MVSS-v1` Sub-Datasets (Probe 1)

| Dataset | `HRNet` | `ViT512` | `ViT768` | `ZS_512` | `ZS_768TTA` | `PR34_ad` | `PR34_fin` | `PR35_ad` | `PR35_base` | `PR35_fin` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`CASIA_V1`** | `0.965 / 1.000` | `0.946 / 0.999` | `0.966 / 1.000` | `0.968 / 1.000` | `0.975 / 1.000` | `0.971 / 1.000` | `0.974 / 1.000` | `0.982 / 1.000` | `0.981 / 1.000` | **`0.982 / 1.000`** |
| **`CASIA_V2`** | `0.898 / 0.999` | `0.845 / 0.947` | `0.940 / 0.998` | `0.906 / 0.999` | `0.942 / 1.000` | `0.913 / 0.997` | `0.918 / 0.996` | `0.969 / 1.000` | `0.968 / 1.000` | **`0.970 / 1.000`** |
| **`Columbia_Gray`** | `0.964 / 1.000` | `0.948 / 1.000` | `0.963 / 1.000` | `0.968 / 1.000` | `0.971 / 1.000` | `0.968 / 1.000` | `0.969 / 1.000` | **`0.973`** `/ 1.000` | **`0.973`** `/ 1.000` | **`0.973`** `/ 0.999` |
| **`Coverage`** | `0.300 / 0.765` | `0.396 / 0.864` | **`0.514`** `/ 0.920` | `0.434 / 0.873` | `0.456 /` **`0.931`** | `0.393 / 0.857` | `0.403 / 0.859` | `0.461 / 0.894` | `0.464 / 0.912` | `0.479 / 0.914` |
| **`IMD2020_RealLife`** | `0.483 / 0.921` | `0.566 / 0.901` | **`0.748`** `/ 0.937` | `0.615 / 0.918` | `0.681 / 0.936` | `0.633 / 0.934` | `0.656 / 0.934` | `0.722 / 0.936` | `0.722 /` **`0.937`** | `0.733 /` **`0.937`** |
| **`NIST16`** | `0.514 /` **`0.882`** | `0.461 / 0.686` | `0.476 / 0.811` | `0.558 / 0.846` | `0.543 / 0.807` | `0.473 / 0.751` | `0.488 / 0.760` | `0.553 / 0.831` | `0.557 / 0.828` | **`0.563`** `/ 0.828` |
| **`SAGI-D`** | `0.556 / 0.897` | `0.509 / 0.853` | `0.363 / 0.800` | `0.568 / 0.917` | **`0.667`** `/ 0.905` | `0.619 /` **`0.919`** | `0.623 / 0.918` | `0.600 / 0.906` | `0.605 / 0.899` | `0.616 / 0.899` |
| **`CocoGlide`** | `0.246 / 0.537` | `0.210 / 0.593` | `0.354 /` **`0.726`** | `0.256 / 0.528` | `0.361 / 0.545` | `0.295 / 0.579` | `0.298 / 0.585` | `0.365 / 0.636` | `0.370 / 0.634` | **`0.377`** `/ 0.633` |
| **`HiFi_IFDL`** | `0.340 / 0.642` | `0.272 / 0.810` | `0.285 / 0.793` | `0.383 / 0.788` | **`0.479 / 0.871`** | `0.446 / 0.871` | `0.416 /` **`0.872`** | `0.391 / 0.765` | `0.390 / 0.766` | `0.371 / 0.767` |
| **`OpenSDID_Flux.1`** | `0.283 /` **`0.787`** | `0.273 / 0.730` | `0.288 / 0.768` | `0.302 / 0.740` | `0.301 / 0.760` | **`0.311`** `/ 0.775` | `0.307 / 0.778` | `0.308 / 0.748` | `0.306 / 0.758` | `0.298 / 0.760` |
| **`OpenSDID_SD1.5`** | `0.075 / 0.585` | `0.065 / 0.599` | `0.062 / 0.583` | `0.058 /` **`0.636`** | `0.066 / 0.612` | **`0.088`** `/ 0.592` | **`0.088`** `/ 0.595` | `0.084 / 0.602` | `0.083 / 0.610` | `0.074 / 0.611` |
| **`OpenSDID_SDXL`** | `0.122 / 0.592` | `0.089 / 0.631` | `0.090 / 0.630` | `0.119 / 0.627` | **`0.154 / 0.670`** | `0.124 / 0.615` | `0.113 / 0.611` | `0.118 / 0.637` | `0.116 / 0.640` | `0.110 / 0.639` |
| **`COCO_Inpaint`** | **`0.075 / 0.640`** | `0.050 / 0.466` | `0.014 / 0.457` | `0.027 / 0.498` | `0.062 / 0.622` | `0.025 / 0.538` | `0.020 / 0.543` | `0.016 / 0.462` | `0.013 / 0.504` | `0.011 / 0.506` |
| **`IMD2020`** | `0.039 /` **`0.658`** | `0.029 / 0.515` | `0.028 / 0.528` | **`0.048`** `/ 0.624` | `0.030 / 0.588` | `0.037 / 0.538` | `0.038 / 0.544` | `0.042 / 0.554` | `0.042 / 0.567` | `0.042 / 0.568` |
| **`TGIF2_flux1dev`** | `0.000 / 1.000` | `0.000 / 1.000` | `0.000 / 1.000` | `0.000 / 1.000` | `0.000 / 1.000` | `0.000 / 1.000` | `0.000 / 1.000` | `0.000 / 1.000` | `0.000 / 1.000` | `0.000 / 1.000` |
| **MEAN (14 pos)** | `0.418 / 0.779` | `0.404 / 0.757` | `0.435 / 0.782` | `0.444 / 0.785` | **`0.478 / 0.803`** | `0.450 / 0.783` | `0.451 / 0.785` | `0.470 / 0.783` | `0.471 / 0.790` | `0.471 / 0.790` |

*(Each cell reports continuous `IoU / AUROC` on balanced `N=20` per dataset, `N=7` on `SAGI-D`.)*

Look at the headline takeaway across the 14 positive subsets: **`ZS_768TTA` (zero-shot 768 ViT + HRNet + hflip TTA, with 0 trained adapter/refiner weights!) achieves `0.478` IoU / `0.803` AUROC**, beating both adapted `PR34_fin` (`0.451 / 0.785`) and `PR35_fin` (`0.471 / 0.790`) on equal sample footing—because PR #35's gains (`+0.028 CASIA_V2`, `+0.052 IMD2020_RealLife`, `+0.023 Coverage`, `+0.020 NIST16`) are offset by adapter/refiner OOD regression (`-0.108 HiFi_IFDL`, `-0.051 COCO_Inpaint`, `-0.044 OpenSDID_SDXL`, `-0.051 SAGI-D`).

### 6.2 Failure Mechanism Decomposition (`Oracle` vs `Semantic Hijack` vs `Polarity Flip` vs `Refiner Bias`)

**Table 6.2 — Geometric & Causal Telemetry on `PR35_ad` / `PR35_fin`:**

| Dataset | `OrcAUC` | `Fisher` | `GTsep` | `KMsep` | `GT/KM` | `AxisAl` | `PolSgn` | `Ad IoU` | `+O_Pol` | `Fin IoU` | `Fin+O_Pol` | `Fin AUC` | `Fin+O_PolAUC` | `V~H IoU` | `r35Bias` | `r35Scale` | `sep_min` | `adDrift` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `CASIA_V1` | `1.000` | `60.64` | `0.532` | `0.533` | `1.00` | `1.000` | `+1.000` | `0.982` | `0.982` | `0.982` | `0.982` | `1.000` | `1.000` | `0.957` | `+0.035` | `0.980` | `0.516` | `0.0250` |
| `CASIA_V2` | `1.000` | `44.20` | `0.488` | `0.488` | `1.00` | `1.000` | `+1.000` | `0.969` | `0.969` | `0.970` | `0.970` | `1.000` | `1.000` | `0.901` | `-0.053` | `1.025` | `0.255` | `0.0234` |
| `Columbia_Gray` | `0.999` | `50.85` | `0.516` | `0.521` | `0.99` | `1.000` | `+1.000` | `0.973` | `0.973` | `0.973` | `0.973` | `0.999` | `0.999` | `0.963` | `-0.114` | `1.056` | `0.499` | `0.0203` |
| `Coverage` | `0.982` | `2.45` | `0.154` | `0.164` | `0.94` | `0.816` | `+0.772` | `0.461` | `0.490` | `0.479` | `0.506` | `0.914` | `0.920` | `0.313` | `-0.203` | `1.102` | `0.039` | `0.0275` |
| `IMD2020_Real` | `0.993` | `9.68` | `0.306` | `0.295` | `1.04` | `0.968` | `+0.868` | `0.722` | `0.770` | `0.733` | `0.780` | `0.937` | `0.986` | `0.439` | `-0.203` | `1.102` | `0.056` | `0.0222` |
| `NIST16` | `0.986` | `6.45` | `0.199` | `0.266` | `0.75` | `0.760` | `+0.644` | `0.553` | `0.644` | `0.563` | `0.656` | `0.828` | `0.890` | `0.365` | `-0.204` | `1.102` | `0.039` | `0.0233` |
| `SAGI-D` | `0.991` | `7.54` | `0.195` | `0.211` | `0.92` | `0.887` | `+0.776` | `0.600` | `0.652` | `0.616` | `0.668` | `0.899` | `0.956` | `0.366` | `-0.204` | `1.102` | `0.056` | `0.0310` |
| **`CocoGlide`** | `0.975` | `1.90` | `0.131` | `0.157` | `0.84` | **`0.823`** | **`+0.230`** | `0.365` | **`0.607`** | `0.377` | **`0.619`** | `0.633` | **`0.895`** | `0.380` | `-0.141` | `1.070` | `0.035` | `0.0388` |
| **`HiFi_IFDL`** | `0.977` | `1.12` | `0.087` | `0.088` | `0.99` | **`0.853`** | **`+0.480`** | `0.391` | **`0.510`** | `0.371` | **`0.485`** | `0.767` | **`0.928`** | `0.377` | `-0.187` | `1.094` | `0.035` | **`0.0410`** |
| `OpenSDID_Flux` | `0.951` | `1.19` | `0.087` | `0.157` | `0.56` | `0.737` | `+0.445` | `0.308` | `0.376` | `0.298` | `0.369` | `0.760` | `0.843` | `0.327` | `-0.205` | `1.102` | `0.041` | `0.0264` |
| `OpenSDID_SD1.5` | `0.933` | `0.42` | `0.034` | `0.135` | **`0.25`** | `0.549` | `+0.070` | `0.084` | **`0.224`** | `0.074` | **`0.217`** | `0.611` | `0.730` | `0.276` | `-0.205` | `1.103` | `0.029` | `0.0236` |
| `OpenSDID_SDXL` | `0.927` | `0.59` | `0.050` | `0.143` | **`0.35`** | `0.538` | `+0.167` | `0.118` | **`0.212`** | `0.110` | **`0.211`** | `0.639` | `0.744` | `0.321` | `-0.205` | `1.102` | `0.042` | `0.0269` |
| `COCO_Inpaint` | `0.932` | **`0.30`** | `0.020` | `0.073` | **`0.27`** | `0.314` | `-0.117` | `0.016` | `0.056` | `0.011` | `0.054` | `0.506` | `0.666` | **`0.094`** | `-0.206` | `1.103` | `0.041` | `0.0188` |
| `IMD2020` | `0.962` | `0.81` | `0.057` | `0.146` | **`0.39`** | `0.544` | `+0.032` | `0.042` | `0.052` | `0.042` | `0.052` | `0.568` | **`0.782`** | `0.187` | `-0.198` | `1.099` | **`0.023`** | `0.0273` |
| `TGIF2_flux1dev` | — | — | — | `0.327` | — | — | — | `0.000` | `0.000` | `0.000` | `0.000` | `1.000` | `1.000` | `0.690` | `-0.206` | `1.103` | `0.067` | `0.0153` |

Two distinct, orthogonal clustering failure mechanisms jump straight out of Table 6.2:
1. **Cluster Polarity Inversion (`CocoGlide`, `HiFi_IFDL`, `OpenSDID_SD1.5/SDXL`)**:
   - On `CocoGlide` and `HiFi_IFDL`, the unsupervised clustering hyperplane already aligns well with the true tampering axis (`AxisAl = 0.823` and `0.853`; `GT/KM = 0.84` and `0.99`), YET **`PolSgn` is only `+0.230` and `+0.480`** because FOCAL assigns whichever cluster has fewer pixels as "tampered"! Either genuine manipulations occupy $>50\%$ (`15.6%` of `CocoGlide` has `mask_ratio > 0.5`) OR background texture splits make the untouched region the smaller cluster.
   - Resolving polarity alone (via a lightweight learned cluster-polarity classifier instead of `c1_count > c0_count`) lifts **`CocoGlide` AUROC `0.633 -> 0.895` (+0.262)** & **IoU `0.377 -> 0.619` (+0.242)**, **`HiFi_IFDL` AUROC `0.767 -> 0.928` (+0.161)** & **IoU `0.371 -> 0.485` (+0.114)**, **`OpenSDID_SD1.5` IoU `0.074 -> 0.217` (2.9x)**, and **`OpenSDID_SDXL` IoU `0.110 -> 0.211` (1.9x)**!
2. **Semantic Variance Hijacking (`COCO_Inpaint`, `OpenSDID_SD1.5`, `OpenSDID_SDXL`, `IMD2020`)**:
   - Here `GTsep / KMsep = 0.25–0.39` and `Fisher = 0.30–0.81`: non-forensic scene semantics (sky vs. grass, foreground vs. background) produce $2.5\times\text{--}4\times$ larger feature distance than the subtle generative artifact! Consequently ViT and HRNet disagree wildly on what two clusters to segment (`V~H IoU = 0.094–0.321` vs. `0.901–0.963` on splicing), and even oracle-polarity cannot recover high IoU without fixing Section 3 (`blk_14..blk_23` vs `neck`).

### 6.3 Dead Pristine Gate & Hard `kmeans` Corner-Pixel Asymmetry

- Across all `287` probed images (including completely blank-GT `TGIF2_flux1dev`, where semantic segmentation gives mean `sep35 = 0.327`, minimum `0.067`), the global minimum `sep35_ad` observed was **`0.023`** (`IMD2020`), which is **`15x` larger** than `cluster_separability_threshold = 0.0015`. The pristine gate triggered on **`0 / 287` (`0.0%`)** samples!
- Furthermore, in `sid_unet/models/focal.py`, hard `kmeans` mode still initializes centroids starting from top-left corner pixel `(0, 0)` (`f_img[:, 0:1, :]`). While PR #35 tested horizontal-flip equivariance (`test_tta_hflip_equivariance_and_vit_input_size`) under `soft_kmeans`, standard `kmeans` uses an asymmetric corner seed that also makes centroids sensitive to border padding or single-pixel artifacts.

---

## 7. Evaluation Pipeline Bugs & `MVSS-v1` Dataset Integrity Audit

### 7.1 Post-Processing Binary AUROC Collapse (`-0.11` Drop & Ranking Reversal)

In `configs/cross-eval/MVSS-v1-protocol.yaml`, `post_processing.enabled` is `true` (`min_area: 64, fill_holes: true, morphology: open_close`).
Previously, both `sid_unet/evaluate.py` (`eval_single_batch`) and `sid_unet/cross_eval.py` (`eval_single_batch`) passed **hard binarized `{0.0, 1.0}` masks** (`post_masks`) into `post_seg_tracker.update(..., is_logit=False)`, and then selected `post_seg` as `chosen_seg` for the headline table. Inside `compute_binary_metrics_batch`:
1. Binary inputs trigger `auroc = (recall + specificity) / 2.0` (single-threshold Balanced Accuracy rather than continuous pixel ranking ROC-AUC).
2. Additionally, for `n_pos == 0` samples (`fp > 0`, e.g. `TGIF2_flux1dev`), `compute_binary_auroc_batch` returns `1.0 if mean_p < 0.5 else 1 - mean_p` (`1.000` on `TGIF2`), whereas the binary fast-path returned `specificity / 2 = 0.4549`!

Exact distortion across all `17,929` samples of `MVSS-v1-protocol` (verified directly in the published HF JSON reports):

| Model (Full `17,929` Samples) | Metric Stage | Mean IoU | Dice / F1 | **Headline AUROC** | Precision | Recall | Specificity |
|---|---|---:|---:|---:|---:|---:|---:|
| **PR #34 (`hybrid-casia`)** | `Baseline (Raw UNet)` | **`0.2641`** | `0.3206` | **`0.7399`** | `0.3314` | **`0.4309`** | `0.8831` |
| **PR #34 (`hybrid-casia`)** | `+ Post-Processing` *(reported)* | `0.2639` | `0.3206` | **`0.6332` (`-0.1067`)** | `0.3317` | `0.4307` | `0.8835` |
| **PR #35 (`focal-v2-imgrel`)** | `Baseline (Raw UNet)` | **`0.2759`** | **`0.3250`** | **`0.7431`** | `0.3440` | **`0.4159`** | `0.8903` |
| **PR #35 (`focal-v2-imgrel`)** | `+ Post-Processing` *(reported)* | `0.2755` | `0.3248` | **`0.6301` (`-0.1130`)** | `0.3443` | `0.4152` | `0.8910` |

This bug **artificially erased `~0.11` AUROC** from both PRs, masked the fact that post-processing actually slightly *lowered* IoU on both (`-0.0002` / `-0.0004`), and **reversed the true AUROC ordering** (`0.6301 < 0.6332` reported vs. **`0.7431 > 0.7399`** true continuous ROC-AUC!). *(Fixed in this PR).*

### 7.2 `MVSS-v1` (`KhangTruong/Tampered-Images-Combined-Dataset-Fake`) Data Integrity Findings

Scanning all `36` Parquet shards (`17,929` rows) revealed three major dataset anomalies:

1. **`TGIF2_flux1dev` (`1,986 / 17,929 = 11.1%`) Has 100% Blank All-Zero PNG Masks**:
   - All `1,986` rows across shards `20..24` contain `label = 1 (fake)` and hardcoded Parquet `mask_ratio = 0.1500`, yet **every single decoded PNG mask (`1,986 / 1,986`) has `min = 0, max = 0, unique = [0]`** (`pos_count = 0` in the HF reports!).
   - Because `SegmentationMetricTracker(filter_zero_positive=True)` excludes zero-positive images from `overall` positive IoU (`pos = 15,943 / 17,929`), overall IoU omits `TGIF2`, while per-dataset metrics fall back to `iou_all = 0.0000`, `f1_all = 0.0000`, `recall_all = 0.0000` (and with the old binary AUROC bug, `auroc = 0.4549`). This explains why `TGIF2` scored `0.0000` for every model—including `PR #34`, `PR #35`, and oracle!
2. **`COCO_Inpaint` (`N = 2,000`) Mod-8 Diffusion Padding Dimension Mismatch (`60.9%`)**:
   - Across shards 4–5 (`727` rows), `443 / 727` (`60.9%`) have image dimensions larger than mask dimensions; inspecting `101` rows in shard 4 shows `66 / 101` (`65.3%`) mismatched where `100%` of padded images are divisible by 8 with `dw = img_w - mask_w in [0, 7]` and `dh = img_h - mask_h in [0, 7]` (e.g. image `(640, 432)` vs mask `(640, 425)`, or `(504, 336)` vs `(500, 332)`). Additionally, `sid_unet/dataset/mask_utils.py` resizes via `Image.NEAREST` instead of matching the VAE padding alignment (note: on small `<1%` masks this introduces a minor shift relative to the dominant `neck[0..3]` representation failure).
3. **`IMD2020` (`N = 2,000`, `mean = 0.0129`, `Med = 0.0126`) vs `IMD2020_RealLife` (`N = 1,000`, `mean = 0.0811`)**:
   - In this benchmark build, `IMD2020` (`method='Inpainting'`) consists exclusively of compact `~100x100` diffusion inpaint patches around ~1.3% of image area (`90%` quantile `0.0148`), explaining why `32x32`/`48x48` ViT grids struggle far more than on `IMD2020_RealLife`.

---

## 8. Code Fixes & Verification Summary

### 8.1 Fixes Included in This Repository (`ai-detection-unet-gpu`)

| File | Fix Applied | Verification |
|---|---|---|
| [`sid_unet/metrics/segmentation.py`](../sid_unet/metrics/segmentation.py) | Fixed `compute_binary_metrics_batch` single-class binary AUROC fast-path to match `compute_binary_auroc_batch`; added `score_masks`/`score_is_logit` support + `binary_auroc` preservation in `SegmentationMetricTracker`. | Covered by `test_binary_auroc_fastpath_matches_continuous_on_single_class_masks` & `test_segmentation_tracker_preserves_continuous_auroc_with_score_masks`. |
| [`sid_unet/cross_eval.py`](../sid_unet/cross_eval.py), [`sid_unet/evaluate.py`](../sid_unet/evaluate.py) | Forwarded continuous `mask_logits` as `score_masks` into `post_seg_tracker`, binary `sam_seg_tracker`, and `both_seg_tracker` so post-processing/SAM thresholding no longer destroys continuous ROC-AUC (`-0.11` fix). | Covered by `test_evaluate_and_cross_eval_preserve_continuous_auroc_under_postprocessing`. |
| [`sid_unet/models/focal.py`](../sid_unet/models/focal.py) | Anchored `cluster_refiner_l2_weight` for 1D normalization scales (`GroupNorm`, `BatchNorm`, `LayerNorm`, `InstanceNorm`) around `1.0` (`(p - 1.0)^2`) instead of pulling `gamma -> 0`; fixed `_ensure_raw_rgb` to denormalize **per sample** (`amin`/`amax` over `(1,2,3)`) rather than across the whole batch. | Covered by `test_cluster_refiner_l2_weight_anchors_groupnorm_to_one_not_zero` & `test_ensure_raw_rgb_is_batch_independent`. |
| [`docs/focal_v2_image_relative_casia_adaptation.md`](../docs/focal_v2_image_relative_casia_adaptation.md) | Clarified `tta_hflip` averages L2-normalized backbone *features* rather than logits. | Manual doc inspection. |

### 8.2 Fixes & Cross-Domain Tools Added in Companion Repo [`khangtruong-ui/nn-toolbox#1`](https://github.com/khangtruong-ui/nn-toolbox/pull/1)

- Added `inspect_pretrained_weight_health` (auto-reproduced HRNet subnormal BN blocks `5.605e-45`, `cluster_refiner.out_conv` `-0.9942` collinearity, ViT `blocks.0.attn.proj` `cond=57998.3`, and `blocks.23.norm1` negative scale/bias live on `focal-v2-image-relative-casia`).
- Added `batchnorm_domain_shift_test` and `layer_representation_discrepancy_test` (with strict spatial split-half CV AUROC and `GT/KM` bottleneck detection).
- Fixed `diagnose.py` missing `input_var`, `"effective_rank_ratio"` key mismatch (`"rank_ratio"`), and dict-output model support (`skip_mask_loss: False` compatibility across `diagnose`, `train_eval`, `eval_determinism`, `ablation`, and `sensitivity`).

---

## 9. Actionable Suggested Changes vs. Structural Limitations

1. **Immediate Evaluation & Bug Fixes (Ready in This PR + Benchmark)**:
   - Use this PR's fixed `score_masks` continuous AUROC tracking so post-processing ablations no longer deflate AUROC by `-0.11`.
   - Exclude or repair the `1,986` all-zero `TGIF2_flux1dev` PNG masks in `KhangTruong/Tampered-Images-Combined-Dataset-Fake` (and align `COCO_Inpaint` mod-8 image padding vs mask crop/pad geometry rather than nearest-neighbor stretching).
2. **Highest-ROI Next Architectural Experiments (Zero New Pretraining Needed)**:
   - **Tap Pre-Neck ViT Blocks (`blk_17` / `blk_22` / `blk_23`) + Bypass/Debias `neck[3]`**: Adding a learnable multi-tap weighted mix of intermediate ViT blocks `[blk_17, blk_22, blk_23, neck_out]` (initialized cleanly and centered to break `neck[3]`'s `cone_cos = 0.777` bias cone) unlocks up to `+0.10`–`+0.18` zero-shot IoU headroom already present inside the frozen ViT (`HiFi_IFDL`, `SAGI-D`, `OpenSDID_SDXL`).
   - **Replace the `minority-is-tampered` (`c1_count > c0_count`) Heuristic with a Lightweight Learnable Cluster-Polarity / Ranking Head**: Directly addresses the `CocoGlide` (`+0.242` IoU / `+0.262` AUROC oracle gap) and `HiFi_IFDL` (`+0.114` IoU / `+0.161` AUROC) inversion failures using rotation-invariant cluster asymmetry descriptors (`area`, border-touch ratio, edge gradient density inside each cluster).
   - **Make `ImageRelativeClusterRefiner` Gated by `sep_f` & Fix Output Head Rank Collapse**: Gate residual shifts proportionally to `min(1, sep_f / 0.3)` (so low-separability OOD images revert smoothly to neutral bias `0.0` instead of receiving `-0.205` global suppression) and regularize `cos(w_bias, w_scale)` orthogonally to prevent rank-1 collapse.
   - **Test-Time BatchNorm Adaptation (Tent/BN-recal) on HRNet `stage4` & Pruning Dead `stage4.2.branches.0.{1..3}`**: Addresses the `2.8x` OOD `|log_var_ratio|` shift on `HiFi_IFDL` (`0.588` vs `0.212` CASIA) and `1.85x` shift on `COCO_Inpaint` (`0.392`).
3. **Structural Limitation Cannot Be Fixed by CASIA Tuning Alone**:
   - When `Fisher < 0.6` and `GTsep / KMsep <= 0.35` (`COCO_Inpaint`, `OpenSDID_SD1.5`, `OpenSDID_SDXL`, tiny-patch `IMD2020`), **no CASIA-only post-hoc adapter or unsupervised 2-means refiner can beat semantic variance hijacking**. Breaking above `0.25–0.35` IoU on seamless modern diffusion inpainting requires unfreezing/LoRA-adapting late ViT/HRNet representations on **paired multi-generator diffusion inpainting data** rather than solely traditional splicing/copy-move (`CASIA-v2.0`).
