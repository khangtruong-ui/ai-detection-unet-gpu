# MSCCNet & OmniFD Integration Summary

This document summarizes the integration of **MSCCNet** (ACM TOMM 2025) and **OmniFD** (arXiv 2025) into `sid_unet`, including architectural details, detection and localization branches, pretrained weight links, configuration profiles, and detailed 8-column PDF inference reporting.

---

## 1. Comparative Overview

| Model | Paper & Venue | Official Code & License | Detection Branch (Image-Level) | Localization Branch (Pixel-Level) | Pretrained Weights Status |
|---|---|---|---|---|---|
| **MSCCNet** | Miao et al., *Multi-spectral Class Center Network for Face Manipulation Localization*, **ACM TOMM**, 2025 ([DOI](https://doi.org/10.1145/3747296), [arXiv:2305.10794](https://arxiv.org/abs/2305.10794)) | [miaoct/MSCCNet](https://github.com/miaoct/MSCCNet) · MIT | `MSCCClassifier` (`AdaptiveAvgPool2d` + `Linear`) predicting 2-class real/fake logits (`class_logits`). | Multi-level dilated ResNet bottlenecks + `MSCCModule` (2D DCT `MultiSpectralDCTLayer`, `decoder_map`, `MSCCGCN`, spectral attention) + `decoder` (`mask_logits` & `fres_map_logits`). | Official repo does **not** release task-trained MSCCNet checkpoints; uses **ImageNet-pretrained** ResNet-50/101 `v1c` backbones (or explicit `checkpoint_path`). |
| **OmniFD** | Liu et al., *OmniFD: A Unified Model for Versatile Face Forgery Detection*, **arXiv preprint**, 2025 ([arXiv:2512.01128](https://arxiv.org/abs/2512.01128)) | [haotianll/OmniFD](https://github.com/haotianll/OmniFD) · MIT | `OmniFDImageClsHead` pooling the 64 shared Interaction latent tokens (`[B, 512, 64] -> [B, 2]`). | `OmniFDSpatialHead` fusing 4 multi-scale SwinTransformer3D stages (`2112 -> 256`) with projected Interaction latents (`einsum('bchw,bck->bkhw')` -> `[B, 1, H, W]`). *(Upstream also includes `VideoClsHead` & `TemporalHead` for video/temporal tasks).* | Official pretrained checkpoints hosted on **Google Drive** (folder + 8 checkpoints listed below), auto-downloaded via `gdown` or loaded from local `checkpoint_path`. |

---

## 2. Pretrained Weights & Resource Links

### 2.1 OmniFD Official Google Drive Links (`haotianll/OmniFD`)

- **Pretrained Weights Folder**: <https://drive.google.com/drive/folders/1Bh8FbF-jiyPtgr7-URNuGAOJhi8kZSEC?usp=sharing>
- **Processed Annotation Files Folder**: <https://drive.google.com/drive/folders/1auf-c_82SpKP0XfJlx9vJdcz7BiwLOy_?usp=sharing>

| Checkpoint Key | Filename | Google Drive Link | Backbone | Tasks Supported |
|---|---|---|---|---|
| `omnifd_swin-t_50k_forgerynet` | `omnifd_swin-t_50k_forgerynet.pth` | <https://drive.google.com/file/d/1ZJAHtTL21qaCTOSacxeZ-MkGTnrEY0BL/view> | Swin-T (`omnivore_swinT`) | Unified (`image`, `spatial`, `video`, `temporal`) |
| `omnifd_swin-s_50k_forgerynet` | `omnifd_swin-s_50k_forgerynet.pth` | <https://drive.google.com/file/d/14Y9hpsKlGlPLUh2dPFugssSXS4ftFXL2/view> | Swin-S (`omnivore_swinS`) | Unified (`image`, `spatial`, `video`, `temporal`) |
| `omnifd_swin-s_20k_ff_image` | `omnifd_swin-s_20k_ff_image.pth` | <https://drive.google.com/file/d/1RMvEDQD6Uzj2NBm9RgS6KA5bpblEXc41/view> | Swin-S (`omnivore_swinS`) | `image` (FaceForensics++) |
| `omnifd_swin-s_20k_ff_spatial` | `omnifd_swin-s_20k_ff_spatial.pth` | <https://drive.google.com/file/d/1oKXW99ZyNGRwnxwbC22AwJoCoDkrM-Ih/view> | Swin-S (`omnivore_swinS`) | `spatial` (FaceForensics++) |
| `omnifd_swin-s_20k_fmld_spatial` | `omnifd_swin-s_20k_fmld_spatial.pth` | <https://drive.google.com/file/d/1jiHcVpHhcdSBb7eqnBs_89tZ3UzAPIIk/view> | Swin-S (`omnivore_swinS`) | `spatial` (FMLD) |
| `omnifd_swin-s_20k_wdf_video` | `omnifd_swin-s_20k_wdf_video.pth` | <https://drive.google.com/file/d/1jAS3WIBgQj_9sUby8xwQzB-ZzqoR3DVf/view> | Swin-S (`omnivore_swinS`) | `video` (WildDeepfake) |
| `omnifd_swin-s_50k_ff_video` | `omnifd_swin-s_50k_ff_video.pth` | <https://drive.google.com/file/d/18YEN4E0OgCYUHucIaEaFPTWHDjpChga9/view> | Swin-S (`omnivore_swinS`) | `video` (FaceForensics++) |
| `omnifd_swin-s_50k_lavdf_temporal` | `omnifd_swin-s_50k_lavdf_temporal.pth` | <https://drive.google.com/file/d/1kevdiALBKD0JRigp3sPrAjA6h-rMBlId/view> | Swin-S (`omnivore_swinS`) | `temporal` (LAV-DF) |

Omnivore trunk initializations (used when `pretrained=False, backbone_pretrained=True`):
- `omnivore_swinT`: <https://dl.fbaipublicfiles.com/omnivore/models/swinT_checkpoint.torch>
- `omnivore_swinS`: <https://dl.fbaipublicfiles.com/omnivore/models/swinS_checkpoint.torch>
- `omnivore_swinB`: <https://dl.fbaipublicfiles.com/omnivore/models/swinB_checkpoint.torch>

### 2.2 MSCCNet Backbone Links (`miaoct/MSCCNet`)

- Official repo issues ([#1](https://github.com/miaoct/MSCCNet/issues/1), [#3](https://github.com/miaoct/MSCCNet/issues/3)) confirm that task-trained MSCCNet weights were not publicly released; `scripts/distest_vis.sh` expects a user-trained `./epoch_best.pth`.
- ImageNet-pretrained backbone weights used for MSCCNet initialization:
  - `resnet50stem` (`resnet50_v1c`): <https://download.openmmlab.com/pretrain/third_party/resnet50_v1c-2cccc1ad.pth>
  - `resnet101stem` (`resnet101_v1c`): <https://download.openmmlab.com/pretrain/third_party/resnet101_v1c-e67eebb6.pth>
  - Standard ResNet-50 / ResNet-101 (`use_stem=false`): PyTorch `model_zoo` URLs.

---

## 3. Code Integration Architecture

- **`sid_unet/models/msccnet.py`**: Pure-PyTorch implementation of `MSCCNet`, `MSCCResNet`, `MSCCClassifier`, `MultiSpectralDCTLayer` (with dynamic resolution interpolation for arbitrary input sizes), `MSCCGCN`, and `MSCCModule`. Matches upstream `miaoct/MSCCNet` state-dict keys (`backbone_net`, `classifer`, `bottleneck1..3`, `bottleneck_fusion`, `mscc`, `decoder`) and implements SID-UNet's `(mask_logits, class_logits)` and `return_dict=True` contracts.
- **`sid_unet/models/omnifd.py`**: Pure-PyTorch implementation of `OmniFD`, `SwinTransformerOmni` (`SwinTransformer3D`), `InteractionNeck` (`InteractionModule` + `InteractionAttention`), `OmniFDImageClsHead`, and `OmniFDSpatialHead` without requiring `mmcv`/`mmengine`/`mmpretrain`. Matches official `omnifd_swin-*.pth` checkpoint keys 1-to-1 (`backbone.trunk.*`, `neck.*`, `head.decoder_modules.image.*`, `head.decoder_modules.spatial.*`).
- **`sid_unet/models/unet.py`**: Updated `build_model()` and `UNet.from_checkpoint()` to construct and restore both `msccnet` and `omnifd` checkpoints (including raw state-dicts).
- **Configs**:
  - `configs/experiments/msccnet/default.yaml`, `configs/experiments/msccnet/msccnet_resnet101.yaml`
  - `configs/experiments/omnifd/default.yaml`, `configs/experiments/omnifd/omnifd_swin_s_forgerynet.yaml`, `configs/experiments/omnifd/omnifd_swin_s_ff_spatial.yaml`, `configs/experiments/omnifd/omnifd_swin_s_fmld_spatial.yaml`
- **Detailed 8-Column PDF Inference Reporting**:
  - Added `plot_detailed_inference_pdf()` in `sid_unet/utils/plotting.py` and wired it into `sid-illu` (`sid_unet/illustration.py`, `--pdf` enabled by default) as well as `scripts/infer_msccnet_omnifd_pdf.py`.
  - Produces a multi-page PDF:
    - **Page 1**: Model & pretrained weights inventory table, aggregate quantitative metrics table, and comparative bar chart.
    - **Pages 2+**: **8 columns** per sample-model row:
      1. Input Image (with resolution & GT class)
      2. Ground Truth Mask (with tampered %)
      3. Continuous Probability Heatmap `[0, 1]` (`min / mean / max`)
      4. Binary Predicted Mask (`IoU`, `Dice/F1`, predicted %)
      5. Auxiliary / Spectral Residual / Post-Processed Map (`MSCC Spectral Map` for MSCCNet; `Post-Processed Mask` for OmniFD)
      6. Red Mask Overlay on Input Image (with inference latency)
      7. Pixel Error Map (`TP=Green, FP=Red, FN=Blue` + `Pixel Acc`, `Precision`, `Recall`)
      8. Image-Level Real/Fake Detection Bar & Pixel Probability Histogram

---

## 4. Usage Examples

```bash
# Train MSCCNet (ResNet-50)
sid-train --config configs/experiments/msccnet/default.yaml

# Train / Evaluate OmniFD (Swin-T ForgeryNet 50k)
sid-train --config configs/experiments/omnifd/default.yaml

# Run inference across OmniFD & MSCCNet and produce detailed 8-column PDF
python scripts/infer_msccnet_omnifd_pdf.py \
    --output-pdf docs/msccnet_omnifd_inference_report.pdf \
    --num-samples-per-dataset 2

# Run qualitative multi-model illustration + 8-column PDF via sid-illu
sid-illu --model-ckpts path/to/ckpt.pt --dataset-configs configs/cross-eval/casia_v2.0.yaml --pdf
```
