# FOCAL Domain-Split (4-Class / 4-Cluster) & Dual-Backbone Fusion

**Target Repository**: `khangtruong-ui/ai-detection-unet-gpu`  
**Scope**: `sid_unet/models/focal.py`, `sid_unet/dataset/loader.py`, `sid_unet/training/trainer.py`, `sid_unet/models/unet.py`, `configs/experiments/focal/*.yaml`

---

## 1. Domain-Split 4-Class / 4-Cluster Finetuning & 2-Class Inference

### Problem & Motivation
Combining `CASIAv2.0` and `IMD2020` (`KhangTruong/IMD2020-CASIAv2-Joint`, which provides a `dataset_source` column with `"casia"` and `"imd2020"`) under a single shared `pristine` / `tampered` representation forces two incompatible camera/compression/manipulation domains together, causing catastrophic forgetting of FOCAL's strong pretrained `CASIAv2.0` prior.

### Solution
During **training** (`domain_split_mode: true`), `FOCAL` splits classes / clusters by `(dataset_source, tampered_state)`:
- `0`: `casia_pristine` (CASIA pristine)
- `1`: `casia_tampered` (CASIA tampered)
- `2`: `imd_pristine` (IMD2020 pristine)
- `3`: `imd_tampered` (IMD2020 tampered)

During **inference** (`eval()` / `predict()` / `cross_eval()`), **no `dataset_source` is required**, and the 4 domain-split logits/clusters collapse into **strictly 2 classes (`pristine` vs `tampered`)** via exact log-sum-exp:
$$\text{logit}_{\text{tampered}} = \operatorname{logsumexp}(z_1, z_3) - \operatorname{logsumexp}(z_0, z_2)$$
$$\sigma(\text{logit}_{\text{tampered}}) = P(c=1) + P(c=3)$$

#### Two Supported Modes
1. **Mode 1A — Supervised (`finetune_mode: "supervised"`)**:
   - `FOCALSupervisedHead` predicts 4 channels `[B, 4, H, W]`, supervised by 4-class pixel cross-entropy (`domain_loss_weight`) plus collapsed binary `CombinedMaskLoss` (BCE + Dice + Edge).
   - At inference, collapses `[B, 4, H, W]` $\to$ `mask_logits [B, 1, H, W]`, `raw_logits [B, 2, H, W]`, `pred_mask [B, 1, H, W]`.
2. **Mode 1B — Contrastive + Clustering (`finetune_mode: "contrastive"`)**:
   - Maintains 4 learnable unit-normalized prototypes `domain_prototypes [4, C]` initialized from the first seen centroid of each domain-class.
   - Optimizes intra-image `MyInfoNCE` + 4-class Prototype InfoNCE (`domain_proto_weight`) + cross-image 4-class SupCon (`domain_supcon_weight`).
   - At inference, computes cosine similarity against the 4 domain-split prototypes and collapses `{0, 2} -> pristine` and `{1, 3} -> tampered` into 2-class outputs.

### Configurations (2 per Existing Backbone)
| Config File | Backbone | Mode | Dataset |
| :--- | :--- | :--- | :--- |
| `configs/experiments/focal/focal_hrnet_domain_split.yaml` | `hrnet` | Supervised (4-class $\to$ 2-class) | `KhangTruong/IMD2020-CASIAv2-Joint` |
| `configs/experiments/focal/focal_hrnet_contrastive_domain_split.yaml` | `hrnet` | Contrastive (4-cluster $\to$ 2-class) | `KhangTruong/IMD2020-CASIAv2-Joint` |
| `configs/experiments/focal/focal_vit_domain_split.yaml` | `vit` (DoRA PEFT) | Supervised (4-class $\to$ 2-class) | `KhangTruong/IMD2020-CASIAv2-Joint` |
| `configs/experiments/focal/focal_vit_contrastive_domain_split.yaml` | `vit` (DoRA PEFT) | Contrastive (4-cluster $\to$ 2-class) | `KhangTruong/IMD2020-CASIAv2-Joint` |

---

## 2. Independent Dual-Backbone Fusion

### Architecture
`FOCAL` supports fusing a **frozen primary pretrained FOCAL backbone** (`backbone`: `"hrnet"` or `"vit"`, locked when `freeze_primary_backbone: true`) with an **independent secondary backbone** (`secondary_backbone`):
- Complementary FOCAL pretrained backbones: `"vit"` or `"hrnet"` (loaded from `KhangTruong/FOCAL` or `secondary_checkpoint_path`, optionally PEFT/stage-adapted via `secondary_peft_enabled` / `secondary_trainable_stages`).
- Pretrained `timm` vision backbones: `"timm:<model_name>"` (multi-scale FPN projection with internal ImageNet standardization on raw `[0, 1]` inputs).

### Fusion Modes (`fusion_mode`)
- `"concat"`: L2-normalized channel concatenation `[F_norm(f_primary), F_norm(f_secondary)]` (`C1 + C2`, FOCAL paper style).
- `"proj_concat"`: L2-normalized concatenation projected via `1x1 Conv -> GroupNorm -> GELU -> 1x1 Conv` to `fusion_out_dim`.
- `"zero_gated"`: Zero-initialized residual projection `f_primary + zero_proj(f_secondary)` so fused features equal `f_primary` identically at step 0.

Spatial resolution is aligned via `fusion_spatial_align: "primary" | "max"`.

### Configurations
| Config File | Primary (Frozen) | Secondary (Trainable) | Fusion Mode | Mode |
| :--- | :--- | :--- | :--- | :--- |
| `configs/experiments/focal/focal_hrnet_vit_fusion.yaml` | `hrnet` | `vit` (DoRA PEFT) | `zero_gated` (`primary` align) | Supervised |
| `configs/experiments/focal/focal_vit_hrnet_fusion_contrastive.yaml` | `vit` | `hrnet` (`stage4`) | `concat` (`max` align) | Contrastive |
