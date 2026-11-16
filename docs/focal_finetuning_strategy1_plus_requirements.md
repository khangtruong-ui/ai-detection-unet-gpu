# Requirements Specification: FOCAL Strategy 1+ — Anchored Full-Rank Finetuning (`512×512`)

**Author**: Antigravity Assistant  
**Date**: November 16, 2026  
**Target Repository**: `khangtruong-ui/ai-detection-unet-gpu`  
**Status**: Implemented & Verified (PR #26)  
**Predecessors**:  
- Strategy 1: [`docs/focal_finetuning_strategy1_requirements.md`](focal_finetuning_strategy1_requirements.md) (PR #24, `73e4a11`)  
- Strategy B: [`docs/focal_finetuning_strategy_b_requirements.md`](focal_finetuning_strategy_b_requirements.md) (PR #25, `998f3bd`)  
**Scope**: `sid_unet/models/focal.py`, `sid_unet/models/unet.py`, `sid_unet/losses/auxiliary.py`, `sid_unet/training/trainer.py`, `configs/experiments/focal/*.yaml`

---

## 1. Executive Summary & Motivation

Empirical evaluation of Strategy 1 (PR #24) and Strategy B (PR #25) at `512×512` established two opposite failure modes:
- **Strategy 1 (Unconstrained Full-Rank Finetuning)** learned strongly on target datasets, but unfreezing all backbone layers uniformly with zero weight/feature anchoring caused **prior drift** (erasing pretrained SAM/FOCAL forensic representations).
- **Strategy B (PEFT LoRA/DoRA + Uniform Prior Anchor)** preserved priors, but **under-learned** because:
  1. `FOCAL._compute_prior_anchor_terms()` (`sid_unet/models/focal.py`, line 1642) applied `(1.0 - (f_ad * f_ba).sum(dim=1)).mean()` **uniformly across all pixels (both pristine and tampered)**, directly fighting `MyInfoNCE` / BCE on newly tampered pixels.
  2. Low-rank `r=8` `qkv/proj` adapters on 9 of 24 ViT blocks (with MLPs frozen) + `neck_lr_multiplier=0.25` (`1.25e-5` effective LR) + HRNet `stage4`-only freezing (`1.25e-5` LR at `batch_size=128`) throttled learning capacity too aggressively.

**Strategy 1+ ("Anchored Full-Rank Finetuning")** builds directly on **Strategy 1**'s full-rank learning capacity while eliminating prior drift through **four cohesive mini-strategies**:
1. **Mini-Strategy 1**: Asymmetric **Pristine-Only** Feature Anchoring (`prior_anchor_mode: "pristine_only"`).
2. **Mini-Strategy 2**: **Layer-Wise Learning Rate Decay (LLRD)** + Permanent Stem / Positional-Embedding / `BatchNorm2d` Lock.
3. **Mini-Strategy 3**: **L2-SP** Starting-Point Weight Regularization (replacing decay-to-zero on pretrained backbone weights).
4. **Mini-Strategy 4**: **Hybrid Joint Supervised + Contrastive Mode** (`finetune_mode: "hybrid"`), preserving FOCAL's L2-normalized angular hypersphere during supervised decoder training.

---

## 2. Shared `512×512` Input & Backbone Invariants

All Strategy 1+ configurations **must** preserve the `512×512` preprocessing invariants established in Strategy 1:
- `data.image_size: [512, 512]`
- `data.normalize: false` (raw `[0.0, 1.0]` RGB tensors, with `_ensure_raw_rgb()` kept as defensive fallback)
- `model.force_vit_1024: false` (native `512×512` `32×32` ViT token grid without `512→1024` upsampling waste)
- Strict `pretrained=True` zero-fallback weight loading (`KhangTruong/FOCAL`) executed **before** any parameter-freeze masks or weight snapshots are created.

---

## 3. Mini-Strategy 1: Asymmetric Pristine-Only Feature Anchoring

### 3.1 Root Defect in Uniform Anchoring
In `sid_unet/models/focal.py` (line 1642), uniform cosine anchoring:
$$\mathcal{L}_{\text{uniform}} = \lambda_{\text{anchor}} \cdot \frac{1}{B H' W'} \sum_{b,h,w} \left(1 - \left\langle \hat{F}[b,:,h,w], \, \text{sg}\!\left(\hat{F}_{\text{base}}[b,:,h,w]\right) \right\rangle\right)$$
penalizes representation movement on **forged pixels** ($M = 1$) just as strongly as on **pristine pixels** ($M = 0$). Whenever a pretrained model fails to separate a new forgery type, $\mathcal{L}_{\text{uniform}}$ suppresses the exact feature shift needed to detect the manipulation.

### 3.2 Pristine-Only Masked Cosine Anchor Specification
Add `model.prior_anchor_mode: str = "pristine_only"` (supporting `"pristine_only"` and legacy `"uniform"`):

1. **Conservative Downsampled Pristine Mask (`m_pristine`)**:
   - Given `gt_masks` of shape `[B, 1, H, W]` (binary `{0, 1}`), downsample to the backbone feature spatial grid `(H', W')` using area averaging (`F.interpolate(gt_masks.float(), size=(h_f, w_f), mode="area")`).
   - Identify **purely pristine** feature tokens using a strict threshold `pristine_anchor_threshold` (default `0.05`):
     $$m_{\text{pristine}}[b, h, w] = \mathbb{I}\!\left(\text{gt\_small}[b, 0, h, w] \le \tau_{\text{pristine}}\right)$$
     *(Using `<= 0.05` instead of `< 0.5` ensures partially tampered boundary tokens are excluded from the anchor and remain free to separate.)*
2. **Masked Loss Formulation**:
   $$\mathcal{L}_{\text{pristine-anchor}} = \lambda_{\text{anchor}} \cdot \frac{\sum_{b,h,w} m_{\text{pristine}}[b,h,w] \cdot \left(1 - \left\langle \hat{F}[b,:,h,w], \, \text{sg}\!\left(\hat{F}_{\text{base}}[b,:,h,w]\right) \right\rangle\right)}{\sum_{b,h,w} m_{\text{pristine}}[b,h,w] + \epsilon}$$
   where $\hat{F} = \text{normalize}(F, p=2, \text{dim}=1)$, $\hat{F}_{\text{base}} = \text{normalize}(F_{\text{base}}, p=2, \text{dim}=1)$, and $\epsilon = 10^{-6}$.
3. **Edge-Case Safety**:
   - If `gt_masks is None`: if `prior_anchor_mode == "pristine_only"`, fall back cleanly to `"uniform"` or return `0.0` (configurable; returning `0.0` with a debug warning avoids accidentally anchoring tampered pixels when masks are absent).
   - If an entire batch has zero pristine tokens ($\sum m_{\text{pristine}} == 0$, e.g., synthetic full-image deepfakes where every pixel is labeled `1`): return a differentiable zero (`feats.sum() * 0.0`) with **zero `NaN`/`Inf`**.

### 3.3 Full-Rank Reference Feature Extraction (`_anchor_backbone`)
Because Strategy 1+ unfreezes `self.net` directly (without `SurgicalLoRALinear.bypass`), computing $F_{\text{base}}$ requires a frozen reference copy of the pretrained backbone when `prior_anchor_weight > 0.0`:
1. **Non-Registered Frozen Teacher Reference**:
   - Maintain `_anchor_backbone` via `object.__setattr__(self, "_anchor_backbone", ...)` (extending the existing `_anchor_neck` / `_anchor_hrnet_stage4` pattern in `focal.py`, lines 988–989 and 1192–1201) so it:
     - Follows `.to(device)` and `.to(dtype)` via `FOCAL._apply()`.
     - **Never** appears in `self.parameters()`, `self.state_dict()`, `Trainer` optimizer param groups, or DDP parameter buffers.
     - Always executes in `.eval()` under `with torch.no_grad():`.
2. **VRAM & Precision Guidance**:
   - **HRNet-W32**: ~29M params (~58 MB in `bf16`/`fp16`) — negligible VRAM cost; enable `prior_anchor_weight: 0.1–0.25` by default.
   - **SAM ViT-Large**: ~307M params (~614 MB in `bf16`/`fp16`). Supported on 24GB+ GPUs (e.g., L4/A10/A100); configs must also allow `prior_anchor_weight: 0.0` (falling back cleanly to Mini-Strategies 2, 3, and 4 when VRAM is tight) or optional half-precision casting of `_anchor_backbone`.
3. **Trainer Integration**:
   - `FOCAL.requires_gt_masks` **must** return `True` whenever `prior_anchor_weight > 0.0 and prior_anchor_mode == "pristine_only"`, ensuring `Trainer._step_batch_train()` (`sid_unet/training/trainer.py`, line 1111) automatically forwards `gt_masks`.

---

## 4. Mini-Strategy 2: Layer-Wise Learning Rate Decay (LLRD) + Stem / Positional / BN Lock

Rather than freezing 15+ blocks completely (Strategy B) or updating every layer at a flat LR (Strategy 1), Strategy 1+ unfreezes **all** high-level blocks/stages with exponentially decaying learning rates while locking low-level spatial/positional primitives.

### 4.1 Permanent Spatial & Statistical Locks
Even when `self.net` is unfrozen (`set_backbone_trainable(True)`), the following **must remain permanently frozen** (`requires_grad = False`):
1. **For `FOCALViTNet` (`SAM ViT-L`)**:
   - `image_encoder.patch_embed.*` (low-level 16×16 RGB patch projection)
   - `image_encoder.pos_embed` (`[1, 64, 64, 1024]` pretrained at `1024×1024` — prevents `512×512`-specific positional distortion)
   - `image_encoder.blocks.*.attn.rel_pos_h` and `rel_pos_w` (prevents relative spatial attention geometry drift)
   - Configurable cutoff `freeze_vit_blocks_below: int = 0` (optional hard freeze on earliest blocks `0 .. k-1`, default `0` so all 24 blocks can learn via LLRD, or `4`/`6` for extra protection).
2. **For `HRNet` (`HRNet-W32`)**:
   - Early stem & residual stage: `conv1`, `bn1`, `conv2`, `bn2`, `layer1` (configurable via `hrnet_freeze_stem: bool = True`).
   - **BatchNorm2d Protection (`freeze_backbone_bn: bool = True`)**: Lock all `nn.BatchNorm2d` layers in `self.net` (both frozen stem and optionally unfrozen stages) in `.eval()` mode (`track_running_stats` frozen; affine params frozen or optionally excluded from BN stat drift) via `FOCAL.train()` / `_enforce_frozen_backbone_eval()`, preventing small-batch `512×512` statistics corruption.

### 4.2 Exponential LLRD Parameter Grouping (`FOCAL.get_param_groups`)
Add `model.llrd_enabled: bool = True` (or `training.llrd_decay`), `model.llrd_decay: float = 0.85`, and `model.hrnet_stage_lr_multipliers`:

1. **ViT (`depth = 24`) LLRD Formula**:
   - **`head` (`decoder`, `class_head`, `feature_adapter`)**:
     $$\text{lr}_{\text{head}} = \text{lr}$$
   - **`neck` (`image_encoder.neck`)**:
     $$\text{lr}_{\text{neck}} = \text{lr} \times \text{neck\_lr\_multiplier} \quad (\text{default } 0.5\text{–}1.0)$$
   - **Transformer Block `b` ($b \in \{0, \dots, 23\}$)**:
     $$\text{lr}(b) = \text{lr} \times \text{backbone\_lr\_multiplier} \times \max\!\left(\text{llrd\_min\_scale}, \, \gamma^{(D - 1 - b)}\right)$$
     where $D = 24$, $\gamma = \text{llrd\_decay}$ (default `0.85`), `backbone_lr_multiplier` defaults to `0.5` (Mode 1A/Hybrid) or `1.0` (Mode 1B), and `llrd_min_scale` defaults to `0.02`.
     - *Concrete effect ($\gamma=0.85, \text{mult}=0.5$)*: Block `23` learns at `0.50× lr`, Block `17` at `0.19× lr`, Block `11` at `0.071× lr`, Block `0` at `0.012× lr`.
2. **HRNet Stage-Wise LLRD Formula**:
   - Explicit configurable stage multipliers `hrnet_stage_lr_multipliers` (defaults: `stage4: 1.0`, `transition3: 0.6`, `stage3: 0.4`, `transition2: 0.25`, `stage2: 0.15`, `transition1: 0.10`), multiplied by `backbone_lr_multiplier`.
3. **Trainer & Cosine Scheduler Integration**:
   - Every parameter group emitted by `FOCAL.get_param_groups()` is assigned `lr_scale = group["lr"] / base_lr` in `Trainer._build_optimizer()` (`sid_unet/training/trainer.py`, line 755), which `Trainer._build_scheduler()` (`trainer.py`, lines 847–852) already uses to preserve relative LLRD ratios across cosine annealing!
4. **Safe `set_backbone_trainable()` Interaction**:
   - `_surgical_backbone_param_names` **must** exclude permanently locked ViT (`patch_embed`, `pos_embed`, `rel_pos_h/w`, `blocks < freeze_vit_blocks_below`) and HRNet (`stem`, `layer1`, frozen `BatchNorm2d`) parameters so Epoch 2+ unfreezing after `freeze_backbone_epochs` **never** unlocks them.

---

## 5. Mini-Strategy 3: L2-SP Starting-Point Weight Regularization

### 5.1 Root Defect in Standard AdamW `weight_decay`
Standard AdamW weight decay ($\theta \leftarrow (1 - \eta \lambda)\theta - \dots$) pulls every pretrained parameter toward **`0.0`**. Over multi-epoch finetuning on a target dataset, weights in `self.net` drift away from their pretrained values toward zero, degrading out-of-distribution priors.

### 5.2 L2-SP Specification (`l2_sp_weight`)
Add `model.l2_sp_weight: float = 1e-4` (recommended range `1e-4` to `1e-3`):

1. **Initial Pretrained Weight Snapshot ($\theta^{(0)}$)**:
   - Immediately after `_load_pretrained()` succeeds (and **only** when loading base pretrained weights, or from the original base checkpoint if resuming — see §5.3), snapshot detached copies of every trainable pretrained weight in `self.net`:
     $$\theta_k^{(0)} = p_k.\text{detach}().\text{clone}()$$
   - Store snapshots in a non-persistent container (`self._l2_sp_init_params`, e.g., via `object.__setattr__` / non-registered dict moved lazily on `.to(device)` or compared on matching device/dtype) so `state_dict()` and `DDP` parameter counts are unaffected.
   - **Target Filter**: Apply L2-SP to trainable multi-dimensional weights (`p.ndim >= 2`, or all trainable `self.net` params except optional norms/biases) so scalar/bias shifts aren't over-penalized.
2. **Scale-Invariant Penalty Formulation**:
   To ensure a single `l2_sp_weight` behaves consistently across 29M-param `HRNet` and 307M-param `ViT-L`:
   $$\mathcal{L}_{\text{L2-SP}} = \lambda_{\text{sp}} \cdot \frac{\sum_{k \in \mathcal{P}_{\text{net}}} \|p_k - \theta_k^{(0)}\|_2^2}{\sum_{k \in \mathcal{P}_{\text{net}}} \text{numel}(p_k)}$$
   *(Mean squared parameter deviation per element, rather than raw unnormalized sum, so `l2_sp_weight` does not explode by `10×` between HRNet and ViT-L.)*
3. **Decouple AdamW Weight Decay on Pretrained Backbone Groups**:
   - When `l2_sp_weight > 0.0`, `FOCAL.get_param_groups()` **must** set `"weight_decay": 0.0` on all pretrained `self.net` LLRD groups (`neck`, `block_*`, `stage*`), while retaining normal `weight_decay` (`0.0001`) on newly initialized modules (`decoder`, `class_head`, `feature_adapter`).
   - This ensures pretrained weights are regularized strictly toward $\theta^{(0)}$ and **never** toward $0$.
4. **Loss & Metric Reporting**:
   - Include $\mathcal{L}_{\text{L2-SP}}$ in `prior_anchor_penalty` (or return `l2_sp_loss` alongside `prior_anchor_loss` in `FOCAL.forward()` / `SIDTotalLoss` / `Trainer` epoch logs) during `self.training == True`.

### 5.3 Resume Behavior for L2-SP
- When resuming from a Strategy 1+ training checkpoint (`resume_from_checkpoint`), $\theta^{(0)}$ **should ideally still reference the original upstream pretrained weights** (`KhangTruong/FOCAL`), or—if unavailable—fall back to the resumed checkpoint weights with an explicit warning logged. Storing/referencing the upstream `hf_repo` base checkpoint for $\theta^{(0)}$ prevents L2-SP anchor drift across multi-stage resumes.

---

## 6. Mini-Strategy 4: Hybrid Joint Supervised + Contrastive Mode (`finetune_mode: "hybrid"`)

### 6.1 Why Mode 1A Alone Drifts Hypersphere Geometry
In Strategy 1 Mode 1A (`finetune_mode: "supervised"`), `FOCAL.forward()` routes `feats` exclusively to `FOCALSupervisedHead` (`BCE + Dice + Edge`), setting `align_loss = None`. Because `FOCALSupervisedHead` is a nonlinear multi-layer conv decoder, it can learn to solve `IMD2020` masks even while `feats` loses its L2-normalized angular separation between tampered and pristine regions—destroying FOCAL's core contrastive prior.

### 6.2 Hybrid Mode Specification (`finetune_mode: "hybrid"`)
Support `finetune_mode: "hybrid"` (aliases: `"supervised_contrastive"`, `"1a_plus"`, `"strategy1_plus"`):

1. **Dual Supervision During Training (`self.training == True`)**:
   - **Supervised Spatial Branch**: Execute `self._run_supervised_head(feats, orig_h, orig_w)` (`FOCALSupervisedHead`) to produce `mask_logits` and `pred_mask = torch.sigmoid(mask_logits)`.
   - **Contrastive Hypersphere Branch**: Compute `align_loss = self.compute_contrastive_loss(feats, gt_masks)` (`MyInfoNCE` per image on `F.normalize(feats, p=2, dim=1)`).
   - **Critical Loss Flag**: Set `res["skip_mask_loss"] = False` (unlike Mode 1B `"contrastive"`, which sets `skip_mask_loss = True`). Thus `SIDTotalLoss` optimizes:
     $$\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}} + \lambda_{\text{edge}} \mathcal{L}_{\text{edge}}}_{\text{FOCALSupervisedHead}} + \lambda_{\text{align}} \mathcal{L}_{\text{MyInfoNCE}} + \mathcal{L}_{\text{pristine-anchor}} + \mathcal{L}_{\text{L2-SP}}$$
   - Recommended `align_weight`: `0.25`–`0.5` (strong enough to maintain angular cluster separation on `feats` without overpowering pixel mask gradients).
2. **`requires_gt_masks` & `Trainer` Compatibility**:
   - `FOCAL.requires_gt_masks` **must** return `True` when `finetune_mode == "hybrid"`, so `Trainer._step_batch_train()` passes `gt_masks` automatically.
   - `self.fc` remains frozen (`requires_grad=False`) when `head_type == "decoder"` so DDP does not encounter unused parameters.
3. **Evaluation Head Parity & Optional Clustering Cross-Check**:
   - By default (`eval_use_clustering: false` / `use_clustering: false`), `eval()` runs `FOCALSupervisedHead` to match the trained mask predictor (`train`/`eval` parity).
   - Optionally expose `eval_clustering_diagnostic` (or configurable `use_clustering: true` during standalone cross-eval) so practitioners can verify that `feats` still produces high-IoU unsupervised k-means clustering masks (`_cluster_features_with_stats`) even after supervised finetuning!

---

## 7. Recommended Strategy 1+ Configurations (`512×512`)

New experiment configs should live alongside Strategy 1 configs in `configs/experiments/focal/` (preserving existing Strategy 1 & Strategy B configs intact):

### 7.1 `configs/experiments/focal/focal_hrnet_strategy1_plus.yaml` (Hybrid `1A+`)
- `data`: `image_size: [512, 512]`, `normalize: false`, `batch_size: 32` (empirically verified on NVIDIA L4 22.5 GiB: 73.3% peak VRAM, 16.5 GiB, 23.9 img/s; `batch_size: 64` OOMs without activation checkpointing)
- `model`:
  - `backbone: "hrnet"`, `finetune_mode: "hybrid"`, `head_type: "decoder"`, `use_clustering: false`
  - `hrnet_freeze_stem: true`, `freeze_backbone_bn: true`
  - `llrd_enabled: true`, `hrnet_stage_lr_multipliers: {stage4: 1.0, transition3: 0.6, stage3: 0.4, transition2: 0.25, stage2: 0.15, transition1: 0.10}`
  - `prior_anchor_weight: 0.15`, `prior_anchor_mode: "pristine_only"`, `pristine_anchor_threshold: 0.05`
  - `l2_sp_weight: 0.0005`
- `loss`: `bce_weight: 0.5`, `dice_weight: 0.5`, `edge_weight: 0.5`, `align_weight: 0.3`
- `training`: `learning_rate: 0.0002`, `backbone_lr_multiplier: 0.5`, `freeze_backbone_epochs: 1`, `epochs: 10`

### 7.2 `configs/experiments/focal/focal_vit_strategy1_plus.yaml` (Hybrid `1A+`)
- `data`: `image_size: [512, 512]`, `normalize: false`, `batch_size: 32`, `gradient_accumulation_steps: 1` (empirically verified on NVIDIA L4 22.5 GiB with `gradient_checkpointing: true`: 80.9% peak VRAM, 18.3 GiB)
- `model`:
  - `backbone: "vit"`, `finetune_mode: "hybrid"`, `head_type: "decoder"`, `use_clustering: false`, `force_vit_1024: false`
  - `freeze_vit_stem_and_pos: true` (`patch_embed`, `pos_embed`, `rel_pos_h/w` permanently frozen)
  - `llrd_enabled: true`, `llrd_decay: 0.85`, `llrd_min_scale: 0.02`, `neck_lr_multiplier: 0.5`
  - `prior_anchor_weight: 0.1`, `prior_anchor_mode: "pristine_only"`, `anchor_half_precision: true`, `gradient_checkpointing: true`
  - `l2_sp_weight: 0.0005`
- `loss`: `bce_weight: 0.5`, `dice_weight: 0.5`, `edge_weight: 0.5`, `align_weight: 0.3`
- `training`: `learning_rate: 0.0001`, `backbone_lr_multiplier: 0.5`, `freeze_backbone_epochs: 1`, `epochs: 10`

### 7.3 `configs/experiments/focal/focal_vit_contrastive_strategy1_plus.yaml` & `focal_hrnet_contrastive_strategy1_plus.yaml` (Contrastive `1B+`)
- Same LLRD + Stem/Pos/BN lock + L2-SP + **Pristine-Only** Feature Anchor (`prior_anchor_mode: "pristine_only"`, `prior_anchor_weight: 0.15`), with `finetune_mode: "contrastive"`, `use_clustering: true`, `align_weight: 1.0`, `learning_rate: 0.00005`, `backbone_lr_multiplier: 0.5`, `freeze_backbone_epochs: 0`.
- L4-verified batch sizes: `batch_size: 64` for HRNet Contrastive `1B+` (93.0% peak VRAM, 21.0 GiB) and `batch_size: 32` (`gradient_checkpointing: true`, `gradient_accumulation_steps: 1`) for ViT Contrastive `1B+` (81.1% peak VRAM, 18.3 GiB).

---

## 8. Observability, DDP, AMP/FP8 & Checkpoint Requirements

1. **Diagnostic Logging**:
   - Startup summary must log active Strategy 1+ settings: `finetune_mode`, `llrd_enabled`/`llrd_decay`, per-stage/block LR min–max range, locked modules (`patch_embed`, `pos_embed`, `rel_pos`, frozen BN count), `prior_anchor_mode`, `prior_anchor_weight`, and `l2_sp_weight`.
   - `SIDTotalLoss` and `Trainer` epoch logs should report `align_loss` (`MyInfoNCE`), `prior_anchor_loss` (pristine-only cosine drift), and `l2_sp_loss` independently for clear tuning visibility.
2. **DDP & AMP/FP8 Safety**:
   - `_anchor_backbone` and `_l2_sp_init_params` must stay outside `nn.Module` parameter registration so DDP `find_unused_parameters=False` remains valid and `prepare_model_for_fp8_training()` (`sid_unet/utils/precision.py`) does not conflict with optimizer state.
   - All loss terms (`prior_anchor_loss`, `l2_sp_loss`, `compute_contrastive_loss`) must accumulate in `float32` to prevent `float16`/`bfloat16`/`fp8` underflow.
3. **Backward Compatibility**:
   - Existing Strategy 1 (`tests/test_focal_strategy1.py`), Strategy B (`tests/test_focal_strategy_b.py`), and audit tests (`tests/test_focal_pim_mismatch_audit.py`) **must** remain green.

---

## 9. Acceptance Criteria (Verification Checklist for Implementation PR)

1. **Pristine-Only Anchoring**:
   - Unit test verifies that changing features on tampered pixels ($M=1$) produces **zero** `prior_anchor_loss`, while changing features on pristine pixels ($M=0$) produces positive `prior_anchor_loss`—and all-tampered batches ($M \equiv 1$) return `0.0` with finite gradients (zero `NaN`/`Inf`).
2. **LLRD & Permanent Locks**:
   - `FOCAL.get_param_groups()` produces monotonically decaying LRs from `neck`/`block_23` down to `block_0` (ViT) and `stage4` down to `stage2` (HRNet).
   - After `set_backbone_trainable(False)` → `set_backbone_trainable(True)` (Epoch 1 → Epoch 2), `patch_embed`, `pos_embed`, `rel_pos_h/w`, HRNet stem (`conv1..layer1`), and frozen `BatchNorm2d` running stats remain bit-identical (`torch.equal`) before and after optimizer steps.
3. **L2-SP Regularization**:
   - Pretrained `self.net` param groups have `weight_decay == 0.0` when `l2_sp_weight > 0.0`; `l2_sp_loss` is `0.0` at step 0, positive after a perturbed step, and scale-normalized by total regularizable parameter elements.
4. **Hybrid Mode (`1A+`) Parity & Dual Supervision**:
   - In `finetune_mode="hybrid"`, `forward()` outputs both supervised `mask_logits` (`skip_mask_loss == False`) and `align_loss` (`MyInfoNCE`), and `SIDTotalLoss` backpropagates gradients into both `self.decoder` and `self.net`.
   - `512×512` forward/backward + optimizer step passes cleanly on both `FOCALHRNet` and `FOCALViT` under AMP (`bfloat16`/`float16`) and FP32.

---

## 10. Out of Scope
- Post-hoc **WiSE-FT** weight-space interpolation ($\theta_{\text{merged}} = (1 - \alpha)\theta_0 + \alpha\theta_{\text{finetuned}}$) is excluded from this specification.
- Multi-backbone fusion (`HRNet + ViT`), soft-EM clustering, and `ForensicVectorMemory` remain out of scope.
