# Requirements Specification: FOCAL Strategy B — Prior-Preserving Parameter-Efficient Model Surgery at `512×512`

**Author**: Antigravity Assistant  
**Date**: November 16, 2026  
**Target Repository**: `khangtruong-ui/ai-detection-unet-gpu`  
**Status**: Proposed Requirements Specification (No Implementation Code)  
**Predecessor**: [`docs/focal_finetuning_strategy1_requirements.md`](focal_finetuning_strategy1_requirements.md) (PR #24, commit `fac1b40`)  
**Scope**: `sid_unet/models/focal.py`, `sid_unet/models/unet.py`, `sid_unet/training/trainer.py`, `sid_unet/losses/auxiliary.py`, `sid_unet/utils/precision.py`, `configs/experiments/focal/*.yaml`

---

## 1. Executive Summary & Problem Diagnosis (Why Strategy 1 Drifted Priors)

Following the implementation of Strategy 1 (PR #24, `73e4a11`), full-backbone finetuning on `512×512` images exhibited severe **prior drift** (catastrophic forgetting of pretrained SAM and FOCAL forensic representations). Auditing the merged implementation reveals three concrete structural causes:

1. **Unbounded Full-Rank Backbone Updates**:
   - In **Mode 1B** (`configs/experiments/focal/focal_vit_contrastive.yaml`), all ~307M parameters of `FOCALViTNet` (`ImageEncoderViT`: `patch_embed`, `pos_embed`, all 24 `SAMBlock` layers including `rel_pos_h`/`rel_pos_w`, and `neck`) are trainable from step 0. Even with `learning_rate: 0.00002`, updating 307M weights across narrow/domain-specific datasets (e.g., `KhangTruong/IMD2020`) rapidly overwrites pretrained attention routing and positional priors.
   - In **Mode 1A** (`configs/experiments/focal/focal_vit.yaml`, `focal_hrnet.yaml`), `freeze_backbone_epochs: 1` freezes `self.net` only for Epoch 1. Starting Epoch 2, `Trainer.train_epoch()` (`sid_unet/training/trainer.py`, lines 1191–1205) calls `FOCAL.set_backbone_trainable(True)` (`sid_unet/models/focal.py`, lines 902–909), which **blindly unfreezes 100% of `self.net`** (`for p in self.net.parameters(): p.requires_grad_(True)`), exposing early patch/conv stems, `pos_embed`, and `BatchNorm2d` statistics to full-rank gradient drift.
2. **Resolution & Positional-Embedding Drift at `512×512`**:
   - Pretrained `FOCALViTNet` stores `pos_embed` at `[1, 64, 64, 1024]` (`1024×1024`) and `rel_pos_h`/`rel_pos_w` at `[127, 64]` (global blocks) and `[27, 64]` (windowed blocks). When unfreezing the full ViT at `512×512` (`32×32` patch grid), gradients reshape `pos_embed` and `rel_pos` specifically for `512×512` interpolation, degrading multi-scale/cross-eval consistency.
3. **Lack of Prior-Anchoring & Adapter Decomposition**:
   - Without low-rank/weight-decomposed constraints (LoRA/DoRA) or explicit feature-space prior regularization, both Mode 1A and Mode 1B can arbitrarily rotate FOCAL's L2-normalized forensic hypersphere, degrading out-of-distribution generalization.

**Strategy B Objective**: Replace full-backbone unfreezing with **Prior-Preserving Parameter-Efficient Model Surgery** tuned for **`512×512`** resolution across **both Mode 1A (Supervised)** and **Mode 1B (Contrastive)**, locking pretrained base weights while adapting only low-rank subspaces, the forensic projection `neck` (or late HRNet stage/adapter), and optional prior-anchored regularizers.

---

## 2. Target `512×512` Spatial Geometry & Preprocessing Specification

All Strategy B configurations **must** standardize on `512×512` raw RGB inputs:

### 2.1 Input Pipeline Invariants
- `data.image_size: [512, 512]`
- `data.normalize: false` (raw `[0.0, 1.0]` RGB tensors, with `FOCAL._ensure_raw_rgb()` retained solely as a defensive guard for external callers).
- **Mandatory ViT Resolution Setting**: `model.force_vit_1024: false`. Strategy B **must** run `FOCALViTNet` natively on `[B, 3, 512, 512]` without `512→1024` internal upsampling.

### 2.2 `FOCALViTNet` (`SAM ViT-Large`) Geometry at `512×512`
- **`PatchEmbed` (`kernel_size=16, stride=16`)**: Produces a `32 × 32` spatial token grid (`N = 1,024` tokens, `embed_dim = 1024`).
- **`pos_embed`**: Pretrained `[1, 64, 64, 1024]` tensor bicubically interpolated on the fly to `[1, 32, 32, 1024]` (`sid_unet/models/focal.py`, lines 596–600). **Requirement**: `patch_embed` and `pos_embed` **must remain permanently frozen** (`requires_grad = False`).
- **`SAMBlock` Attention Windows**:
  - **Local windowed blocks** (`depth=24`, excluding `(5, 11, 17, 23)`): `window_size=14` pads `32×32 → 42×42` (`3 × 3 = 9` windows per image) via `window_partition_sam` / `window_unpartition_sam` (`focal.py`, lines 376–398).
  - **Global attention blocks** (`(5, 11, 17, 23)`, `window_size=0`): Attends over all `32×32` tokens with `SAMAttention._get_rel_pos` linearly interpolating `rel_pos_h/w` (`[127, 64] → [63, 64]`). **Requirement**: `rel_pos_h` and `rel_pos_w` **must remain permanently frozen** (`requires_grad = False`).
- **`neck` Output**: `[B, 256, 32, 32]` (`H/16, W/16`).
  - In **Mode 1A** (`FOCALSupervisedHead`), `num_upsample_stages = 4` progressively upsamples `32×32 → 64 → 128 → 256 → 512×512`.
  - In **Mode 1B** (`contrastive`), `32×32 = 1,024` spatial tokens fit entirely within `contrastive_sample_size: 1024` (requiring zero lossy subsampling at `512×512`).

### 2.3 `HRNet-W32` Geometry at `512×512`
- Stem (`conv1` stride-2, `conv2` stride-2) + Stages 1–4 produce `[B, 32, 128, 128]` (`H/4, W/4`, `16,384` spatial tokens).
- In **Mode 1A**, `FOCALSupervisedHead` (`num_upsample_stages = 2`) upsamples `128×128 → 256 → 512×512`.
- In **Mode 1B**, `contrastive_sample_size` (recommended `1024` or `2048` at `512×512`) bounds stratified token sampling in `compute_contrastive_loss()` to $O(K^2)$.

---

## 3. Requirement 1: `FOCALViTNet` PEFT Surgery (LoRA / DoRA + Selective Neck)

To eliminate full-rank drift in the 307M-parameter `ImageEncoderViT` while preserving its semantic and forensic representation geometry, Strategy B **must** apply Parameter-Efficient Fine-Tuning (PEFT) via the repository's existing `SurgicalLoRALinear` (`sid_unet/models/mask_clip.py`, lines 1088–1133).

### 3.1 Reuse of `SurgicalLoRALinear` (LoRA & DoRA)
- `FOCAL` **must** import and use `SurgicalLoRALinear` (supporting both standard LoRA when `use_dora: false` and Weight-Decomposed Low-Rank Adaptation when `use_dora: true`, per `configs/experiments/mask_clip/mask_clip_dora_surgery.yaml`).
- **Why DoRA (`use_dora: true` as Strategy B default)**: DoRA decomposes weight updates into magnitude ($m$) and direction ($W_0 + \Delta W$), allowing subtle forensic boundary adaptations without rotating base attention directions uncontrollably.
- **Base Weight Freezing**: Inside every `SurgicalLoRALinear`, `base_linear.weight` and `base_linear.bias` **must** remain frozen (`requires_grad = False`).

### 3.2 Block Scope & Target Linear Modules
Strategy B **must** expose explicit configuration controls under `model:`:
1. **`peft_enabled` / `use_peft`** (`bool`, `true` for Strategy B ViT configs)
2. **`lora_r`** (`int`, default `8`; recommended range `4–16`)
3. **`lora_alpha`** (`float`, default `16.0`; recommended `2 * lora_r`)
4. **`lora_dropout`** (`float`, default `0.05`)
5. **`use_dora`** (`bool`, default `true`)
6. **`lora_block_scope`** (`str` or `List[int]`, default `"late_and_global"`):
   - `"late_and_global"`: Adapts global attention blocks `[5, 11, 17, 23]` plus the final local+global stage `[18, 19, 20, 21, 22, 23]` (i.e., `{5, 11, 17, 18, 19, 20, 21, 22, 23}`), keeping early blocks `0..4, 6..10, 12..16` completely frozen to preserve low-level SAM/forensic feature extractors.
   - `"last_k"` (via `lora_last_k_blocks`, e.g., last `6` or `8` blocks: `18..23` or `16..23`).
   - `"global_only"`: Only global attention blocks `[5, 11, 17, 23]`.
   - `"all"`: All 24 blocks `0..23`.
   - Explicit list of block indices (e.g., `[11, 17, 20, 21, 22, 23]`).
7. **`lora_target_modules`** (`List[str]`, default `["qkv", "proj"]`):
   - Targets `SAMAttention.qkv` (`1024 → 3072`) and `SAMAttention.proj` (`1024 → 1024`) within the selected `SAMBlock`s.
   - Optionally includes `["lin1", "lin2"]` (`MLPBlock` FFN layers: `1024 ↔ 4096`) when higher domain-shift capacity is explicitly requested.

### 3.3 Selective `neck` & Normalization Policy
1. **`train_neck`** (`bool`, default `true`):
   - `self.net.image_encoder.neck` (`Conv2d(1024→256, 1×1) → LayerNorm2d → Conv2d(256→256, 3×3) → LayerNorm2d`, ~1.77M params) is the primary projection from SAM feature space into FOCAL's contrastive forensic space.
   - When `train_neck: true`, `neck` parameters are trainable with a configurable learning-rate multiplier (`neck_lr_multiplier`, default `0.5` or `0.25`) so the projection adapts smoothly without destabilizing upstream blocks.
   - When `train_neck: false`, only LoRA/DoRA adapters (and any Mode 1A decoder) are trainable—enforcing maximum prior preservation.
2. **`train_vit_norms`** (`bool`, default `false`):
   - `SAMBlock.norm1` and `SAMBlock.norm2` (`LayerNorm`) **must** default to frozen (`requires_grad = False`) to prevent activation scale drift across frozen vs. adapted blocks.

### 3.4 Expected Parameter Footprint (`FOCALViTNet`)
- Full `FOCALViTNet`: ~307M trainable parameters (Strategy 1).
- Strategy B (`lora_r=8`, `qkv`+`proj`, `late_and_global` = 9 blocks, `train_neck=true`):
  - LoRA/DoRA adapters: `9 blocks × (qkv + proj)` ≈ **~0.6M–0.7M** trainable parameters.
  - `neck`: **~0.85M** trainable parameters (`1024×256` + `256×256×9` + norms).
  - Total backbone trainable params: **< 0.6% of `FOCALViTNet`** (plus `FOCALSupervisedHead` in Mode 1A), cutting optimizer state memory drastically and eliminating full-rank prior drift.

---

## 4. Requirement 2: `HRNet` Prior-Preserving Surgery (`512×512`)

While `FOCALViTNet` is the primary beneficiary of Strategy B LoRA/DoRA surgery, `FOCALHRNet` **must** also enforce prior-preserving constraints at `512×512` rather than unfreezing all 29M convolutional parameters and `BatchNorm2d` statistics.

### 4.1 Stage-Wise Freezing (`hrnet_trainable_stages`)
- Expose `model.hrnet_trainable_stages` (default `["stage4"]` or `["none"]` when using an adapter-only setup):
  - **Frozen by default**: `conv1`, `bn1`, `conv2`, `bn2`, `layer1`, `transition1`, `stage2`, `transition2`, `stage3`, and `transition3` (optional).
  - **Selectively adaptable**: Only `stage4` (or `stage4[-1]` final `HighResolutionModule`) and/or an attached residual forensic adapter are trainable.

### 4.2 BatchNorm Running Statistics Protection (`freeze_backbone_bn`)
- Expose `model.freeze_backbone_bn` (`bool`, default `true` in Strategy B):
  - Any frozen `HRNet` stage **must** have its `nn.BatchNorm2d` layers kept in `.eval()` mode (`track_running_stats` not updated) and affine parameters (`weight`, `bias`) frozen (`requires_grad = False`) even during `model.train()` and `Trainer.train_epoch()`.
  - Optionally allow freezing `BatchNorm2d` running stats across `stage4` as well when `batch_size <= 8` at `512×512`, avoiding batch-stat jitter.

### 4.3 Optional Zero-Initialized Residual Forensic Adapter (`use_feature_adapter`)
- Support an optional lightweight residual adapter (`FeatureAdapter` / `1×1 Conv → GroupNorm → GELU → 1×1 Conv`) applied to `feats` (`[B, C, H', W']`) before clustering/head projection:
  - The final convolution (or residual gate scalar $\alpha$) **must** be zero-initialized so that at training step 0:
    $$F_{\text{adapted}} = F_{\text{base}} + \text{Adapter}(F_{\text{base}}) \equiv F_{\text{base}}$$
  - This allows `self.net` (HRNet or ViT) to remain 100% frozen when ultra-strict prior preservation is desired, adapting solely via a zero-initialized residual offset in Mode 1A or Mode 1B.

---

## 5. Requirement 3: Fixing the `set_backbone_trainable()` & `Trainer` Freeze Hazard

### 5.1 Critical Interaction Bug in Current Codebase
In the current Strategy 1 code:
- `Trainer.train_epoch()` (`sid_unet/training/trainer.py`, lines 1191–1205) invokes `raw_target.set_backbone_trainable(backbone_trainable)` whenever `freeze_backbone_epochs > 0`.
- `FOCAL.set_backbone_trainable(True)` (`sid_unet/models/focal.py`, lines 902–909) unconditionally loops over **every parameter** of `self.net` and sets `p.requires_grad_(True)` and `self.net.train(self.training)`.
- **Failure if left unchanged**: Combining `freeze_backbone_epochs: 1` with Strategy B LoRA/DoRA or partial HRNet freezing would **silently destroy the surgical freeze mask at Epoch 2**, unfreezing all 307M base parameters and `BatchNorm2d` layers!

### 5.2 Required Behavior
1. **Surgical Trainable Mask Tracking**:
   - `FOCAL` **must** maintain the authoritative set/mask of parameters intended to be trainable under the active surgery (e.g., `_surgical_trainable_param_names` / module-level rules) computed right after surgery injection.
   - `set_backbone_trainable(False)` freezes **all** backbone parameters (including LoRA/DoRA adapters, `neck`, and trainable HRNet stages) during head warmup (`epoch <= freeze_backbone_epochs`), leaving only `self.decoder` / `self.class_head` / `self.adapter` trainable.
   - `set_backbone_trainable(True)` unfreezes **only** the permitted surgical backbone parameters (LoRA/DoRA parameters, `neck` if `train_neck=True`, permitted HRNet stages if `hrnet_trainable_stages` includes them), and **never** unfreezes frozen base weights (`base_linear`, `patch_embed`, `pos_embed`, `rel_pos_h/w`, frozen blocks/stages) or frozen `BatchNorm2d` layers.
2. **Train-Mode BN Invariant**:
   - Override or hook `FOCAL.train(mode=True)` so that any permanently frozen `BatchNorm2d` submodules (and frozen `self.net` when in adapter-only mode) remain in `.eval()` whenever `model.train()` is called by `Trainer.train_epoch()`.

---

## 6. Requirement 4: Prior-Anchoring Regularization & Optimization Policy

To explicitly bound representation drift relative to the pretrained FOCAL hypersphere—especially in **Mode 1B (`contrastive`)** and **Mode 1A (`supervised`)** at `512×512`:

### 6.1 Optional Feature-Space Prior Anchoring (`prior_anchor_weight`)
1. **Definition**:
   - Add `model.prior_anchor_weight` (or `loss.prior_anchor_weight`, default `0.0` for Mode 1A, recommended `0.05–0.2` for Mode 1B).
   - When `prior_anchor_weight > 0.0` during training, obtain the frozen base representation $F_{\text{base}}$ (for PEFT/adapter models, computed either via a `torch.no_grad()` pass with LoRA/adapter scaling temporarily disabled or a lightweight feature-anchor hook—without duplicating the 307M model in VRAM!) and compute the L2-normalized cosine drift penalty:
     $$\mathcal{L}_{\text{anchor}} = \frac{1}{B \cdot H' \cdot W'} \sum_{b, h, w} \left(1 - \left\langle \hat{F}_{\text{adapted}}[b, :, h, w], \, \text{sg}\!\left(\hat{F}_{\text{base}}[b, :, h, w]\right) \right\rangle\right)$$
     where $\hat{F} = \text{normalize}(F, p=2, \text{dim}=1)$ and $\text{sg}(\cdot)$ is stop-gradient.
   - **Memory Constraint**: Because `SurgicalLoRALinear` stores the unmodified `base_linear` weights in-place, the base forward pass **must not** instantiate a second copy of `FOCALViTNet` or `HRNet`; it must toggle an adapter-bypass context (`lora_bypass` / `scaling = 0` + frozen `neck` or snapshot) or run under `torch.no_grad()` when `neck` is frozen, keeping VRAM footprint minimal.
   - **Alternative / Complementary Weight-Space Anchor**: When `train_neck: true`, optionally support L2 anchoring of `neck` weights toward their pretrained initialization $\theta_{\text{neck}}^{(0)}$ (`neck_l2_anchor_weight`), costing zero extra forward pass.
2. **Loss Integration**:
   - Combine $\lambda_{\text{anchor}} \mathcal{L}_{\text{anchor}}$ cleanly into `res["align_loss"]` (or expose `prior_anchor_loss` in `res` / `SIDTotalLoss` metrics) so `Trainer` logs and optimizes it without breaking `SIDTotalLoss`.

### 6.2 Optimizer Parameter Groups & Hyperparameters (`512×512`)
1. **`FOCAL.get_param_groups()` Extension**:
   - Separate trainable parameters into distinct LR & weight-decay groups compatible with `Trainer._build_optimizer()` (`sid_unet/training/trainer.py`, lines 719–790) and `Trainer._build_scheduler()` (`lr_scale` support at lines 847–852):
     - **`head`**: `decoder`, `class_head`, `feature_adapter` → `lr_scale = 1.0`
     - **`lora`**: `lora_A`, `lora_B`, `dora_m` → `lr_scale = lora_lr_multiplier` (default `1.0`), reduced weight decay (`0.0` or `0.01` to avoid shrinking DoRA/LoRA updates to zero)
     - **`neck` / `backbone_partial`**: trainable `neck` or `stage4` → `lr_scale = neck_lr_multiplier` / `backbone_lr_multiplier` (default `0.25`–`0.5`)
2. **Recommended `512×512` Strategy B Config Defaults**:
   - **Mode 1A (`focal_vit_peft.yaml` / updated `focal_vit.yaml`)**:
     - `image_size: [512, 512]`, `normalize: false`, `force_vit_1024: false`
     - `finetune_mode: "supervised"`, `use_clustering: false`, `head_type: "decoder"`
     - `peft_enabled: true`, `lora_r: 8`, `lora_alpha: 16.0`, `lora_dropout: 0.05`, `use_dora: true`, `lora_block_scope: "late_and_global"`, `train_neck: true`
     - `learning_rate: 1e-4`, `neck_lr_multiplier: 0.25`, `freeze_backbone_epochs: 1`, `batch_size: 4`–`8`, `gradient_accumulation_steps: 4`
   - **Mode 1B (`focal_vit_contrastive_peft.yaml` / updated `focal_vit_contrastive.yaml`)**:
     - `image_size: [512, 512]`, `normalize: false`, `force_vit_1024: false`
     - `finetune_mode: "contrastive"`, `use_clustering: true`
     - `peft_enabled: true`, `lora_r: 8`, `lora_alpha: 16.0`, `lora_dropout: 0.05`, `use_dora: true`, `lora_block_scope: "late_and_global"`, `train_neck: true`
     - `learning_rate: 5e-5` (higher than full-tuning `2e-5` because only low-rank adapters + `neck` move), `neck_lr_multiplier: 0.25`, `align_weight: 1.0`, `prior_anchor_weight: 0.1`
     - `batch_size: 8`–`16` (realistic for `512×512` ViT-L; `auto_batch_size` / gradient accumulation preserves effective batch size)

---

## 7. Requirement 5: Checkpointing, Weight Merging, DDP & FP8 Compatibility

### 7.1 Strict Initialization Order
`FOCAL.__init__()` **must** execute in this exact order:
1. Build clean `HRNet` or `FOCALViTNet` and supervised `decoder` / `class_head` (if enabled).
2. Run `_load_pretrained()` with strict verification against upstream `KhangTruong/FOCAL` weights (`FOCAL_ViT_weights.pth` / `FOCAL_HRNet_weights.pth`). Zero-fallback policy (`pretrained=False` raises `ValueError`; missing weights raise `RuntimeError`) remains strictly enforced.
3. Apply Strategy B surgery (`inject_peft_lora()`, HRNet stage freezing, BN eval-locking, or residual adapter attachment) **after** base weights are loaded and verified.

### 7.2 Checkpoint Save/Reload & Weight Merging
1. **Resuming Strategy B Checkpoints**:
   - Checkpoints saved by `CheckpointManager` contain `SurgicalLoRALinear` keys (`base_linear.*`, `lora_A`, `lora_B`, `dora_m`) and/or `decoder.*` / `adapter.*` keys.
   - `FOCAL` (or `build_model` / `resume_from_checkpoint`) **must** cleanly reload Strategy B checkpoints without failing key checks in `_load_pretrained()` when `checkpoint_path` points to a Strategy B checkpoint (auto-detecting PEFT keys or applying PEFT wrapping before loading a PEFT-formatted `state_dict`).
2. **Lossless Weight Merging (`merge_peft_weights`)**:
   - Provide a `merge_peft_weights()` method on `FOCAL` (handling both standard LoRA $W' = W_0 + \frac{\alpha}{r} B A$ and DoRA $W' = m \odot \frac{W_0 + \frac{\alpha}{r} B A}{\|W_0 + \frac{\alpha}{r} B A\|_2}$) that collapses `SurgicalLoRALinear` back into plain `nn.Linear` layers for zero-latency deployment and standard `sid-cross-eval` compatibility.

### 7.3 DDP & Precision (`AMP` / `FP8`) Safety
1. **DDP (`DistributedDataParallel`)**:
   - Every parameter with `requires_grad=True` **must** participate in `forward()` and receive gradients during training so `find_unused_parameters=False` works cleanly.
   - Specifically, unused `self.fc` (`requires_grad=False`) and frozen base layers (`requires_grad=False`) must remain excluded from DDP gradient buckets.
2. **FP8 / TorchAO Safety (`sid_unet/utils/precision.py`)**:
   - **Critical Hazard**: In `SurgicalLoRALinear.forward()` (`sid_unet/models/mask_clip.py`, lines 1124–1129), DoRA directly computes `combined_w = self.base_linear.weight + delta_w`. If `prepare_model_for_fp8_training()` (`sid_unet/utils/precision.py`, lines 324–360) converts `base_linear` inside `SurgicalLoRALinear` into a TorchAO `Float8Linear`, accessing `self.base_linear.weight + delta_w` will fail or trigger invalid tensor-subclass arithmetic!
   - **Requirement**: `fp8_linear_module_filter()` in `sid_unet/utils/precision.py` (or `SurgicalLoRALinear` FP8 guard) **must** exclude `base_linear` children of `SurgicalLoRALinear` from FP8 `Float8Linear` conversion (or perform DoRA/LoRA adaptation safely in `bfloat16`/`float16` AMP), while allowing non-PEFT eligible linear layers to use FP8/AMP cleanly.

---

## 8. Requirement 6: Observability & Diagnostic Metrics

During Strategy B training, `FOCAL` / `Trainer` **must** log:
1. **Startup Surgery Summary**: Total parameters, trainable parameters, trainable percentage (`< 1%`–`2%` for ViT Strategy B), adapted block indices, `use_dora`, `train_neck`, and frozen BN status.
2. **Per-Step / Epoch Drift Diagnostics**:
   - `align_loss` / `contrastive_loss` (Mode 1B)
   - `prior_anchor_loss` (when `prior_anchor_weight > 0`)
   - Mean `cluster_separability` ($1 - \langle c_0, c_1 \rangle$) during clustering eval, verifying that learned adaptations separate forged vs. pristine regions *without* collapsing pretrained clustering margins.

---

## 9. Acceptance Criteria (Verification Checklist for Implementation PR)

An implementation PR fulfilling Strategy B will be considered complete when all of the following are verified:

1. **Parameter Freezing & No Full-Backbone Leakage**:
   - Instantiating `FOCAL(backbone="vit", peft_enabled=True, ...)` freezes 100% of `patch_embed`, `pos_embed`, `rel_pos_h/w`, `base_linear` weights, and unadapted blocks; trainable backbone params are strictly limited to injected LoRA/DoRA matrices (and `neck` when `train_neck=True`).
   - Calling `set_backbone_trainable(False)` then `set_backbone_trainable(True)` (simulating `Trainer` Epoch 1 → Epoch 2 after `freeze_backbone_epochs=1`) **never** unfreezes frozen base parameters or `BatchNorm2d` running stats.
2. **`512×512` Native Forward/Backward Pass**:
   - Both Mode 1A (`supervised`) and Mode 1B (`contrastive`) run forward + backward cleanly on `[B, 3, 512, 512]` with `force_vit_1024=False` and `normalize=False` on both `FOCALViT` and `FOCALHRNet`.
3. **Prior-Drift Containment**:
   - A short multi-step finetuning test confirms: (a) frozen base weights remain bit-identical before and after optimizer steps (`torch.equal`), (b) `prior_anchor_weight > 0` produces finite positive anchor loss that bounds cosine drift from base features, and (c) `LoRA`/`DoRA` weights + `decoder`/`neck` update cleanly.
4. **Checkpoint & Merge Round-Trip**:
   - Saving a Strategy B checkpoint, reloading it, and calling `merge_peft_weights()` produces identical forward outputs (within floating-point tolerance) to the unmerged PEFT model.
5. **AMP / FP8 & Existing Regression Tests**:
   - `SurgicalLoRALinear` (both `use_dora=False` and `use_dora=True`) does not crash under `prepare_model_for_fp8_training()` or `bfloat16`/`float16` autocast.
   - `tests/test_focal_pim_mismatch_audit.py` and `tests/test_focal_strategy1.py` remain green alongside new `tests/test_focal_strategy_b.py` tests.

---

## 10. Out of Scope (Deferred to Later Strategy PRs)
- **Strategy C**: Dual-backbone fusion (`HRNet` + `FOCALViTNet` L2 concatenation / `ZeroGatedFiLM` learned fusion).
- **Strategy D**: End-to-end differentiable soft-EM clustering head.
- **Strategy E**: Multi-scale HRNet stage-4 branch fusion / intermediate ViT global block tapping & `ForensicVectorMemory` integration.
