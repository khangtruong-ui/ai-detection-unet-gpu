# Integration Note: IML MaskFormer with Official SAM 2.1 Hiera Vision Backbone

**Target Repository**: `ai-detection-unet-gpu`  
**Branch**: `fix-iml`  
**Model**: `IMLMaskformer` (aliases: `RefactorMaskformer`, `RefectorMaskformer`)  
**Backbone**: Meta Segment Anything Model 2.1 (SAM 2.1 Hiera Vision Backbone)  

---

## 1. Problem Diagnosis

During training execution via `sid-train --config configs/experiments/iml/iml_maskformer.yaml`, the runtime log reported:

```text
[INFO] Downloading SAM2 checkpoint from https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt to /content/sam2/checkpoints/sam2.1_hiera_large.pt...
[WARNING] Could not load official SAM2 trunk (No module named 'sam2'). Falling back to lightweight dummy trunk.
...
[INFO] Model parameters: 34,326,211
```

### Root Causes
1. **Missing Module Resolution & Package Dependencies**: The official `sam2` Python package from Meta (`facebookresearch/sam2`) was not pre-installed in the environment, and its required configuration library `hydra-core` was absent from base dependencies.
2. **Directory Conflict Preventing Automated Git Clones**: `_download_checkpoint` downloaded `sam2.1_hiera_large.pt` into `/content/sam2/checkpoints/` prior to cloning. Because `/content/sam2` already existed as a non-empty directory, naive git cloning failed.
3. **Silent Fallback to Dummy Trunk**: In `SAM2ImageEncoder.__init__`, any exception caught when attempting `build_sam2` silently logged a warning, set `self.use_dummy = True`, and left `self.image_encoder = None`. Consequently, only the Pixel Decoder and Transformer Decoder were instantiated (~34.3M trainable parameters), completely bypassing the ~212M parameter pretrained Hiera ViT vision trunk.
4. **Dropped Initialization Arguments**: In `PixelModule` and `PixelEncoder`, arguments passed to `IMLMaskformer` (`sam2_repo`, `config_path`, `checkpoint_path`, `device`) were not forwarded to `SAM2ImageEncoder`, causing fallback to default search paths.

---

## 2. Integrated Solution

### A. Robust SAM 2 Package & Repository Discovery (`_ensure_sam2_available`)
Added `_ensure_sam2_available(repo_dir: Path)` to automatically resolve and integrate Meta's official SAM 2 package:
- Checks if `sam2` and `build_sam2` are already importable in Python.
- If not, inspects `repo_dir` (e.g., `/content/sam2`) and candidate directories (`/workspace/sam2`, `./sam2`, `../sam2`) for a valid `sam2` package and injects it into `sys.path`.
- If missing, clones the lightweight repository to a staging directory (`/tmp/sam2_clone`) and copies repository files into `repo_dir` without disturbing existing checkpoints or subdirectories.

### B. Strict Dummy Fallback Prevention
When `use_dummy=False` (default in training and evaluation):
- Silent fallback to `use_dummy=True` is strictly disabled.
- If official weights fail to load, a descriptive `RuntimeError` is raised with full diagnostic details.
- Dummy weights are only used when explicitly requested (`use_dummy=True`, e.g., in unit tests for fast offline execution).

### C. Complete Parameter Forwarding
Updated the hierarchy:
$$\text{IMLMaskformer} \longrightarrow \text{PixelModule} \longrightarrow \text{PixelEncoder} \longrightarrow \text{SAM2ImageEncoder}$$
Passing `sam2_repo`, `config_path`, `checkpoint_path`, and `device` seamlessly down to the vision backbone loader.

### D. Device and Data Type Alignment
Enhanced `SAM2ImageEncoder.forward` to verify device alignment:
- Automatically aligns input image tensors with the trunk's parameter device before passing into `self.image_encoder`.
- Supports mixed precision (`bf16` / `fp16`) under `torch.amp.autocast`.

### E. Packaging and Dependencies
- Added `hydra-core>=1.3.2`, `omegaconf>=2.3.0`, and `iopath>=0.1.10` to core dependencies in `pyproject.toml`.
- Added an explicit `[project.optional-dependencies]` target `iml = ["hydra-core>=1.3.2", "omegaconf>=2.3.0", "iopath>=0.1.10"]`.
- Added `hydra-core>=1.3.2` to the `all` dependency group.

---

## 3. Verification & Metrics

- **Parameter Count Verification**:
  - Without trunk (dummy): **34,326,211 parameters** (only Pixel Decoder and Transformer Decoder).
  - With official SAM 2.1 Hiera Large trunk: **246,475,507 total parameters** (212,149,296 frozen vision trunk parameters + 34,326,211 trainable decoder parameters).
- **Trunk Output Shapes**:
  - $F_4$: `[B, 144, 96, 96]`
  - $F_8$: `[B, 288, 48, 48]`
  - $F_{16}$: `[B, 576, 24, 24]`
  - $F_{32}$: `[B, 1152, 12, 12]`
- **Forward & Loss Execution**:
  - Full forward pass produces `dense_mask_logits` of shape `[B, 1, 384, 384]`.
  - Bipartite Hungarian matching and `RefactorMaskformerLoss` compute finite losses without NaN/Inf across all pyramid levels.
