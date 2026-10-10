# GAP-SAM: Global Artifact Prior with Distilled SAM3 Models

Implementation of **GAP-SAM** (*"A Global Artifact Prior for Generalizable AI-Generated Image Manipulation Localization"*, Yan et al., arXiv:2608.20929), adapted to use the distilled foundation models from the **`sam-distil`** module (`EfficientSAM3`: TinyViT, EfficientViT, and RepViT backbones) with parameter-efficient fine-tuning via **PEFT Low-Rank Adaptation (LoRA)**.

---

## 1. Motivation & Background

Existing AI image manipulation localizers fine-tune semantic segmentation foundation models, but often suffer from **boundary adhesion**: the tendency of fine-tuned models to snap predictions to semantic object contours (people, furniture, cars) rather than true manipulation boundaries. Furthermore, dense pixel supervision entangles forensic evidence with dataset-specific mask geometry and semantic shortcuts.

**GAP-SAM** solves this by introducing a **Global Artifact Prior**:
1. **Frozen VAE Reconstruction:** Takes an input image $\mathbf{x}$ and passes it through a frozen Variational Autoencoder (e.g. Stable Diffusion 2.1 or 1.5 AutoencoderKL) to produce a deterministic reconstruction $\mathbf{x}_{\text{rec}}$.
2. **Dual Vision Branches:**
   - **Adaptive Branch:** Encodes observed image $\mathbf{x}$ with trainable LoRA adapters, yielding multiscale FPN features $\mathcal{F}_o$ and final-layer map $\mathbf{F}_o^L$.
   - **Frozen Branch:** Encodes reconstructed image $\mathbf{x}_{\text{rec}}$ through a frozen configuration of the EfficientSAM3 backbone, yielding $\mathbf{F}_r^L$.
3. **Paired Artifact Encoder:**
   - Computes Global Average Pooling (GAP) and branch-specific LayerNorms:
     $$\mathbf{h}_o = \text{LayerNorm}_o(\text{GAP}(\mathbf{F}_o^L)), \quad \mathbf{h}_r = \text{LayerNorm}_r(\text{GAP}(\mathbf{F}_r^L))$$
   - Concatenates the pooled descriptors and applies a 2-layer interaction MLP with GELU and intermediate LayerNorm:
     $$\mathbf{t}_{\text{art}} = \mathbf{W}_2 \, \text{LayerNorm}(\text{GELU}(\mathbf{W}_1 [\mathbf{h}_o ; \mathbf{h}_r] + \mathbf{b}_1)) + \mathbf{b}_2$$
     producing a 256-dimensional Global Artifact Prior Token $\mathbf{t}_{\text{art}}$.
4. **Zero-Gated FiLM Conditioning:**
   - Modulates each level $l$ of the FPN multiscale feature pyramid before mask decoding:
     $$\gamma^l, \beta^l = \text{split}(\mathbf{W}_f^{\text{FPN}} \mathbf{t}_{\text{art}} + \mathbf{b}_f^{\text{FPN}})$$
     $$\tilde{\mathbf{F}}_o^l = \mathbf{F}_o^l + \alpha_f \cdot (\gamma^l \odot \mathbf{F}_o^l + \beta^l)$$
   - The shared scalar gate $\alpha_f$ is initialized to zero, ensuring exact identity mapping at initialization and smoothly learning to inject forensic prior context without inserting spatial boundary shortcuts.
5. **Artifact Classifier:**
   - Attaches a linear classifier $c(\mathbf{h}) = \sigma(\mathbf{W}_c \mathbf{h} + b_c)$ to pooled descriptors $\mathbf{h}_o$ and $\mathbf{h}_r$.
   - Enforces a binary cross-entropy loss $\mathcal{L}_{\text{art}}$ anchoring the paired latent space along the real-versus-synthetic forensic axis:
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{loc}} + \lambda_{\text{art}} \mathcal{L}_{\text{art}}, \quad (\lambda_{\text{art}} = 1.0)$$

---

## 2. Configuration Files

All configurations strictly align with the hyperparameters reported in the paper (arXiv:2608.20929):
- **LoRA Hyperparameters:** Rank $r=8$, scaling factor $\alpha=16$, dropout $0.0$.
- **Training Epochs:** 3 epochs with AdamW optimizer, learning rate $2\times 10^{-4}$, weight decay $0.05$, effective batch size $32$ ($8 \times 4$).
- **Intra-Epoch Evaluation & Early Stopping:** Validation loss evaluated every $0.1$ epoch (`val_check_interval: 0.1`) with early stopping patience of 5 evaluations (`early_stopping_metric: val_loss`, `early_stopping_mode: min`).
- **Learning Objective:** $\mathcal{L} = \mathcal{L}_{\mathrm{SAM3}} + \lambda_{\mathrm{art}} \mathcal{L}_{\mathrm{art}}$ ($\lambda_{\mathrm{art}}=1.0$). Pure GAP-SAM without semantic/tamper-type auxiliary classifier head (`aux_classifier: false`), eliminating unnecessary parameters and compute overhead.

| Config File | Backbone | Variant | VAE | LoRA $(r, \alpha, \text{dropout})$ | Epochs | Eval Interval | Use Case |
|---|---|---|---|---|---|---|---|
| [`default.yaml`](default.yaml) | `tinyvit` | `11m` | `sd-vae-ft-mse` | $(8, 16, 0.0)$ | 3 | $0.1$ epoch | Paper-aligned GAP-SAM baseline on COCO-inpainted |
| [`gap_sam_tinyvit_lora.yaml`](gap_sam_tinyvit_lora.yaml) | `tinyvit` | `11m` | `sd-vae-ft-mse` | $(8, 16, 0.0)$ | 3 | $0.1$ epoch | Optimized TinyViT configuration |
| [`gap_sam_efficientvit_lora.yaml`](gap_sam_efficientvit_lora.yaml) | `efficientvit` | `b0` | `sd-vae-ft-mse` | $(8, 16, 0.0)$ | 3 | $0.1$ epoch | Real-time / mobile inference latency profile |
| [`gap_sam_repvit_lora.yaml`](gap_sam_repvit_lora.yaml) | `repvit` | `m1.1` | `sd-vae-ft-mse` | $(8, 16, 0.0)$ | 3 | $0.1$ epoch | Structural re-parameterization vision backbone |
| [`gap_sam_sd21_vae.yaml`](gap_sam_sd21_vae.yaml) | `tinyvit` | `11m` | `SD 2.1 (vae)` | $(8, 16, 0.0)$ | 3 | $0.1$ epoch | Stable Diffusion 2.1 VAE configuration matching paper |

---

## 3. CLI Usage

### Training
```bash
# Train default GAP-SAM (TinyViT-11m) on single GPU
sid-train --config configs/experiments/gap_sam/default.yaml

# Train GAP-SAM with EfficientViT-b0 for faster edge inference
sid-train --config configs/experiments/gap_sam/gap_sam_efficientvit_lora.yaml

# Train with Stable Diffusion 2.1 VAE configuration matching paper
sid-train --config configs/experiments/gap_sam/gap_sam_sd21_vae.yaml
```

### Evaluation
```bash
# Evaluate checkpoint on validation split
sid-eval --config configs/experiments/gap_sam/default.yaml \
         --checkpoint outputs/experiments/gap_sam/default/checkpoints/checkpoint_best.pt
```

### Cross-Dataset Generalization (OOD Benchmark)
```bash
# Evaluate on DiffSeg30K, CASIA, or OpenSDID
sid-cross-eval --config configs/cross-eval/diffseg30k.yaml \
               --checkpoint outputs/experiments/gap_sam/default/checkpoints/checkpoint_best.pt
```

### Inference / Prediction
```bash
sid-predict --checkpoint outputs/experiments/gap_sam/default/checkpoints/checkpoint_best.pt \
            --input path/to/suspicious_image.png \
            --output outputs/predictions/
```

---

## 4. Caching & High-Throughput Training Optimizations

### Forensic Profiling & Latency Bottleneck
In standard GAP-SAM training, the model executes two distinct branches:
1. **Adaptive Branch:** Learns manipulation features via LoRA-adapted EfficientSAM3 backbones.
2. **Frozen Artifact Branch:** Runs input $\mathbf{x}$ through a full Stable Diffusion Variational Autoencoder (`AutoencoderKL`, 84M parameters) at high resolution ($1008 \times 1008$) to produce reconstruction $\mathbf{x}_{\text{rec}}$, then encodes $\mathbf{x}_{\text{rec}}$ through a frozen SAM vision backbone to extract $\mathbf{F}_r^L$ and pooled descriptor $\mathbf{gap}_r \in \mathbb{R}^{B \times 256}$.

On an NVIDIA RTX 3060 (12 GB VRAM):
- Frozen VAE encode/decode: **~4.1s per batch of 2**, peaking at **~8.7 GB VRAM**.
- Frozen SAM vision backbone: **~0.38s**, peaking at **~1.15 GB VRAM**.
- **Key Insight:** The entire frozen branch is completely deterministic and exists solely to compute the 256-dimensional vector $\mathbf{gap}_r$ (a negligible ~1 KB per image). Recomputing this on every single training epoch wastes over 80% of computation time and bloats VRAM requirements.
- **Text Encoder:** Precomputed once and cached via text cache (`self._text_cache`), avoiding repeated forward passes.

### Two-Tier Caching Architecture

#### 1. In-Memory Dynamic Artifact Cache (Zero-Setup)
When training directly from image datasets, the model dynamically caches extracted $\mathbf{gap}_r$ vectors in RAM:
```yaml
# configs/experiments/gap_sam/default.yaml
model:
  enable_artifact_cache: true
  artifact_cache_size: 50000  # LRU cache holding 50k descriptors (~50 MB RAM)
```
- During Epoch 1, $\mathbf{gap}_r$ descriptors are computed and indexed by `img_id` (or image content MD5 hash).
- From Epoch 2 onwards, all cache hits completely skip the frozen VAE reconstruction and frozen vision backbone, accelerating step time by **~5x** and halving peak VRAM usage.

#### 2. Offline Precomputed Dataset Cache (`sid-cache`)
For maximum throughput and massive multi-epoch runs, precompute descriptors offline:
```bash
# 1. Precompute gap_r descriptors into Hugging Face Parquet dataset shards (stores images automatically)
sid-cache --model gap_sam \
          --config configs/experiments/gap_sam/default.yaml \
          --hf-repo KhangTruong/gap-sam-coco-cache \
          --batch-size 8 \
          --push-to-hub

# 2. Train with zero VAE overhead and bypassed AutoencoderKL
sid-train --config configs/experiments/gap_sam/default.yaml \
          --cached-hf-repo KhangTruong/gap-sam-coco-cache
```

##### Parquet Shard Schema & Architecture Differences
Unlike `diffusion_diff` (which replaces input images with high-dimensional 84/244-channel spatial latent feature maps), GAP-SAM operates with a hybrid representation:
- **`z_high_dim`**: Stores the 256-dimensional pooled descriptor $\mathbf{gap}_r$ extracted from the frozen VAE reconstruction branch as a 1D float16 array.
- **`latent_h` & `latent_w`**: Correctly serialized as $1 \times 1$ for 1D vectors (avoiding shape mismatches).
- **`image`**: Compressed JPEG bytes preserving the original RGB image, which is required by the trainable LoRA vision backbone (EfficientSAM3) for mask segmentation.
- **`mask`**: Compressed PNG bytes of the ground-truth binary tampering mask.

##### Backward Compatibility with Legacy Shards
If training on legacy remote repositories (such as older shards where `image` was omitted and spatial dimensions were hardcoded to $32 \times 32$):
- **Robust Tensor Decoding**: `decode_cached_tensor` automatically detects 1D vectors ($256$ elements) and prevents `ValueError: cannot reshape array of size 256 into shape (256, 32, 32)`.
- **Image Fallback / Pairing**: Datasets seamlessly fall back to an active image provider or synthesized image tensor without crashing.
- **Bypassed VAE Probing**: Automatic batch size memory probing cleanly injects pre-allocated $\mathbf{f}_r$ dummy vectors, allowing safe batch size auto-scaling even when the VAE is purged from GPU VRAM.

When `--cached-hf-repo` is passed:
- `bypass_vae_for_cached_training()` deletes `AutoencoderKL` from GPU memory.
- `f_r` / `gap_r` tensors are streamed directly from disk/network alongside image and mask tensors.
- GPU VRAM consumption drops to just the trainable LoRA backbone (~2.5 GB peak VRAM), allowing batch sizes up to $16\times$ larger and blazing-fast epoch runtimes.

