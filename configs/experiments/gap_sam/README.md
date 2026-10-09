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

| Config File | Backbone | Variant | VAE | LoRA $(r, \alpha)$ | Target Size | Use Case |
|---|---|---|---|---|---|---|
| [`default.yaml`](default.yaml) | `tinyvit` | `11m` | `sd-vae-ft-mse` | $(16, 32)$ | $1008 \times 1008$ | General-purpose GAP-SAM baseline on COCO-inpainted |
| [`gap_sam_tinyvit_lora.yaml`](gap_sam_tinyvit_lora.yaml) | `tinyvit` | `11m` | `sd-vae-ft-mse` | $(16, 32)$ | $1008 \times 1008$ | Optimized TinyViT configuration |
| [`gap_sam_efficientvit_lora.yaml`](gap_sam_efficientvit_lora.yaml) | `efficientvit` | `b0` | `sd-vae-ft-mse` | $(16, 32)$ | $1008 \times 1008$ | Real-time / mobile inference latency profile |
| [`gap_sam_repvit_lora.yaml`](gap_sam_repvit_lora.yaml) | `repvit` | `m1.1` | `sd-vae-ft-mse` | $(16, 32)$ | $1008 \times 1008$ | Structural re-parameterization vision backbone |
| [`gap_sam_sd21_vae.yaml`](gap_sam_sd21_vae.yaml) | `tinyvit` | `11m` | `SD 2.1 (vae)` | $(8, 16)$ | $1008 \times 1008$ | Direct reproduction of paper setup |

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
