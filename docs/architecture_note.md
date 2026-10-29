# Architecture Note: Learnable Forensic Vector Memory for Detective SAM

**Author**: Antigravity Assistant  
**Date**: October 29, 2026  
**Target Repository**: `ai-detection-unet-gpu` (Branch: `discrete-detective-sam`)  
**Task**: Implement a Learnable Forensic Vector Memory for Detective SAM  

---

## 1. Executive Summary

This architecture note establishes the technical blueprint for integrating a CPU-resident Learnable Forensic Vector Memory into **Detective SAM** (*ICLR 2026*). 

The goal is to store forensic representations extracted from **ground-truth tampered image regions** during training, perform similarity-based retrieval during inference/forward pass, and substitute the forward value of a transformer query tensor via a surrogate stop-gradient construction:
$$R_x = R + (R_t - R).\text{detach}()$$
preserving the identity-path backward gradient to $R$ while preventing gradients from flowing into the retrieved CPU memory vectors $R_t$.

---

## 2. Detective SAM Architecture & Tensor Flow Trace

Detective SAM couples Meta's foundation model (SAM2 Hiera) with forensic adaptation modules:

```
[Input Image X: B x 3 x H x W]
         │
         ├──> [ImagePerturbation] ───────────────> [Perturbed Image X': B x 3 x H x W]
         │                                                        │
         └───┬────────────────────────────────────────────────────┘
             ▼
[Frozen SAM2 Hiera Vision Encoder (facebook/sam2-hiera-small)]
             │
             ├── Unperturbed Hierarchical Features: F_i = [F_0, F_1, F_2]
             │   (Channels: [32, 64, 256]; Spatials: 1/4, 1/8, 1/16 input resolution)
             └── Perturbed Hierarchical Features:   F_pert = [F'_0, F'_1, F'_2]
             │
             ▼
[Multi-Scale Feature Adapters: FeatureAdapter (1x1 Conv delta correction)]
             │
             └── Adapted Features: F_tilde_s = F_s + A_s([F_s, F'_s])
             │
             ▼
[Automatic Mask Adapter: MaskAdapter]
             │
             ├── Cross-Scale & Cross-Stream Conv Fusion:
             │   F_fuse in R^{B x d x s_hat x s_hat} (d = 64, s_hat = H/4, W/4)
             │
             ├── Coarse Transformer Branch:
             │   F_down = AdaptiveAvgPool2d(F_fuse, (coarse_h, coarse_w)) -> [B, d, 8, 8]
             │   Tokens R in R^{B x N_coarse x d} (N_coarse = 64 tokens, d = 64)
             │   Transformer Encoder: R -> trans_out -> Coarse Head -> [L_coarse, U]
             │
             ├── Fine Refinement Branch:
             │   F_fuse -> 2-layer ConvNet -> L_refine in R^{B x 1 x s_hat x s_hat}
             │
             └── Linear Spatial Gating:
                 g = sigmoid(Conv1x1([L_coarse, U]))
                 M = g * L_refine + (1 - g) * L_coarse (dense heatmap prompt)
             │
             ▼
[SAM2 Prompt Encoder & Mask Decoder (Frozen)]
             │
             ├── Prompt Encoder: input_masks = M_256 (256x256) -> Dense Prompt Embeddings (64x64x256)
             │                   Dummy Point -> Sparse Prompt Embeddings [B, 1, 1, 256]
             │
             └── Sam2MaskDecoder:
                 Queries: point_embeddings = cat([output_tokens, sparse_prompt_embeddings]) [B, 1, 7, 256]
                 output_tokens = [obj_score_token (1), iou_token (1), mask_tokens (4)]
                 Two-Way Cross-Attention with coarsest adapted feature F_tilde_2 (256 ch)
                 Upscale Convolutions & Hypernetwork MLP -> Predicted Mask Logits [B, 1, H, W]
```

---

## 3. Query Tensor Selection & Technical Analysis

### 3.1 Candidate Integration Points

We identified two transformer query candidate tensors in the pipeline:

1. **Candidate 1: MaskAdapter Coarse Transformer Query Sequence ($R$)**
   - **Source Module**: `MaskAdapter` (lines 192–202, 262–270 in `sid_unet/models/detective_sam.py`).
   - **Tensor**: Coarse spatial token sequence:
     $$R = \text{tokens} = \text{Flatten}(\text{AvgPool}(F_{\text{fuse}})) \in \mathbb{R}^{B \times N_{\text{coarse}} \times d}$$
     with $d=64$, $N_{\text{coarse}} = 64$ (for default $8 \times 8$ downscaling).
   - **Downstream Consumers**: `MaskAdapter.transformer` (`nn.TransformerEncoderLayer`), `coarse_head`, spatial gating $g$, heatmap prompt $M$, and SAM2 `prompt_encoder`.
   - **Trainable Gradient Path**: Active and direct. The tensor $R$ is dynamically computed from $F_{\text{fuse}}$, which receives backward gradients from the entire loss objective. $d R_x / d R = I$ propagates back directly into the trainable `FeatureAdapter` layers and `MaskAdapter` fusion weights.
   - **Compatibility**: Operates identically whether `sam2_model` is in full pretrained mode or dummy test mode.

2. **Candidate 2: SAM2 Mask Decoder Query Tokens (`point_embeddings`)**
   - **Source Module**: `Sam2MaskDecoder` (`output_tokens` concatenated with `sparse_prompt_embeddings`).
   - **Tensor**: Shape $[B, 1, 7, 256]$.
   - **Issue / Incompatibility**: In standard Detective SAM, the base SAM2 model is strictly frozen (`freeze_sam2: True`, `freeze_decoder: True`).
   - The query tokens (`mask_tokens.weight` $[4, 256]$, `iou_token.weight` $[1, 256]$, `obj_score_token.weight` $[1, 256]$) are static frozen embeddings. If $R$ is selected as `output_tokens`, the identity gradient $d R_x / d R = I$ terminates at frozen parameters and yields zero parameter updates during training.
   - In addition, Detective SAM uses automatic dense mask prompts $M$ rather than point prompts; `sparse_prompt_embeddings` is merely a dummy padding point with label `-1`.

### 3.2 Selected Integration Point

We select **Candidate 1: `MaskAdapter` Coarse Transformer Tokens ($R \in \mathbb{R}^{B \times N_{\text{coarse}} \times d}$)** as the primary integration point. 

**Why it is the appropriate integration point:**
1. It is the core transformer representation introduced by Detective SAM that drives prompt formulation.
2. It has an active, uninterrupted autograd path to trainable adapters.
3. Replacing $R$ with retrieved tampered forensic representations directly informs the coarse tampering likelihood map $L_{\text{coarse}}$ and uncertainty $U$, altering the spatial gating and downstream SAM2 prompt.
4. It works consistently in both full-scale GPU execution and standalone CPU unit testing.

---

## 4. Ground-Truth Mask Alignment & Feature Sampling

### 4.1 Feature Map Alignment
- Input image resolution: $(H, W) = (512, 512)$ (or target resolution).
- Ground-truth binary mask: $Y \in \{0, 1\}^{B \times 1 \times H \times W}$.
- Feature representation grid $F_{\text{fuse}}$: $(s_{\text{hat}}, s_{\text{hat}}) = (128, 128)$.
- Feature-map coordinate mapping:
  $$Y_{\text{feat}} = \left(\text{Interpolate}(Y.\text{float}(), \text{size}=(s_{\text{hat}}, s_{\text{hat}}), \text{mode}='area') \ge 0.5\right)$$
  Area downsampling preserves true positive tampering coverage while strictly suppressing sub-pixel interpolation artifacts.

### 4.2 Positive Region Sampling Policy
- For each image $b$ in the training batch:
  1. Identify valid spatial locations where $Y_{\text{feat}}[b, 0, y, x] == 1$.
  2. If the image has zero tampered pixels (authentic/untampered image), **zero** vectors are collected (strict authentic exclusion).
  3. If tampered pixels exist, sample up to $K_{\text{per\_img}} = \text{memory\_vectors\_per\_image}$ (default: 16) spatial coordinates.
  4. Extract corresponding feature vectors $v \in \mathbb{R}^{d}$ from $F_{\text{fuse}}[b, :, y, x]$.
  5. Detach from the computation graph and transfer in a single batch to CPU memory:
     $$\text{epoch\_observations}.\text{append}(v.\text{detach}().\text{cpu}())$$

---

## 5. Tensor Shapes & Specifications

| Tensor | Role | Dimension / Shape | Device | Dtype |
| :--- | :--- | :--- | :--- | :--- |
| `keys` | Stored forensic memory keys | $[N_{\text{mem}}, d]$ ($d=64$, $N_{\text{mem}} \le N_{\text{max}}$) | CPU | `float32` |
| `values` | Stored forensic memory values | $[N_{\text{mem}}, d]$ ($d=64$) | CPU | `float32` |
| $q$ | Image query key | $[B, d]$ (global average pool of $R$) | GPU/CPU | `float32` |
| $V_{\text{ret}}$ | Retrieved nearest-neighbor vectors | $[B, k, d]$ ($k=\text{retrieval\_k}$) | GPU | Same as $R$ |
| $R$ | Original coarse query tensor | $[B, N_{\text{coarse}}, d]$ | GPU | `float32` / `bfloat16` |
| $R_t$ | Retrieved query tensor | $[B, N_{\text{coarse}}, d]$ | GPU | Same as $R$ |
| $R_x$ | Stop-gradient query tensor | $[B, N_{\text{coarse}}, d]$ | GPU | Same as $R$ |

### 5.1 Query Formulation & Retrieval
1. Global image-level query key:
   $$q_b = \frac{1}{N_{\text{coarse}}} \sum_{i=1}^{N_{\text{coarse}}} R[b, i, :]$$
2. For each image $b \in [1, B]$, search the CPU database using the configured similarity metric (Cosine, L2, Dot).
3. Retrieve top-$k$ value vectors $V_{\text{ret}}[b] \in \mathbb{R}^{k \times d}$.
4. Form $R_t[b]$ by aggregating retrieved vectors and projecting/broadcasting to $[N_{\text{coarse}}, d]$:
   $$R_t[b, i, :] = \frac{1}{k} \sum_{j=1}^k V_{\text{ret}}[b, j, :]$$
5. Apply the stop-gradient operator:
   $$R_x = R + (R_t - R).\text{detach}()$$

---

## 6. Memory Lifecycle & Consolidation Strategy

1. **Initialization & Warm-Up**:
   - Memory starts empty: $N_{\text{mem}} = 0$.
   - Warm-up condition: Retrieval is bypassed until $N_{\text{mem}} \ge \text{memory\_retrieval\_k}$. During warm-up, $R_x = R$ (exact baseline behavior).
2. **Epoch Collection**:
   - Observations accumulate in `epoch_observations` (CPU-resident list of tensors).
   - Training batches access only the consolidated database from previous epochs (no future-batch leakage).
3. **Epoch-End Consolidation**:
   - Observations are concatenated and deduplicated against existing memory using chunked similarity:
     - Vector chunk size: 512 candidates.
     - Vectors with similarity $\ge \text{memory\_dedup\_threshold}$ to an existing entry are discarded.
   - Bounded database enforcement: If total entries exceed `memory_max_entries`, deterministic diversity/strided pruning or k-means selection with `memory_seed` bounds the size to `memory_max_entries`.
   - Never forms an unrestricted $N \times N$ similarity matrix, ensuring $O(N)$ memory and predictable execution.
4. **Checkpointing & Resumption**:
   - `keys`, `values`, `entry_count`, and configuration parameters are saved into the checkpoint dictionary.
   - Full backward compatibility: Checkpoints without memory load cleanly in baseline mode.
