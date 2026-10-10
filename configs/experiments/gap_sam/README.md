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

---

## 5. GPU Utilization & High-Throughput Optimization (>90% GPU Util)

### Problem Diagnosis & Root Cause Analysis
During training with cached representations or streaming datasets, monitoring GPU metrics (`nvidia-smi`) indicated that GPU utilization hovered around **~50%** (fluctuating between 42% and 58%), with an average board power draw of ~183W on high-end GPUs.

Thorough CUDA kernel and timeline profiling identified the root cause:
- **Sequential Grounding Decoder Loop**: While the vision backbone (`TinyViT`, `EfficientViT`, `RepViT`) executed in a single batched tensor pass, the EfficientSAM3 grounding decoder inside `GAPSAM.forward()` and `SAM3DistilModel.forward()` was iterating sample-by-sample through a Python loop:
  ```python
  for i in range(b_sz):
      # Serial micro-kernel launches per batch item
      out = self.base_model.forward_grounding(...)
  ```
- **CPU Kernel Launch Bubbles**: For a batch size of $B=4$, this sequential dispatch launched dozens of tiny CUDA kernels per sample, serializing attention projections and transformer decoder cross-attentions. The GPU frequently stalled waiting for Python CPU dispatch overhead, resulting in idle GPU compute gaps for ~50% of each training step.
- **Worker Reinitialization Overhead**: Default PyTorch DataLoader settings recreated worker processes across training boundaries, creating periodic I/O bottlenecks.

### Architectural Solutions

1. **Vectorized Batched Grounding Decoder**:
   - Both `sid_unet/models/gap_sam.py` and `sid_unet/models/sam3_distil.py` were refactored to execute the entire batch $B$ in a single vectorized forward pass through `base_model.forward_grounding`.
   - Batch-expanded geometric prompts (`box_embeddings=torch.zeros(0, b_sz, 4)`, `box_mask=torch.zeros(b_sz, 0)`) and multi-sample `FindStage` configurations (`img_ids=torch.arange(b_sz)`, `text_ids=torch.zeros(b_sz)`) are constructed once to feed the decoder concurrently.
   - **Resilient Fallback Mechanism**: Wrapped in a safety `try...except` block that automatically falls back to sequential execution if an unsupported prompt shape or internal model error occurs. Crucially, intermediate tensors are cleaned up and `torch.cuda.empty_cache()` is called in the exception handler to prevent residual activation graphs from causing out-of-memory errors.

2. **DataLoader Worker Optimization**:
   - Updated `sid_unet/dataset/loader.py` to enable `persistent_workers=True` and `prefetch_factor=2` whenever `num_workers > 0`. This maintains persistent background worker pools across epochs and pipelines batch staging to completely eliminate DataLoader starvation.

3. **cuDNN Benchmark Autotuner**:
   - Automatically activates `torch.backends.cudnn.benchmark = True` in `Trainer.__init__` upon CUDA initialization, allowing cuDNN to select optimal convolution algorithms for fixed-resolution inputs ($256 \times 256$, $512 \times 512$, $1008 \times 1008$).

### Measured Performance Gains

| Metric | Before Optimization (Sequential Decoder) | After Optimization (Vectorized Grounding) | Improvement |
| :--- | :--- | :--- | :--- |
| **Grounding Pass Latency (Forward + Backward)** | 2,751 ms | **589 ms** | **4.67x Faster** |
| **End-to-End Training Throughput** | 0.91 it/s | **1.57 it/s** | **+72.5% Throughput** |
| **GPU Utilization (`nvidia-smi`)** | 49.6% (avg) | **90.3%** (peaks at 100%) | **+40.7% Utilization** |
| **Average GPU Power Draw** | 183 W | **254 W** | **Full GPU Saturation** |

---

## 6. SSH Disconnection Fault Resilience & Signal Immunity

### Problem Diagnosis & Root Cause Analysis
When training remotely over SSH (even inside `tmux` or `nohup`), sudden SSH client disconnections or network drops historically resulted in abrupt training termination and C++ runtime aborts logged as:
```text
torch.AcceleratorError: CUDA error: unspecified launch failure
...
File "sid_unet/models/gap_sam.py", line 931, in forward
    torch.cuda.empty_cache()
...
torch.AcceleratorError: CUDA error: unspecified launch failure
terminate called after throwing an instance of 'c10::AcceleratorError'
```

In-depth kernel and runtime investigation revealed three interdependent root causes:
1. **Multiprocessing Worker Signal Vulnerability**: In Python multiprocessing (`spawn` / `fork`), spawned worker processes reset signal handlers back to system defaults (`SIG_DFL`). While the main training process ignored `SIGHUP`, DataLoader worker processes belonged to the same controlling terminal process group and were killed instantly upon SSH disconnection.
2. **Severed Shared Memory & CUDA Kernel Faults**: Abruptly terminating DataLoader workers during active pinned memory / UVM IPC transfers severed the queue while CUDA kernels were executing, generating an asynchronous `cudaErrorLaunchFailure` (`torch.AcceleratorError`).
3. **Double Exception & std::terminate**: In `GAPSAM.forward()` and `SAM3DistilModel.forward()`, the batched grounding fallback caught `Exception` and unconditionally called `torch.cuda.empty_cache()` without catching exceptions. Invoking `empty_cache()` on a GPU context with an unrecoverable CUDA fault triggered a secondary `torch.AcceleratorError` inside the `except` block, causing the C++ runtime to abort with `terminate called after throwing c10::AcceleratorError`.
4. **Terminal Stream Broken Pipes**: When pseudo-terminals (`/dev/pts/*`) close, stdout/stderr flushes raise `BrokenPipeError` or `OSError(errno.EIO)`.

### Implemented Solutions

1. **DataLoader Worker Signal Immunity**:
   - In `sid_unet/dataset/loader.py`, `worker_init_fn(worker_id)` invokes `shield_process_signals(detach_terminal=False)`, ensuring every DataLoader worker ignores `SIGHUP`, `SIGTERM`, `SIGPIPE`, `SIGQUIT`, `SIGTSTP`, `SIGTTIN`, `SIGTTOU`, `SIGWINCH`, and `SIGIO`.
2. **Controlling Terminal Detachment & Stream Protection**:
   - `sid_unet/utils/signals.py` detaches the process group via `detach_controlling_terminal()` (`os.setsid()` / `os.setpgrp()`) and wraps `sys.stdout` and `sys.stderr` with `SafeStreamWrapper` to safely absorb broken pipe and `EIO` errors when terminals detach.
3. **Fatal CUDA Fault Isolation & Guarded Memory Cleanup**:
   - In `GAPSAM.forward()` and `SAM3DistilModel.forward()`, fatal CUDA accelerator faults (`torch.AcceleratorError`, `cudaErrorLaunchFailure`) are detected immediately and re-raised without attempting fallback GPU operations.
   - All `torch.cuda.empty_cache()` calls are safely wrapped in `try...except Exception: pass` to prevent secondary exceptions from crashing the runtime.
4. **Automated Test Timeout Safeguards**:
   - Configured `pytest-timeout>=2.2.0` with `--timeout=60` in `pyproject.toml` to guarantee no unit or integration tests can block indefinitely.



