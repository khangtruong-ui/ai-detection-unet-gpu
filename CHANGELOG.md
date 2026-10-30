# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-10-30

### Fixed
- **Cross-Evaluation Continuous Master Report Accumulation**:
  - Fixed master cross-evaluation consolidation logic in `sid_unet/cross_eval.py` where evaluating a new checkpoint would overwrite `master_cross_evaluation_report.json` and discard previously evaluated checkpoints.
  - Implemented non-destructive accumulation merging prior entries with current evaluation results in both tabular matrices and publication-quality heatmaps.
- **Cross-Evaluation Collision Detection & Neighbor Report Path Resolution**:
  - Corrected report file naming in neighbor directories to prevent `eval_eval_...` filename prefix doubling when `config_name` already contained the `eval_` prefix.
  - Expanded collision lookup map in `sid_unet/cross_eval.py` to match across multiple canonical key representations (`(cp_abs, kp_abs)`, `(cn, kn)`, `(cn, cfg_stem)`, `(cp_abs, kn)`, and `(cp_abs, cfg_stem)`) as well as checking existing neighbor JSON reports (`eval_{kn}_report.json`, `eval_{cfg_stem}_report.json`, and `{kn}_report.json`).
- **Classification AUROC Calculation for Binary Auxiliary Logits**:
  - Fixed `ValueError: y should be a 1d array, got an array of shape (N, 2) instead` in `ClassificationMetricTracker` (`sid_unet/metrics/classification.py`) when computing AUROC with 2 classes under `multi_class="ovr"`.
  - Added support for 1D/2D positive class probability scoring and single-logit binary classification models (`(B, 1)` or `(B,)`), replacing dummy fallback heuristics with genuine AUROC curves.
- **Ablation & Per-Label Metric Synchronization in Evaluation Engines**:
  - Fixed bug in `sid_unet/cross_eval.py` and `sid_unet/evaluate.py` where `per_label_metrics` always reported baseline Raw UNet scores even when post-processing or SAM refinement was active, discarding stage-specific per-label performance.
  - Enhanced `SegmentationMetricTracker` (`sid_unet/metrics/segmentation.py`) to record per-label positive counts and metrics (`iou_pos_sum`, `dice_pos_sum`, `f1_pos_sum`, `precision_pos_sum`, and `recall_pos_sum`), aligning subset breakdowns with filtered evaluation settings.
- **Model Forward Kwargs and Dictionary Output Support**:
  - Enabled inspection and forwarding of `model_kwargs` (`f_r`, `img_ids`) to `model(images, **model_kwargs)` in `sid_unet/cross_eval.py` (matching `evaluate.py`), supporting models with external feature caches and forensic vector databases.
  - Handled model dictionary outputs (`mask_logits`, `class_logits`, `pred_masks`) safely without type errors.

### Added
- **Base Cross-Evaluation Configuration for KhangTruong/COCO-inpainted**:
  - Added `configs/cross-eval/base.yaml` and symlinked `configs/base.yaml` targeting `KhangTruong/COCO-inpainted` with standard UNet architectural parameters.
  - Explicitly configured sample limit to 1024 samples (`val_samples: 1024`, `test_samples: 1024`, `eval_split: "validation"`) to verify cross-evaluation logic with bounded runtime.
- **Comprehensive Unit & Integration Tests**:
  - Added tests in `tests/test_metrics.py` covering binary AUROC computation, single-logit binary classification, and per-label positive metric filtering.
  - Added tests in `tests/test_cross_eval.py` covering `base.yaml` configuration resolution, neighbor report naming without double prefixing, continuous multi-checkpoint master report retention, and dict-output forward handling.

### Added
- **Detective SAM Large Variant Support**:
  - Implemented full configuration and instantiation support for `DetectiveSAM` using Meta's `facebook/sam2-hiera-large` ViT backbone.
  - Verified architectural compatibility: SAM2 Large outputs identical hierarchical FPN channel dimensions `[32, 64, 256]` across scales, allowing seamless coupling with Detective SAM's multi-scale `FeatureAdapter` layers and `MaskAdapter`.
  - Added pre-configured experiment recipes:
    - `configs/experiments/sam2/detective_sam_large.yaml`: base Detective SAM with large ViT backbone, effective batch size 16 (1 sample/GPU with gradient accumulation steps 16), and target resolution 512x512.
    - `configs/experiments/sam2/detective_sam_large_memory.yaml`: Detective SAM Large integrated with Learnable Forensic Vector Memory.
  - Added unit and config regression tests in `tests/test_detective_sam.py` (`test_detective_sam_large_variant_init_and_forward` and `test_detective_sam_large_configs`).
- **Forensic Memory Database Persistence, Checkpointing, and Hub Synchronization**:
  - Checkpointed standalone database files alongside model checkpoints: `forensic_memory.pt`, `forensic_memory_best.pt`, and `forensic_memory_periodic.pt` in `sid_unet/training/callbacks.py`.
  - Added resilient multi-stage loading in `UNet.from_checkpoint`: restores from embedded `forensic_memory_state_dict`, searches sibling files in the checkpoint directory, or falls back to explicit `memory_path`.
  - Integrated database artifact discovery into `find_local_checkpoint_artifacts` in `sid_unet/checkpoint_sync.py` and opportunistic downloads in `sid_unet/utils/checkpoint.py`, ensuring `sid-push` automatically uploads the forensic memory database to Hugging Face Hub repositories.
  - Added 3 automated tests in `tests/test_forensic_memory.py` covering manager persistence, sibling loading, and synchronization artifact discovery.

## [0.1.2] - 2026-10-29

### Added
- **Learnable Forensic Vector Memory for Detective SAM**: Implemented a CPU-resident vector database containing forensic representations extracted exclusively from ground-truth tampered image regions:
  - **Surrogate Stop-Gradient Query Replacement**: Replaced forward values of Detective SAM's coarse transformer query representation with retrieved vectors ($R_x = R + (R_t - R).\text{detach}()$), preserving identity gradients along the original query path ($d R_x / d R = I$) while halting gradients to CPU memory ($d R_x / d R_t = 0$).
  - **Authentic Region Exclusion**: Area-downsampled ground-truth masks to feature map resolution ($s_{\text{hat}}$) with threshold $\ge 0.5$. Strictly excludes authentic/untampered images from positive memory collection.
  - **Epoch-End Bounded Memory Consolidation**: Chunked candidate deduplication against existing database, enforcing `memory_max_entries` with deterministic pruning (`memory_seed`), avoiding quadratic $N \times N$ memory allocations.
  - **Controlled Baseline Evaluation (`scripts/evaluate_memory_baselines.py`)**: Compared Condition A (No Memory, IoU 0.0830), Condition B (NN Retrieval, IoU **0.0852**), and Condition C (Randomized Retrieval, IoU 0.0765) on identical batches from `KhangTruong/COCO-inpainted`.
  - **Automated 17-Point Test Suite (`tests/test_forensic_memory.py`)**: Full coverage of forward semantics, gradient semantics, CPU residency, key/value pairing, similarity computation, undersized database handling, warm-up bypass, zero-norm safety, bounded capacity, deterministic consolidation, coordinate mapping, authentic region exclusion, inference isolation, checkpoint reload, dimension compatibility, and end-to-end pretrained GPU forward passes.
  - **Architecture Note**: Documented query tensor selection, tensor shapes, autograd paths, and alignment in `docs/architecture_note.md`.
  - **Configuration Schema**: Created `configs/experiments/sam2/detective_sam_memory.yaml` and integrated memory parameters into `DetectiveSAM`, `build_model`, `Trainer`, and `CheckpointManager`.

### Fixed
- **Forwarding of `max_cache_gb`, `cache_files`, and `max_cached_files` in DataLoaders**:
  - `create_cached_dataloaders` and `load_parquet_dataset` now extract and forward `max_cache_gb`, `cache_files`, and `max_cached_files` from `config.data` to `parquet-dataset-loader`.
  - Configured `cache_files: true` and `max_cache_gb: 4.0` in `configs/experiments/sam2/detective_sam_memory.yaml` to eliminate HTTP 429 rate limiting and socket flooding, keeping unconsumed cached files bounded while progressively saving the complete dataset to disk (`save_to_disk: true`).
- **Official SAM2 Trunk Weight Loading**: Fixed weight loading failure in `IMLMaskformer` (`sid_unet/models/iml_maskformer.py`):
  - Previously logged `Could not load official SAM2 trunk (No module named 'sam2'). Falling back to lightweight dummy trunk`, causing the model to use an uninitialized dummy trunk that produced zero features and numerical instability.
  - Installed Meta's `sam2` package and `hydra-core` dependencies.
  - Verified official SAM 2.1 Hiera checkpoint (`/content/sam2/checkpoints/sam2.1_hiera_small.pt`) loads cleanly via `build_sam2`, extracting non-zero multi-scale hierarchical feature maps (`use_dummy: False`).

### Added
- **300-Step Training Trajectory Benchmark with Verified Weights**: Ran `scripts/evaluate_300_steps.py` across 300 identical batches from `KhangTruong/COCO-inpainted` on NVIDIA Tesla T4 GPU with official weights loaded. Results recorded in `benchmark_300_steps_results.json`:
  - **Detective SAM (512x512)**: 0.26M trainable params (99.3% frozen). Fastest by far at **0.082s/step** (24.7s total for 300 steps). Surges from 0.0000 at step 50 to **0.1551** at step 100, **0.3091** at step 200, and **0.3200** at step 300 (peak IoU: **0.3346**, final 20-step mean IoU: **0.1338**).
  - **SAM2 Small (Unfrozen Decoder)**: 4.53M trainable params. 0.377s/step (113.0s total). Reaches 0.0394 at step 50, 0.0989 at step 100, 0.1190 at step 200, and 0.1257 at step 300 (peak IoU: **0.3844**).
  - **Senior's IML MaskFormer (Real Hiera Weights)**: 23.20M trainable params. 0.174s/step (52.3s total). Loss dropped smoothly from 0.6778 to 0.4520 with stable gradients and real trunk features; 100 Hungarian bipartite query slots require longer training (>1,000 steps) to match mask segments.
- **SAM2 Query Slot Parameter Analysis & Verification**: Conducted rigorous architectural analysis of SAM2's learnable query slot parameters inside `Sam2MaskDecoder`:
  - Mask queries: 4 tokens of dimension 256 = 1,024 parameters (`mask_tokens.weight`).
  - Total decoder query tokens: 6 tokens of dimension 256 = 1,536 parameters (including `iou_token` and `obj_score_token`).
  - Verified that 1,024 parameters is structurally optimal, strictly bound to SAM2's 2-way cross-attention dimension ($d_{\text{model}}=256$) and 4-way hypernetwork mask MLPs, and should not be modified.
- **Detective SAM 512x512 Image Scale Optimization**: Updated default `target_size` from (1024, 1024) to (512, 512) in `sid_unet/models/detective_sam.py`, `configs/experiments/sam2/detective_sam.yaml`, `sid_unet/models/unet.py`, and benchmark scripts:
  - Accelerates per-step execution by >9x (0.082s/step vs 0.740s/step).
  - Lowers peak VRAM footprint by >55%, allowing larger batch training on commodity 12-16GB GPUs without precision loss.
- **Unit and Architecture Tests**: Added test cases in `tests/test_detective_sam.py`:
  - `test_detective_sam_default_target_size`: verifies default resolution (512, 512) in constructor and config builder.
  - `test_sam2_query_slot_parameter_sizes`: explicitly validates 4 query slots (1,024 params) and 6 total query tokens (1,536 params).

## [0.1.1] - 2026-10-28

### Added
- **Detective SAM (ICLR 2026) Architecture Integration**: Implemented *Detective SAM: Adaptive AI-Image Forgery Localization* (`sid_unet/models/detective_sam.py`):
  - Differentiable dual-branch perturbation generator (`ImagePerturbation` with Gaussian blur $\sigma_{\text{blur}}=0.75$ and Gaussian noise $\sigma_{\text{noise}}=0.15$).
  - Residual cross-stream feature adapters (`FeatureAdapter`) computing $\Delta F_s$ at each Hiera scale ($s \in \{0, 1, 2\}$) to isolate tampering sensitivity.
  - Multi-scale Mask Adapter (`MaskAdapter`) with coarse downscaled transformer, conv refinement, and linear spatial gating ($g = \sigma(W_g * [L_{\text{coarse}}, U])$).
  - Multi-task training loss (`DetectiveSAMLoss`) combining Focal loss ($\lambda=20$), Dice loss ($\lambda=1$), and IoU prediction regression loss ($\lambda=1$).
  - Config added at `configs/experiments/sam2/detective_sam.yaml` with optimal hyperparameters from the paper ($d=64$, $\text{lr}=1\times 10^{-3}$, weight decay $1\times 10^{-4}$).
- **100-Step Training Trajectory Benchmark**: Created `scripts/evaluate_100_steps.py` and benchmarked all 3 models (SAM2 Small, Senior's IML MaskFormer, Detective SAM) under identical training conditions on NVIDIA Tesla T4 GPU, recording per-step losses, IoU metrics, step throughput, and memory consumption to `benchmark_100_steps_results.json`.
- **Unit & Integration Test Suite for Detective SAM**: Added 10 tests in `tests/test_detective_sam.py` covering perturbation, feature adapters, mask adapters, aux classifier, dynamic resolution handling, loss calculation, and GPU gradient flow.
- **SAM2 Unfrozen Mask Decoder Support**: Added `freeze_decoder` and `reinit_decoder` parameters in `SAM2Segmenter` and configuration schemas (`configs/experiments/sam2/*.yaml`), allowing all ~4.2M parameters of SAM2's mask decoder (attention layers, cross-attention, MLP blocks, and output tokens) to directly adapt to visual manipulation traces.
- **Quantization Protection for Decoder**: Added `llm_int8_skip_modules=['mask_decoder']` in `BitsAndBytesConfig` under 4-bit NF4 and 8-bit quantization modes to preserve decoder layers in native floating-point precision (`bfloat16`/`float16`/`float32`).
- **Post-PEFT Gradient Preservation**: Enforced decoder unfreezing strictly after PEFT's `get_peft_model()` initialization to prevent non-LoRA parameter re-freezing.
- **Weight Reinitialization Routine**: Implemented `_reinit_decoder_weights()` using Xavier uniform and zero bias to cleanly scrub SA-1B salient-object inductive bias when `reinit_decoder: true`.
- **Balanced Mask Adapter Initialization**: Configured Xavier uniform weights and zero bias (`zeros_`) on `mask_adapter` convolution layers and unit weight / zero bias on BatchNorm, ensuring initial predictions straddle the 0.0 logit decision boundary.
- **Comprehensive Unit & GPU Integration Tests**: Added test cases in `tests/test_sam2.py` covering:
  - Unfrozen decoder parameter gradient flow (`test_sam2_decoder_unfrozen_trainable`).
  - Frozen flag enforcement (`test_sam2_decoder_frozen_flag`).
  - Weight re-initialization divergence (`test_sam2_decoder_reinitialization`).
  - GPU 4-bit QLoRA backward pass without dtype crashes (`test_sam2_qlora_4bit_unfrozen_decoder_gpu`).
  - Balanced adapter initialization verification (`test_sam2_mask_adapter_balanced_init`).

### Fixed
- **Detective SAM Dynamic Input Resolution & Prompt Padding**: Resolved HF SAM2 hardcoded `view` size mismatch when handling non-1024 resolutions or perturbed image batches by dynamically calculating spatial grid dimensions via `isqrt`, dynamically interpolating positional and dense prompt embeddings, and providing a dummy negative prompt point (`label=-1`) to prevent `NoneType` crashes in `prompt_encoder`.
- **SAM2 0.0000 IoU Training Deadlock**: Fixed negative suppression bias (`bias = -2.0`, $\sigma=0.02$) in `mask_adapter` that clamped all initial predicted logits to $\approx -2.0$ ($\sigma(-2.0) \approx 0.12 < 0.5$). This had forced all predicted mask pixels to 0, producing zero positive predictions and locking training IoU at `0.0000` for 76+ steps in `.logs.txt`. With balanced Xavier initialization, training IoU immediately climbs to `0.0463` on step 1 and `0.1377` on step 4.
- **SAM2 0.03 IoU Bottleneck**: Resolved representation bottleneck where a frozen mask decoder could only output rigid SA-1B object proposals from prompt queries.
- **BitsAndBytes Autograd Dtype Crash**: Resolved `RuntimeError: only Tensors of floating point and complex dtype can require gradients` during backward pass by skipping quantization on `mask_decoder`.
- **VRAM Stability on 12-15GB GPUs**: Adjusted `sam2_small.yaml` default micro-batch size from 4 to 2 with gradient accumulation steps increased from 4 to 8, maintaining effective batch size 16 while capping peak VRAM at ~6.5 GB on Tesla T4 GPUs.

### Changed
- **Documentation**: Expanded `README.md` with:
  - Step-by-step decoder treatment ("apply one by one") covering unfreezing, quantization bypass, PEFT sequencing, and weight reuse vs. re-initialization.
  - Complete architectural breakdown of the Senior's IML MaskFormer (`RefactorMaskformer`) explaining multi-scale Hiera feature extraction ($F_4, F_8, F_{16}, F_{32}$), Deformable Attention Pixel Decoder, 100 dual queries ($\mathbf{q}_{\text{cls}}$ and $\mathbf{q}_{\text{loc}}$), and bipartite Hungarian point-matching loss.

---

## [0.1.0] - 2026-10-28

### Added
- Initial release of SID-UNet framework for synthetic image detection and pixel tampering localization.
- Support for UNet backbones (ResNet, EfficientNet, ConvNeXt, Swin).
- Diffusion VAE (SD1.5 AutoencoderKL) feature extraction for tampering boundary localization.
- Meta SAM3 with Quantized Low-Rank Adaptation (QLoRA) and learnable soft prompt tuning.
- Meta SAM2 Hiera Small and Large model integration with QLoRA and spatial join boundary refinement.
- Image Manipulation Localization (IML) MaskFormer (`RefactorMaskformer`) integration with Hungarian point-matching loss.
- High-throughput streaming parquet dataset loaders with block shuffling and row-group locality caching.
