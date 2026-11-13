# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-11-12

### Added
- **TUI Process Manager (`sid-tui` & `sid-top`)**:
  - Implemented a complete Terminal User Interface (`SIDTerminalUI`) built on standard Python `curses`, replacing static `tail -f .logs.txt` monitoring.
  - Interactive split-screen dashboard displaying real-time process list (PID, Command, Status, Elapsed Duration, RSS RAM) alongside live streaming log tail.
  - Real-time auto-follow mode (`f` key) for continuous streaming output of active processes.
  - Fullscreen log view (`Enter` or `l` key) with scrollback navigation (`Up`, `Down`, `PageUp`, `PageDown`, `Home`, `End`).
  - Terminated process lifecycle tracking: tracks `RUNNING`, `COMPLETED`, `KILLED`, and `EXCEPTION` states.
  - Exception & Traceback Inspector (`e` key): automatically scans log files from initial spawn offset, extracts Python unhandled tracebacks and error messages, and presents an interactive traceback view.
  - Process termination controls: kill targeted process (`k` key) with confirmation and graceful SIGINT/SIGTERM escalation to SIGKILL; kill all active background processes (`K` key).
  - Status filter cycling (`Tab` key): toggle between `ALL`, `RUNNING`, `EXCEPTION`, and `TERMINATED` views.
  - Terminated history cleanup (`c` key) to clear finished records from the registry.
  - CLI and non-interactive headless support: `sid-tui --list`, `sid-tui --status`, `sid-tui --logs <pid>`, `sid-tui --tail <pid>`, `sid-tui --kill <pid>`, `sid-tui --clear-history`, and `--no-tui` fallback for non-TTY environments.
- **Process Registry & Tracking Subsystem (`sid_unet.process_manager`)**:
  - Designed `ProcessRecord` dataclass tracking process metadata, arguments, status, exit codes, start offsets, and failure summaries.
  - Designed `ProcessRegistry` with thread-safe and process-safe file locking via `filelock.FileLock` on `.sid_processes/registry.lock`.
  - Implemented `inspect_log_for_termination` scanning process log streams for clean completion banners, SIGINT/KeyboardInterrupt, unhandled Python exceptions, and abnormal terminations.
  - Integrated automatic process registration in `sid-train`, `sid-cross-eval`, and `sid-cache` upon background detachment (`--detach`), storing process startup markers and printing navigation tips for `sid-tui`.
  - Integrated `sid-kill` process termination state synchronization with `ProcessRegistry`.
- **Focal & PIM Cross-Evaluation Mismatch Audit (`docs/cross_evaluation_focal_pim_mismatch_investigation.md`)**:
  - Investigated root causes of score mismatches on `master_cross_evaluation_report.md` between reported cross-eval scores and cataloged scores for `sid_focal_hrnet` and `sid_pim`.
  - Proved FOCAL inference head saturation: untrained 2D convolution head in `sid_unet/models/focal.py` produces unconditionally positive differences (`diff > 0.12`), driving sigmoid to 1.0 across 100% of pixels and explaining 0.9969 recall with 0.0069 specificity.
  - Proved input resolution and normalization mismatches: FOCAL expects $[0, 1]$ unnormalized inputs at $1024 \times 1024$; PIM expects $512 \times 512$ inputs for Swin window partitioning ($7 \times 7$), while cross-eval downscaled to $256 \times 256$ with ImageNet normalization $[-2.1, 2.6]$.
  - Proved evaluation metric discrepancies: FOCAL upstream evaluates macro-averaged F1 (including background pristine F1), while `sid_unet` evaluates binary foreground Dice; PIM upstream rewards pristine images with 1.0 IoU/F1 when clean, while `sid_unet` excludes zero-positive images (`filter_zero_positive=True`).
  - Identified catalog score semantic confusion: PIM `best_score: 0.852` represents AutoSplice 90 AUC (not IoU, not CASIA/MVSS); FOCAL `best_score: 0.824` represents validation Macro F1 mislabeled under `"val_iou"`.
  - Documented dataset domain shift (>55% generative diffusion images in `eval_mvss_v1_protocol` vs traditional splicing pretraining) and ground truth polarity inversion in NIST16.
  - Implemented automated empirical evidence test suite (`tests/test_focal_pim_mismatch_audit.py`) validating mathematical proofs and empirical bounds for all 5 identified mismatch root causes.
- **FOCAL Model Integration (HRNet & SAM ViT Backbones)**:
  - Integrated the complete FOCAL framework (Forensic Contrastive Clustering for Image Forgery Detection and Localization, ICCV 2023) from official repository `HighwayWu/FOCAL`.
  - Implemented self-contained `HRNet-W32` and `SAM ViT-Large` backbones with pure PyTorch `SAMBlock`, `MLPBlock`, and decomposed relative position embeddings.
  - Strict zero-fallback policy: `pretrained=False` raises an immediate `ValueError`; missing or failed weight loading raises an immediate `RuntimeError`. No dummy or fallback weights are permitted.
  - Hardcoded official pretrained weights from Hugging Face Hub `KhangTruong/FOCAL` (`FOCAL_HRNet_weights.pth` and `FOCAL_ViT_weights.pth`).
  - Added on-the-fly unsupervised k-means clustering mask prediction (`predict_mask_clustering`) and binary prediction helper (`predict_mask`).
  - Added experiment configuration files `configs/experiments/focal/focal_hrnet.yaml` and `configs/experiments/focal/focal_vit.yaml`.
  - Added test suite `tests/test_focal.py` verifying anti-fallback enforcement, genuine weight loading, forward and backward passes, output shapes and bounds, and `build_model` integration.
- **PIM Model Integration (Dual-Branch Swin Transformer & UPerHead)**:
  - Integrated the PIM architecture (Plug-and-Play Image Inpainting Localization Network, ICCV 2023) from official repository `ChenqiKONG/PIM`.
  - Implemented self-contained PyTorch `SwinTransformer` (visual feature stream) and `SwinTransformerMask` (causal-masked reconstruction feature stream).
  - Implemented frequency-sensitive local convolution enhancement (`FeatureLocalConv`) with device-agnostic buffer allocation and dynamic `FeatureFusion`.
  - Implemented multi-task `UPerHead` dense decoders producing forgery mask segmentation, edge boundary localization, and image reconstruction.
  - Strict zero-fallback policy: `pretrained=False` raises an immediate `ValueError`; missing or failed weight loading raises an immediate `RuntimeError`.
  - Hardcoded official pretrained weights from Hugging Face Hub `KhangTruong/PIM` (`ckpt.pth`).
  - Added experiment configuration file `configs/experiments/pim/pim.yaml`.
  - Added test suite `tests/test_pim.py` verifying anti-fallback enforcement, genuine weight loading, multi-task output heads, forward/backward passes, and `build_model` integration.
- **Core Architecture Wiring for FOCAL and PIM**:
  - Exported `FOCAL`, `FOCALHRNet`, `FOCALViT`, `PIM` and their default Hugging Face repository constants in `sid_unet.models`.
  - Wired `focal` and `pim` branches into `build_model` in `sid_unet.models.unet`.
  - Added state_dict key signature pattern matching for `focal` and `pim` checkpoints in `UNet.from_checkpoint`.
- **Hugging Face Hub Checkpoint Formatting & Cross-Evaluation**:
  - Reformatted Hugging Face repositories `KhangTruong/PIM` and `KhangTruong/FOCAL` to standard SID-UNet multi-version layouts containing `checkpoints/checkpoint_best.pt`, `checkpoints/checkpoint_latest.pt`, `config.yaml`, `manifest.json`, and versioned trees under `versions/v1/`, while preserving original `.pth` weights for backwards compatibility.
  - Upgraded weight loading modules in `FOCAL` and `PIM` to automatically unpack embedded `model_state_dict` dictionaries, handle `net.` and `module.` prefix variations, and resolve candidate checkpoint paths on Hugging Face Hub.
  - Upgraded `UNet.from_checkpoint` to auto-detect and instantiate `focal` (HRNet or ViT) and `pim` models directly from Hugging Face Hub IDs (`KhangTruong/FOCAL`, `KhangTruong/PIM`) or local checkpoint paths.
  - Upgraded `sid-pull` with automated experiment directory resolution matching `configs/experiments/<model>/<model>.yaml` destinations.
  - Verified end-to-end `sid-pull` and `sid-cross-eval` matrix evaluation across CASIA v2.0 benchmarks and synchronized master cross-evaluation reports and illustrations to remote model cards.

### Fixed
- **TUI TQDM Multi-Line Carriage Return Spreading (`sid-tui` & CLI logs)**:
  - Resolved issue where `tqdm` progress bar updates occupied a new line for every tick in `sid-tui` and CLI `--logs`, generating thousands of duplicate progress lines.
  - Implemented `collapse_line_carriage_returns()` in `sid_unet/tui.py` to accurately simulate terminal cursor carriage return (`\r`) overwriting within a single physical row.
  - Updated `read_process_logs()` to read with `newline=""`, preserving native `\r` and `\n` line separations and collapsing progress bar states to their latest visible representation.
  - Updated `print_process_logs(tail=True)` with unbuffered `newline=""` streaming to prevent carriage return translation in live follow mode.
  - Added automated unit test `test_read_process_logs_tqdm_carriage_returns` in `tests/test_tui_and_process_manager.py`.
- **Cross-Evaluation Unstable GPU Utilization & Pipeline Stalls**:
  - Diagnosed root causes of erratic, spiky GPU utilization (0% to 100%, 10% average) during cross-evaluation benchmarks (e.g. `MVSS-v1-protocol.yaml`). Confirmed adjusting batch size alone does not resolve the issue due to pipeline synchronization barriers and data starvation:
    1. **FOCAL Clustering Head CPU-GPU Synchronization Elimination**: Replaced the sequential, unvectorized per-image loop in `FOCAL._cluster_features()` with a fully vectorized, batched tensor implementation (`torch.bmm`, `torch.gather`, `torch.where`). Eliminated 136 synchronous CPU-GPU barriers (`cudaStreamSynchronize`) per batch, delivering an 11.72x speedup (from 90.39ms to 7.72ms per batch) while retaining numerical equivalence (error $< 3\times 10^{-6}$).
    2. **DataLoader I/O Starvation & Row-Group Boundary Buffering**: Upgraded `create_eval_dataloader()` in `sid_unet/dataset/loader.py` to forward `prefetch_factor = max(4, ...)` to `PDLDataLoader` and PyTorch `DataLoader`, and increased `max_cached_row_groups = max(4, ...)` in `SIDMapDataset`. Expanded batch prefetch capacity from 8 to 16-32 batches, preventing queue drainage and severe I/O stalls during Parquet boundary decompression.
    3. **Mask Post-Processor Thread Pool Churn**: Replaced per-batch `concurrent.futures.ThreadPoolExecutor` allocation and teardown in `MaskPostProcessor.process_batch()` with a persistent, pooled executor, eliminating OS thread creation overhead across thousands of evaluation batches.
- **Noise-RAG Vector Database No-Grad & Detached Alignment Loss**:
  - Enforced `requires_grad = False` and explicit `.detach()` on vector database keys (`db_keys`) and continuous labels (`db_labels`) in `ForensicVectorDBRetrieval.load_database`.
  - Enforced `target = target.detach().requires_grad_(False)` in `compute_alignment_loss` before computing `F.mse_loss`, ensuring surrogate representations and visual tokens do not backpropagate gradients into the frozen retrieved priors.
  - Added unit test `test_noise_rag_database_buffers_and_labels_no_grad` in `tests/test_noise_rag.py`.
- **Noise-RAG VRAM Hoarding & CUDA Out-Of-Memory Crash Elimination**:
  - Diagnosed and resolved massive VRAM hoarding in `NoiseRAGPrototypeV1` when retrieving from 1,000,000-vector databases (`mask_clip_db_mae.pt`):
    1. **PyTorch Attention Matrix Materialization**: PyTorch `nn.MultiheadAttention` defaults to `need_weights=True`, materializing a `[2, 8, 4096, 4096]` float32 attention matrix per block (~1.27 GB each, totaling ~9 GB across 7 attention blocks on 512x512 inputs). Passing `need_weights=False` enables native FlashAttention/SDPA, reducing memory to <100 MB.
    2. **Autograd Graph Tracking in Retrieval**: Wrapped database retrieval inside `with torch.no_grad():` and detached queries (`q.detach()`), preventing autograd from maintaining a 4.1 GB intermediate similarity graph across millions of keys.
    3. **Query Chunk Size Optimization**: Reduced retrieval chunk size from 2048 to 256, capping maximum temporary similarity matrix allocation to under 256 MB per chunk and reducing peak end-to-end forward+backward VRAM from >16 GB (CUDA OOM) to ~3.5 GB on 1M-entry databases.
    4. **Key Normalization Buffer**: Pre-computed and registered `db_keys_norm` buffer during database loading to eliminate redundant matrix normalizations during forward passes.
  - Added unit test `test_noise_rag_bounded_vram_and_attention_efficiency` in `tests/test_noise_rag.py`.
  - Updated assertion in `test_build_model_from_configs` to accommodate 1,000,000 target entries in `configs/experiments/mask_clip/mask-clip-db-mae.yaml`.

## [Unreleased] - 2026-11-11

### Added
- **XS-to-XR MSE Alignment Loss in `NoiseRAGPrototypeV1`**:
  - Implemented explicit MSE alignment loss enforcing surrogate representations (XS) and visual encoder tokens (XT) to align with retrieved forensic memory priors (XR) via detached targets (`XR.detach()`).
  - Added `"align_loss"` and `"xs_xr_loss"` to `NoiseRAGPrototypeV1` output dictionary and wired loss extraction into `SIDTotalLoss.forward()` and `build_loss()`.
  - Added `test_xs_to_xr_mse_loss` in `tests/test_noise_rag.py` verifying gradient propagation to visual encoder tokens and surrogate representations.
- **Vector Database Auto-Routing in `sid-train` (`--cached-hf-repo`)**:
  - Automatically detect when `--cached-hf-repo` points to a vector database repository (e.g. `KhangTruong/mask-clip-db-mae-cache`) or when training retrieval models like `noise-rag-prototype-v1`.
  - Resolve and download the vector database directly into `config.model.memory_path` via Hugging Face Hub (`hf_hub_download`), preventing accidental assignment to `config.data.cached_hf_repo` which previously generated 84-channel mock latent shards and triggered input shape mismatch crashes (`RuntimeError: Given groups=1, expected input to have 3 channels, but got 84 channels instead`).
  - Added CLI options `--memory-path`, `--cached-vector-db`, and `--vector-db` to `sid-train` for explicit vector DB path specification.
- **Chunked Token Querying in `ForensicVectorDBRetrieval`**:
  - Implemented token query chunking (default chunk size 2048) in `ForensicVectorDBRetrieval.forward`, avoiding peak VRAM spikes and out-of-memory errors when computing similarity against 100k vector database keys on dense feature maps ($H \times W = 4096$ tokens).
- **Hugging Face Hub Resolution & Multi-Path Fallback in `ForensicVectorDBRetrieval`**:
  - Enhanced `ForensicVectorDBRetrieval.load_database` to directly download vector database `.pt` files from Hugging Face Hub dataset repos (`org/repo_name`) and search local candidate locations (`outputs/cache/`, `outputs/dataset_cache/`).
- **Configurable Cross-Attention Mode in `NoiseRAGPrototypeV1`**:
  - Added configurable `cross_mode` ("recurrent" / "chained" vs "xs" / "surrogate") in `NoiseRAGPrototypeV1` and exposed it via `sid_unet.models.unet.build_model`.
- **Dictionary Model Output Compatibility Across Training, Evaluation, and Prediction**:
  - Handled dictionary model outputs (`outputs.get("mask_logits")`, `outputs.get("class_logits")`) in `Trainer._step_batch_train`, `Trainer._eval_batch_step`, `Trainer._check_and_auto_scale_batch_size`, `evaluate_single_checkpoint`, and `predict_image`, preventing `AttributeError: 'dict' object has no attribute 'detach'` / `'dim'`.
- **Validation in `create_cached_dataloaders`**:
  - Added explicit validation to raise an informative `ValueError` if `create_cached_dataloaders` receives a vector DB dataset repo containing `.pt` files without parquet latent shards, and set `default_channels` to `in_channels` for `noise-rag`.
- **Mask Byte Decoding Utility (`decode_mask_bytes`)**:
  - Implemented `decode_mask_bytes` in `sid_unet.dataset.mask_utils` (and exported in `sid_unet.dataset`) to decode raw mask image bytes (PNG, JPEG), bytearrays, BytesIO streams, HF dictionary structures, PIL Images, and numpy/PyTorch tensors into normalized float32 arrays with optional resizing and binarization thresholding.
- **Hugging Face Hub Vector Database Sync in `sid-cache`**:
  - Added synchronous vector database upload to Hugging Face Hub upon cache completion in `sid_unet.cache.cli` when `--cached-hf-repo` / `--push-to-hub` is provided, along with clean manager termination.
- **Vector DB Auto-Restoration in `UNet.from_checkpoint`**:
  - Added automatic detection and restoration of sibling `mask_clip_db_mae.pt` / `vector_db.pt` into the model's `retrieval` module in `sid_unet.models.unet.from_checkpoint`.
- **Extended Test Suite for Noise-RAG and Vector DB Caching**:
  - Added unit tests for `decode_mask_bytes` in `tests/test_mask_utils.py` covering PNG decoding, resizing, continuous vs thresholded decoding, and edge cases.
  - Added comprehensive integration tests in `tests/test_noise_rag.py` (22 tests) verifying vector DB extraction with encoded PNG mask bytes, `ForensicVectorMemory` continuous patch label storage and consolidation, `ForensicVectorDBRetrieval` fallback and loaded modes, chunked token querying, HF Hub download resolution, `UNet.from_checkpoint` roundtrip restoration, combined `mse+entropy` alignment loss gradient flow, full `MaskCLIP.extract_pre_tvca_features` tap, configurable cross-attention modes, `sid-train` vector DB routing, rejection of vector DB in `create_cached_dataloaders`, `sid-eval` and `sid-cross-eval` with dict outputs, and `sid-cache` vector DB CLI execution.

### Fixed
- **Learning Rate Parsing Bug in `Trainer`**:
  - Fixed `Trainer.__init__` and `Trainer.resume_from_checkpoint` which only checked `config.training.get("learning_rate")`, silently ignoring `config.training.lr` and falling back to default `0.001` (10x higher than configured `lr: 0.0001`). Both `learning_rate` and `lr` keys are now checked with proper precedence.
- **Model Collapse & Class Imbalance in `noise_rag_prototype_v1.yaml`**:
  - Replaced unweighted `"combined"` loss with `"bce_dice"` (`pos_weight: 3.0`, `bce_weight: 0.2`, `dice_weight: 0.8`), resolving prediction collapse where the model predicted all-negative logits (IoU = 0.0000) on datasets like CASIA v2.0 with sparse tampered regions (< 5% of pixels).
  - Scaled base batch size to 2 with `gradient_accumulation_steps: 2` to eliminate initial CUDA OOM candidate probing on 22 GB GPUs.
- **`RuntimeError` Input Channel Mismatch When Passing `--cached-hf-repo` to `sid-train`**:
  - Fixed bug in `.logs.txt` where passing `--cached-hf-repo KhangTruong/mask-clip-db-mae-cache` caused `create_cached_dataloaders` to synthesize 84-channel mock latent shards for `NoisePrint++` (17-layer DnCNN expecting 3-channel RGB). The vector DB repo is now cleanly routed to download into `config.model.memory_path` while retaining standard RGB image loading.
- **`UnboundLocalError` on `logger` in `train_single_run`**:
  - Initialized module logger at entry of `train_single_run` in `sid_unet/train.py`.
- **`AttributeError: 'dict' object has no attribute 'detach'` in Trainer Evaluation**:
  - Extracted tensor logits from dict outputs during training and validation steps in `sid_unet/training/trainer.py`.
- **`sid-cache` Vector DB Mask Decoding Import Error**:
  - Resolved `ImportError: cannot import name 'decode_mask_bytes'` when executing `sid-cache` with `mask-clip-db-mae.yaml`.
- **MaskCLIP Experiment Config Validation**:
  - Added `loss` configuration section to `configs/experiments/mask_clip/mask-clip-db-mae.yaml`, resolving `AttributeError: Configuration has no attribute 'edge_weight'` in `test_all_mask_clip_yaml_configs`.

## [0.1.4] - 2026-11-06

### Added
- **Vectorized GPU Evaluation Metrics & AUROC Calculation**:
  - Implemented `compute_binary_metrics_batch` and `compute_binary_auroc_batch` in `sid_unet.metrics.segmentation` computing IoU, Dice/F1, AUROC (via vectorized sort and mid-ranks), Accuracy, Precision, Recall, and Specificity in parallel on GPU tensor memory, eliminating CPU detachment bottlenecks and GPU idle stalls.
  - Refactored `SegmentationMetricTracker.update` to use batch-parallel metric computation with vectorized per-label and per-dataset statistics aggregation.
- **Asynchronous CUDA Memory Prefetcher (`CUDAPrefetcher`)**:
  - Implemented `CUDAPrefetcher` in `sid_unet.dataset.loader` to stream pinned tensor batches from host memory to GPU on a dedicated asynchronous CUDA stream, overlapping data transfer with forward model execution across validation, evaluation, and cross-evaluation loops.
- **Multi-Threaded Mask Post-Processing**:
  - Accelerated `MaskPostProcessor.process_batch` with `ThreadPoolExecutor` parallelization for morphological operations across CPU threads, and implemented a tensor binarization fast path on GPU when post-processing is disabled.
- **Master Cross-Evaluation Upload Support in `sid-push`**:
  - Enhanced `find_local_checkpoint_artifacts` in `sid_unet.checkpoint_sync` to automatically discover master cross-evaluation reports, JSON matrices, logs, and illustrations (`outputs/cross_evaluation/*`, `cross_eval_reports/*`).
  - Updated `push_checkpoint` to upload master cross-evaluation artifacts to Hugging Face Hub under both active root `cross_evaluation/` and versioned `versions/{version}/cross_evaluation/`.
  - Updated `generate_model_card_content` to render master evaluation matrices and updated repository layout in the generated `README.md`.

### Fixed
- **Validation & Cross-Eval GPU Utilization Fluctuations**:
  - Resolved GPU utilization drops to 0% during validation and cross-evaluation by vectorizing metric calculation on device, wrapping evaluation loaders with `CUDAPrefetcher`, and running inference under `torch.inference_mode()` with mixed precision (`torch.amp.autocast`).
- **Bottleneck Warning Timing During Initial Training Warmup**:
  - Updated `Trainer.train_epoch` to only record steps in `BottleneckDetector` after step 2 of the first epoch (`step_in_epoch > 2` when `epoch == start_epoch`), preventing false acute data stall warnings during initial worker startup and PyArrow cache initialization.
- **Periodic Checkpoint Timer Epoch Reset**:
  - Added `reset_periodic_timer()` to `CheckpointManager` and invoked it at the start of each epoch in `Trainer.train()`, ensuring periodic save intervals reset cleanly at epoch boundaries and preventing timer leakage from previous epochs.
- **Probing Safe Batch Size Failure (Detective SAM)**:
  - Expanded `is_oom_error` to catch `torch.AcceleratorError` and CUDA driver/device constraint errors (`invalid argument`, `too many resources requested for launch`, `cuda error`).
  - Enabled graceful fallback in `find_optimal_batch_size` when large candidate batch sizes trigger accelerator/kernel exceptions.
  - Added batch chunking (`max_chunk = 16`) to `DetectiveSAM._extract_sam2_features` to protect SAM2 Hiera vision encoder against SDPA attention dimension overflow on large candidate batches.

## [0.1.3] - 2026-11-05

### Added
- **Per-Dataset Breakdown on Multi-Benchmark Datasets (`--dataset-filter`)**:
  - Implemented multi-benchmark slicing in `sid_unet.metrics.segmentation.SegmentationMetricTracker` via `compute_per_dataset()` and `per_dataset_metrics` property, accumulating IoU, Dice/F1, AUROC, Pixel Accuracy, Precision, Recall, Specificity, and sample count for each constituent sub-dataset.
  - Forwarded `dataset` and `method` metadata from `process_raw_sample` through batch collation and into metric trackers across `sid_unet.cross_eval` and `sid_unet.evaluate`.
  - Added dedicated **Per-Dataset Breakdown (Multi-Benchmark Slicing)** tables to Markdown and JSON evaluation reports (`generate_checkpoint_cross_eval_report`, `generate_master_cross_evaluation_report`, and `generate_evaluation_report`).
  - Added optional `--dataset-filter` / `--sub-dataset` CLI argument to isolate and evaluate specific sub-datasets (e.g. `--dataset-filter CASIA_V2` or `--dataset-filter CocoGlide`).
- **MVSS-v1 Cross-Testing Protocol Configuration (`MVSS-v1-protocol.yaml`)**:
  - Added `MVSS-v1-protocol.yaml` in `configs/cross-eval/` targeting the multi-generator tampering benchmark `KhangTruong/Tampered-Images-Combined-Dataset-Fake`.
  - Configures `eval_split: "cross_test"` with standard resolution `[256, 256]`, combined BCE + Dice loss, and morphological post-processing.
  - Added flexible alias path expansion in `sid_unet.cross_eval.expand_config_patterns` resolving bare filenames and mapping legacy `cross-test/` paths to `configs/cross-eval/`.
- **Binary Fake/Tampered Dataset Mask Extraction**:
  - Implemented `is_mask_provided()` helper in `sid_unet.dataset.mask_utils` and updated `process_sample_mask()` to preserve localized ground truth tampering masks when `label == 1` and a mask is present.
  - Preserves backwards compatibility for 3-class `saberzl/SID_Set` fully synthetic images where `mask_input` is `None` (synthesizing an all-ones mask).
  - Updated `process_raw_sample()` in `sid_unet.dataset.loader` to map binary fake samples with partial masks ($0 < \text{mask\_sum} < \text{total\_pixels}$) to label `2` (tampered) in the 3-class classification taxonomy.
  - Updated `sid_unet.cache.manager` fast-path to avoid replacing localized tampering masks with blank masks when `raw_mask` is provided on label 1.
  - Added comprehensive test coverage in `tests/test_metrics.py`, `tests/test_mask_utils.py`, and `tests/test_cross_eval.py`.

### Removed
- **Removed Deprecated `cross-test/` Folder**:
  - Cleaned up temporary `cross-test/` and `configs/cross-test/` folders in favor of standard repository path `configs/cross-eval/MVSS-v1-protocol.yaml`.
- **Configurable DataLoader Batch Timeout (`timeout` / `dataloader_timeout`)**:
  - Propagated `data.timeout` and `data.dataloader_timeout` (defaulting to 300.0s) to `PDLDataLoader` across `create_dataloaders()` (for both `train_loader` and `val_loader`) and `create_eval_dataloader()`.
  - Prevents premature 60.0s `TimeoutError` during initial cold-cache Parquet downloads and heavy remote streaming pipelines.
  - Added unit test coverage in `tests/test_parquet_loader_integration.py` (`test_pdl_dataloader_timeout_configuration`).

## [0.1.2] - 2026-11-02

### Added
- **Background Detachment for `sid-cache` and `sid-cross-eval`**:
  - Implemented terminal detachment (`--detach`, default `True`) for both `sid-cache` and `sid-cross-eval`, matching the detachment behavior of `sid-train`.
  - Spawns background worker process logging standard output and standard error continuously to `.logs.txt` (or custom `--log-file`).
  - Added foreground attached execution via `--no-detach`, `--foreground`, and `-f`.
  - Added comprehensive test coverage in `tests/test_detach.py` for detachment lifecycle, child arguments, and log headers across all CLIs.
- **Default `--delete-on-update` for `sid-cache`**:
  - Made `--delete-on-update` (and aliases `--delete-local-on-upload`, `--delete-on-upload`) enabled by default (`True`) in `sid-cache` and `DatasetCacheManager` / `AsyncHubUploader`.
  - Automatically deletes local Parquet shards once uploaded to Hugging Face Hub to conserve local disk space.
  - Added `--no-delete-on-update`, `--no-delete-local-on-upload`, and `--keep-local` flags to preserve local shard files.

### Changed
- **Experiment Configs Migration from `COCO-inpainted` to `DEAL300K-30K`**:
  - Switched all 37 experiment configuration files in `configs/experiments/` from `KhangTruong/COCO-inpainted` to `KhangTruong/DEAL300K-30K`.
  - Updated dataset references and descriptions in `configs/experiments/README.md` and related architecture documentation.
  - Updated `tests/test_config.py` to validate `KhangTruong/DEAL300K-30K` as an approved dataset.
- **Cross-Evaluation Config Renaming (`base.yaml` -> `coco-inpainted.yaml`)**:
  - Renamed `configs/cross-eval/base.yaml` to `configs/cross-eval/coco-inpainted.yaml` to explicitly reflect its target evaluation benchmark (`KhangTruong/COCO-inpainted`).
  - Updated project identifier from `eval_base` to `eval_coco_inpainted`.
  - Added transparent backwards compatibility in `sid_unet.utils.config.load_config` and `sid_unet.cross_eval.expand_config_patterns` to resolve `configs/cross-eval/base.yaml` seamlessly to `coco-inpainted.yaml`.

## [0.1.0] - 2026-11-01

### Fixed
- **Multi-GPU Distributed Training Hang & SAM 2 Download Race Condition (`.logs.txt`)**:
  - Fixed the root cause of the indefinite hang following the final line in `.logs.txt`: concurrent DDP ranks raced to clone the SAM 2 repository and download the ~900MB SAM 2 weights directly into identical file paths without locks, atomic renaming, or rank barriers. Rank 1 encountered incomplete/corrupted weights and stalled/crashed, causing Rank 0 to hang indefinitely waiting for collective NCCL broadcast in `DistributedDataParallel(...)`.
  - Implemented cross-process file locks via `filelock.FileLock` and unique per-process temporary cloning paths (`/tmp/sam2_clone_{pid}_{timestamp}`) in `_ensure_sam2_available()`.
  - Implemented atomic download with file locking, size validation (>10MB), and `dist.barrier()` synchronization across ranks in `_download_checkpoint()`.
  - Updated device resolution in `SAM2ImageEncoder.__init__` to properly resolve to `cuda:{get_local_rank()}` in distributed modes, ensuring each rank initializes on its dedicated GPU device.
- **Multi-GPU Cross-Evaluation (`sid-cross-eval`) Robustness**:
  - Added parameter and buffer device reconciliation in `sid_unet/cross_eval.py` prior to wrapping models in `torch.nn.DataParallel`.
  - Added small-batch handling in `eval_single_batch()`: automatically falls back to single-device evaluation (`model.module`) whenever `batch_size < num_gpus` to prevent empty micro-batch tensor dimension errors on secondary GPUs.
  - Filtered non-tensor kwargs (e.g. Python string lists such as `img_ids`) when invoking `DataParallel`, preventing scatter type errors across GPUs.
  - Added automatic model name detection for `iml` (`IMLMaskformer`) in `UNet.from_checkpoint()` based on `pixel_module.` and `transformer_decoder.` keys in `state_dict`.
  - Added test suite in `tests/test_kaggle_multigpu_and_buffer_fixes.py` verifying multi-GPU `DataParallel` cross-evaluation, small-batch fallback, non-tensor kwarg handling, and concurrent download safety.
- **DistributedDataParallel CPU Buffer Synchronization Failure (`.logs.txt`)**:
  - Fixed `RuntimeError: No backend type associated with device type cpu` in `sid_unet/models/iml_maskformer.py` occurring during DDP buffer synchronization over the NCCL backend.
  - Resolved root cause where `SinusoidPositionEmbeddings2D` lazily generated positional embeddings on CPU during the forward pass and called `register_buffer()` dynamically on Rank 0 during memory batch size probing, causing buffer registry asymmetry between ranks.
  - Eagerly pre-computes and registers `position_embeddings` in `__init__`, allowing proper recursive movement to CUDA (`cuda:0`, `cuda:1`) upon model initialization.
  - Prohibited dynamic buffer registration during `forward()`; dynamic spatial shapes are computed purely functional on `x.device` without modifying module `_buffers`.
  - Added pre-DDP buffer device reconciliation in `sid_unet/training/trainer.py` to ensure all registered buffers and parameters reside strictly on `self.device` prior to DDP wrapping and after batch size probing.

- **Kaggle Multi-GPU Acceleration for Cross-Evaluation (`sid-cross-eval`)**:
  - Upgraded `sid-cross-eval` (`sid_unet/cross_eval.py`) to automatically detect Kaggle multi-GPU runtime environments via `is_kaggle_environment()`.
  - Enables `torch.nn.DataParallel` across all available GPUs (e.g. 2x Tesla T4) by default on Kaggle, partitioning batches evenly across devices and doubling evaluation throughput.
  - Added CLI flags `--data-parallel` / `--no-data-parallel` and `--device-ids` to give users full control over device selection.
  - Added real-time multi-device hardware utilization tracking, reporting per-GPU peak VRAM and allocated memory in evaluation summaries and master reports.
  - Extended multi-device support to single checkpoint evaluation (`sid_unet/evaluate.py`).

### Fixed
- **IML MaskFormer Official SAM 2.1 Hiera Vision Backbone Integration**:
  - Resolved silent fallback to dummy weights (`Could not load official SAM2 trunk. Falling back to lightweight dummy trunk`) in `sid_unet/models/iml_maskformer.py`.
  - Implemented `_ensure_sam2_available(repo_dir)` to automatically detect Meta's SAM 2 repository/package or clone it into target directories without colliding with pre-existing checkpoint files.
  - Enforced strict loading when `use_dummy=False`: silent dummy fallback is completely forbidden, raising an explicit `RuntimeError` if official weights cannot be instantiated.
  - Forwarded `sam2_repo`, `config_path`, `checkpoint_path`, and `device` parameters from `IMLMaskformer` through `PixelModule` and `PixelEncoder` down to `SAM2ImageEncoder`.
  - Added input device alignment in `SAM2ImageEncoder.forward` to ensure incoming image tensors match vision trunk parameters across mixed-precision and distributed settings.
  - Added `hydra-core>=1.3.2`, `omegaconf>=2.3.0`, and `iopath>=0.1.10` to `pyproject.toml` base dependencies, `iml` optional dependency group, and `all`.
  - Added unit and regression tests in `tests/test_iml.py` covering real trunk instantiation, parameter forwarding, and strict dummy fallback prevention.
  - Added technical architecture and integration note in `docs/iml_maskformer_integration.md`.
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
