# Automatic GPU-Saturating Batch Size Inference (`batch_size: -1`) & `sid-pull` Directory Resolution

## 1. Automatic Batch Size Inference (`batch_size: -1`)

### Motivation
Different architectures evaluated side-by-side in `sid-cross-eval` (e.g., `FOCAL HRNet` vs. `PIM Swin-B`) or trained across GPUs with different VRAM capacities (e.g., 8 GB RTX 3060 Ti vs. 16 GB T4 vs. 24+ GB A10/A100) have vastly different memory footprints. A hardcoded batch size either under-utilizes smaller/faster models or triggers OOM / sub-batch fallback penalties on heavier models.

Setting `data.batch_size: -1` (or `training.batch_size: -1`, or `--batch-size -1` via CLI) instructs `sid_unet` to automatically infer the largest batch size that safely saturates available GPU VRAM for each model/resolution pair.

### Usage

In YAML configs (e.g. [`configs/cross-eval/MVSS-v1-protocol.yaml`](../configs/cross-eval/MVSS-v1-protocol.yaml)):
```yaml
data:
  image_size: [512, 512]
  batch_size: -1
```

Or via CLI flags:
```bash
sid-train --config configs/train_non_streaming.yaml --batch-size -1
sid-eval --checkpoint outputs/RUN/focal_hrnet/checkpoints/checkpoint_best.pt --batch-size -1
sid-cross-eval --cross-configs configs/cross-eval/MVSS-v1-protocol.yaml --checkpoints ... --batch-size -1
```

Optional tuning knobs under `data` (or `training`):
- `max_auto_batch_size`: Upper probe bound (default: `256`).
- `cpu_fallback_batch_size`: Fallback batch size when running on CPU (default: `16`).

### Algorithm (`sid_unet.utils.memory.infer_optimal_batch_size`)
1. **Mode-Aware Execution**:
   - `mode="eval"` (used by `sid-eval` and `sid-cross-eval`): runs inside `torch.inference_mode()` + configured AMP/FP8 autocast, without allocating optimizer state or gradients, and includes optional `refiner` (SAM) memory in the probe. Targets **78%** of total GPU VRAM.
   - `mode="train"` (used by `Trainer`): runs forward + backward + grad zeroing with BatchNorm running-stat restoration, accounting for unmaterialized AdamW optimizer states (`8 bytes * trainable_params`). Targets **75%** of total GPU VRAM.
2. **Prefetcher & Host RAM Guardrails**: Reserves `(prefetch_batches + 1) * bs * (C*H*W + H*W) * 4` bytes on GPU for `CUDAPrefetcher` + active batch tensors, and bounds `max_batch_size` + `PDLDataLoader` worker/prefetch concurrency against available host RAM (`MemAvailable`) to prevent Linux OOM kills on large batch sizes.
3. **Two-Phase Search**:
   - **Phase 1 (Exponential Doubling)**: Probes `1, 2, 4, 8, ...` (or multiples of `len(device_ids)` under `DataParallel`) until OOM / `AcceleratorError` / VRAM budget threshold is crossed or `max_auto_batch_size` is reached.
   - **Phase 2 (Binary Search Refinement)**: Refines between the last passing batch size and the first failing batch size (up to 5 bounded steps).
4. **Clean Telemetry**: Resets `torch.cuda.reset_peak_memory_stats` in `finally` so probing peaks do not pollute evaluation or training peak VRAM reports.
5. **Dataloader Preservation**: `resolve_batch_size()` in `sid_unet/dataset/loader.py` records `config.data._auto_infer_batch_size = True` instead of clamping `-1` to `1`, allowing `Trainer` to probe the model on GPU and rebuild DataLoaders cleanly.

---

## 2. `sid-pull` `outputs/RUN/default` Bug Fix

### Root Cause
In `sid_unet/checkpoint_sync.py`, `pull_checkpoint()` previously executed a legacy copy block (formerly lines 879–885):
```python
legacy_run_dir = os.path.abspath(os.path.join("outputs", "RUN", "default", "checkpoints"))
if os.path.abspath(ckpt_target_dir) != legacy_run_dir:
    os.makedirs(legacy_run_dir, exist_ok=True)
    ...
```
Every `sid-pull` invocation (including `KhangTruong/FOCAL` and `KhangTruong/PIM`) copied pulled `.pt` files into `outputs/RUN/default/checkpoints/`, even when the model resolved to `outputs/RUN/focal_hrnet` or `outputs/RUN/pim_swin`. Additionally, when local experiment config paths did not match `<repo_name>/default.yaml`, directory resolution fell back to `outputs/RUN/default`.

### Fix
- Added `_resolve_repo_output_dir()`, which checks matching local `configs/experiments/<clean_name>/*.yaml` files, then remote `config.yaml` / `checkpoints/config.yaml` (`project.output_dir`), and finally falls back to `outputs/RUN/<clean_name>` (never `outputs/RUN/default`).
- Removed the unconditional copy into `outputs/RUN/default/checkpoints/`.
- Wired `repo_id` into `find_local_checkpoint_artifacts()` so `sid-push` discovers the matching `outputs/RUN/<model>` directory automatically.

---

## 3. Empirical Cross-Evaluation & GPU Telemetry (RTX 3060 Ti, 8 GB VRAM)

Cross-evaluation was executed via `sid-cross-eval` on `configs/cross-eval/MVSS-v1-protocol.yaml` (`KhangTruong/Tampered-Images-Combined-Dataset-Fake`, `cross_test`, **17,929 samples** across 15 sub-datasets, `batch_size: -1`, BF16 precision) after pulling `KhangTruong/FOCAL` and `KhangTruong/PIM` with `sid-pull` (verified no `outputs/RUN/default` created):

| Model | Auto-Inferred `batch_size` | Peak VRAM (PyTorch) | GPU-Util Mean / p50 / p95 / Max | Mean Power | IoU | Dice / F1 | Continuous AUROC | Pixel Acc |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| **`sid_focal_hrnet`** (`KhangTruong/FOCAL`) | **36** | 1,989.6 MB | 47.7% / 38% / 100% / 100% | 130.0 W | **0.2333** | **0.2944** | **0.7230** | 0.7836 |
| **`sid_pim_swin`** (`KhangTruong/PIM`) | **9** | 5,146.8 MB | **88.0% / 100% / 100% / 100%** | 176.6 W | 0.1675 | 0.2019 | **0.7176** | **0.8557** |

- Across the entire 2-model run (`n = 851` `nvidia-smi` samples at 2s intervals), overall GPU utilization averaged **78.5%** (median **100%**, p95 **100%**), with `batch_size=-1` cleanly differentiating lightweight `FOCAL` (`batch_size=36`) from memory-heavy `PIM` (`batch_size=9`) without OOM.
- Closing `SIDMapDataset` / `IndexedParquetDataset` between checkpoints and releasing unused PyArrow + glibc malloc arenas in `clear_memory_cache()` kept host RAM stable at ~5.5–6.2 GiB used (~9.3–10 GiB available, 0 B swap) on a 15 GiB RAM container.
- Updated cross-evaluation artifacts and master reports were pushed back to [`KhangTruong/FOCAL`](https://huggingface.co/KhangTruong/FOCAL) and [`KhangTruong/PIM`](https://huggingface.co/KhangTruong/PIM).

---

## 4. GPU Utilization Optimization (`FOCAL` Pipeline Bottleneck Elimination)

### Root-Cause Analysis (`47.7%` vs `88.0%`)
Per-stage profiling of `sid-cross-eval` (`512x512`, BF16, RTX 3060 Ti) revealed why `sid_focal_hrnet` (`batch_size=36`) averaged `47.7%` GPU utilization while `sid_pim_swin` (`batch_size=9`) averaged `88.0%`:
1. **Synchronous CPU OpenCV post-processing (`219.5 ms/batch`, `37.2%` of batch time on `FOCAL`)**: `MaskPostProcessor.process_batch()` converted `probs` to per-sample `float32` CPU slices, ran repeated `float32 <-> uint8` conversions inside `remove_small_components()`, `fill_mask_holes()`, and `apply_morphology()`, and blocked the main GPU thread until all CPU workers finished.
2. **Duplicate Continuous AUROC Sorting (`53.3 ms/batch`, `9.1%` on `FOCAL`)**: `post_seg_tracker.update(..., score_masks=mask_logits)` re-sorted `36 x 262,144` pixels via `compute_binary_auroc_batch()` immediately after `raw_seg_tracker.update()` had computed the exact same threshold-free per-sample ROC-AUC values.
3. **Remote HF Parquet DataLoader Stalls (`~15%` + cold start)**: Streaming/cached Parquet shard boundaries and JPEG/PNG decodes under `num_workers=4, prefetch_factor=2` (bound by the 15 GiB host-RAM guardrail) stall fast forward passes between shards.

### Implemented Optimizations
- **Single-Transfer Pure-`uint8` Morphology (`sid_unet/postprocessing.py`)**: Extracted `_to_binary_u8`, `_remove_small_components_u8`, `_fill_mask_holes_u8`, and `_apply_morphology_u8` operating end-to-end on `uint8 {0, 1}` arrays with an empty-mask fast path; `MaskPostProcessor.process_batch()` now thresholds on the tensor's native device and performs a single contiguous `uint8` D2H/H2D transfer.
- **Overlapped Asynchronous Post-Processing Pipeline (`AsyncPostProcessPipeline`)**: Runs a bounded background queue (`max_queue_size=4`) on CPU that overlaps `MaskPostProcessor.process_batch_binary_np()`, `post_seg_tracker.update()`, and visual illustration collection (`collect_samples`) concurrently with GPU forward passes in `sid_unet/cross_eval.py` and `sid_unet/evaluate.py` (when running on CUDA without SAM3 refinement), with clean thread flushing and exception propagation via `finish()`.
- **Continuous AUROC Reuse (`precomputed_auroc` in `sid_unet/metrics/segmentation.py`)**: Added `precomputed_auroc` to `compute_binary_metrics_batch()` and `SegmentationMetricTracker.update()`, returning `batch_metrics["auroc"]` from `update()` and forwarding `raw_auroc` (and `sam_auroc`) across downstream `post_seg_tracker`, `sam_seg_tracker`, and `both_seg_tracker` without re-sorting pixels.

### Measured RTX 3060 Ti Benchmark Comparison (`FOCAL HRNet`, `512x512`, BF16, `bs=36`)
| Benchmark | Baseline (Sync + Duplicate AUROC) | Optimized (`AsyncPostProcessPipeline` + `precomputed_auroc`) | Improvement |
|---|---:|---:|---:|
| **Steady-state batch compute (`bs=36`, 10 batches)** | `584.0 ms/batch` | **`410.6 ms/batch`** | **`1.42x` speedup (`-29.7%` latency)** |
| **Live HF 2,000-sample `MVSS-v1-protocol` (58 batches incl. startup)** | — | **`45.65 s` total (`51.3%` active GPU-Util mean / `48%` p50)** | Active GPU-Util reflects remaining HF remote Parquet HTTP/decode I/O wait |
| **Container Host RAM Tunability Check** | `num_workers=4, prefetch=2` | Tested `6 / 3`: `45.84 s`, `45.5%` active util, `6.00 GiB` min avail RAM | Retained `4 / 2` bound for better runtime + ~3 GiB higher RAM headroom |
