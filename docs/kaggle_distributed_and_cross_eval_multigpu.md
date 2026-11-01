# Multi-GPU Kaggle Distributed Optimization & Cross-Evaluation Acceleration

**Target Repository**: `ai-detection-unet-gpu`  
**Branch**: `fix-kaggle-distributed`  
**Components**: `IMLMaskformer`, `Trainer` (DDP), `sid-cross-eval` (`sid_unet.cross_eval`), `sid-eval` (`sid_unet.evaluate`)  
**Hardware Environment**: Kaggle Dual Tesla T4 GPUs (CUDA SM 7.5, 2x 15GB VRAM)

---

## 1. Problem Diagnosis: `.logs.txt` Training Failure

### Symptoms & Traceback
When launching distributed multi-GPU training via:
```bash
sid-train --config configs/experiments/iml/iml_maskformer.yaml --override data.save_to_disk=true --hf-repo KhangTruong/IML
```
The auto-launching DistributedDataParallel (DDP) system spawned 2 ranks on ports 33873. Both ranks initialized, successfully resolved the SAM 2.1 Hiera Large vision backbone, and verified safe batch size 4 on rank 0. However, on the very first training batch of Epoch 1, training crashed immediately with:

```text
[2026-11-01 05:24:35] [ERROR] 🛑 Error encountered during training execution: No backend type associated with device type cpu. Closing DataLoader workers and resources...
Traceback (most recent call last):
  File "sid_unet/training/trainer.py", line 1545, in train
    train_metrics = self.train_epoch(epoch)
  File "sid_unet/training/trainer.py", line 1179, in train_epoch
    loss_t, loss_dict_batch = self._step_batch_train(batch, loss_divisor=accum_steps)
  File "sid_unet/training/trainer.py", line 1028, in _step_batch_train
    outputs = self.model(images, **model_kwargs) if model_kwargs else self.model(images)
  File "torch/nn/parallel/distributed.py", line 1695, in forward
    inputs, kwargs = self._pre_forward(*inputs, **kwargs)
  File "torch/nn/parallel/distributed.py", line 1591, in _pre_forward
    self._sync_buffers()
  File "torch/nn/parallel/distributed.py", line 2231, in _sync_buffers
    self._sync_module_buffers(authoritative_rank)
  File "torch/nn/parallel/distributed.py", line 2235, in _sync_module_buffers
    self._default_broadcast_coalesced(authoritative_rank=authoritative_rank)
  File "torch/nn/parallel/distributed.py", line 2257, in _default_broadcast_coalesced
    self._distributed_broadcast_coalesced(bufs, bucket_size, authoritative_rank)
  File "torch/nn/parallel/distributed.py", line 2172, in _distributed_broadcast_coalesced
    dist._broadcast_coalesced(
        self.process_group, tensors, buffer_size, authoritative_rank
    )
RuntimeError: No backend type associated with device type cpu
```

### Root Cause Analysis
1. **Lazy Buffer Registration on CPU during Forward Pass**:
   In `sid_unet/models/iml_maskformer.py`, `SinusoidPositionEmbeddings2D` computes 2D sinusoidal positional embeddings for multi-scale feature maps. The author deferred `_build_position_embeddings` to `forward()`:
   ```python
   def forward(self, x: Tensor, pad_mask: Optional[Tensor] = None) -> Tensor:
       if not hasattr(self, "position_embeddings"):
           pe = self._build_position_embeddings()
   ```
   Inside `_build_position_embeddings()`:
   ```python
   pos_x = torch.arange(1, w + 1, dtype=dtype)[None, None, None, :].expand(1, 1, h, w)
   pos_y = torch.arange(1, h + 1, dtype=dtype)[None, None, :, None].expand(1, 1, h, w)
   pe = self._compute_position_embeddings(pos_x, pos_y, dtype=dtype)
   self.register_buffer("position_embeddings", pe, persistent=False)
   ```
   `torch.arange` default device is `cpu`. Hence, `pe` was created on CPU and registered dynamically into `self._buffers`.

2. **DDP Asymmetry Caused by Rank-0 Memory Probe**:
   Prior to Epoch 1, `trainer.py` ran `find_optimal_batch_size` (the auto-batch-sizing probe) exclusively on Rank 0 (`if self.is_main_process:`). When Rank 0 called `raw_model(images)`, `position_embeddings` was dynamically added to Rank 0's buffers on CPU. Rank 1 did not run the probe, so `position_embeddings` did not exist on Rank 1.

3. **NCCL Buffer Sync Rejection**:
   When training started, `DistributedDataParallel.forward()` called `self._sync_buffers()` to broadcast all buffers from the authoritative rank (Rank 0) to Rank 1. Because the distributed process group was initialized with the `"nccl"` backend for CUDA GPUs, passing a CPU tensor to `dist._broadcast_coalesced` raised `RuntimeError: No backend type associated with device type cpu`.

---

## 2. Implemented Fixes

### A. Eager Buffer Registration in `SinusoidPositionEmbeddings2D`
- In `SinusoidPositionEmbeddings2D.__init__`, pre-compute and register `position_embeddings` immediately at module initialization:
  ```python
  self._build_position_embeddings()
  ```
- When `model.to(device)` is called, `position_embeddings` moves to `cuda:0` on Rank 0 and `cuda:1` on Rank 1 before DDP wrapping.
- In `forward()`, `self.register_buffer()` is strictly never called dynamically. If an input with a different spatial shape arrives, embeddings are computed dynamically on `x.device` without mutating registered module buffers.

### B. DDP Buffer Safety Guards in `Trainer`
- In `Trainer.__init__`, immediately before wrapping `self.model` in `torch.nn.parallel.DistributedDataParallel`, an explicit buffer device check ensures all buffers and parameters are on `self.device`:
  ```python
  for buf in self.model.buffers():
      if buf is not None and buf.device != self.device:
          buf.data = buf.data.to(self.device)
  for p in self.model.parameters():
      if p is not None and p.device != self.device:
          p.data = p.data.to(self.device)
  ```
- In `_probe_and_adjust_batch_size`, the `finally:` block restores `self.raw_model` onto `self.device` and verifies all buffers remain on the target GPU.

---

## 3. Kaggle Multi-GPU Cross-Evaluation Acceleration (`sid-cross-eval`)

### Problem
In Kaggle environments with dual Tesla T4 GPUs (`torch.cuda.device_count() == 2`), running `sid-cross-eval` previously placed the evaluation model on `torch.device("cuda")` (`cuda:0`). As a result:
- GPU 0 bore 100% of the evaluation workload.
- GPU 1 sat completely idle at 0% utilization and 0 MB allocated VRAM.
- Evaluation runtimes were double what the dual-T4 hardware could deliver.

### Solution Architecture
1. **Automated Kaggle Environment Detection**:
   Added `is_kaggle_environment() -> bool` in `sid_unet.utils.distributed`:
   - Checks `KAGGLE_KERNEL_RUN_TYPE`, `KAGGLE_CONTAINER_NAME`, `KAGGLE_URL_BASE`, and the `/kaggle` filesystem.
2. **Automatic Multi-Device DataParallel**:
   - In `sid_unet/cross_eval.py`, if multiple CUDA GPUs are present and the environment is Kaggle (or multi-GPU is requested), `evaluate_checkpoint_on_config` automatically wraps `model` with `torch.nn.DataParallel(model, device_ids=target_device_ids)`.
   - Batches are automatically partitioned across all available devices (e.g. 8 samples/batch -> 4 per GPU).
   - Both GPUs run forward evaluation in parallel, doubling throughput.
3. **Hardware Utilization Telemetry**:
   - Logs active GPUs and initial VRAM status before evaluation starts.
   - Measures and logs per-device peak VRAM and allocated memory (`torch.cuda.max_memory_allocated`).
   - Exports `gpu_utilization`, `num_devices`, and `devices` into the evaluation metrics dictionary and master report.
4. **User Controls via CLI**:
   - `--data-parallel`: Force enable multi-GPU DataParallel.
   - `--no-data-parallel`: Force single-GPU execution.
   - `--device-ids 0 1`: Restrict evaluation to specific GPU IDs.

---

## 4. Verification & Testing

| Test Case | Scope | Result |
|-----------|-------|--------|
| `test_sinusoid_position_embeddings_buffer_init_and_device` | Verified buffer registered at init, moves to CUDA, no dynamic registration | **PASSED** |
| `test_is_kaggle_environment` | Tested Kaggle env vars and filesystem detection | **PASSED** |
| `test_cross_eval_kaggle_multigpu_utilization` | Dual-GPU execution on 2x Tesla T4, verified balanced memory & peak VRAM | **PASSED** |
| `test_cross_eval_disable_data_parallel` | Verified `--no-data-parallel` / `data_parallel=False` single-GPU mode | **PASSED** |
| `test_cross_eval_cli_data_parallel_flags` | CLI argument parsing for `--data-parallel` and `--no-data-parallel` | **PASSED** |
| 2-Rank DDP NCCL Forward Test | Tested DDP forward pass on 2 GPUs with `torch.distributed.run` | **PASSED** (0 errors) |
| `test_iml.py` Suite (10 tests) | Full IML MaskFormer unit and integration tests | **PASSED** (10/10) |
| `test_cross_eval.py` Suite (10 tests) | Full cross-evaluation suite | **PASSED** (10/10) |
| `test_evaluation_ablation.py` Suite (4 tests) | Evaluation ablation and post-processing tests | **PASSED** (4/4) |
