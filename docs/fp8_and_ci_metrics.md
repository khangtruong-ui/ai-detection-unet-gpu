# FP8 / BF8 (`float8_e4m3fn`) Precision & 95% Confidence Interval Metrics

This note documents the native **FP8 (`float8_e4m3fn` / `e4m3`)** execution pipeline for training and inference, plus the **95% confidence interval (CI)** reporting subsystem added across SID-UNet.

---

## 1. FP8 / BF8 (`float8_e4m3fn`) Default Precision

### How Precision Resolution Works (`sid_unet.utils.precision`)
- **Entry Point**: `resolve_precision_spec(config, mode=..., device=..., fp8_all_parameters=...)` returns a `PrecisionSpec` instance.
- **Default Configuration**:
  - `training.mixed_precision: "auto"` (`training.precision: "auto"`) and `training.fp8_all_parameters: true` in `sid_unet/utils/config.py`.
  - `model.memory_dtype: "auto"` in `configs/experiments/noise_rag/noise_rag_prototype_v1.yaml`.
  - All CLI tools (`sid-train`, `sid-eval`, `sid-cross-eval`, `sid-predict`) accept `--precision` / `--mixed-precision` (`auto`, `fp8`, `bf8`, `e4m3`, `bf16`, `fp16`, `fp32`).
- **Automatic Hardware Capability Check (`check_fp8_compatibility` / `is_fp8_supported`)**:
  - **SM >= 8.9 (Ada Lovelace / Hopper / Blackwell)** with PyTorch `float8_e4m3fn`, `torch._scaled_mm`, and `torchao`: `"auto"` resolves to `mode="fp8"`, `fp8_dtype=torch.float8_e4m3fn`, `amp_dtype=torch.bfloat16`, `use_grad_scaler=False`.
  - **Pre-Ada CUDA GPUs (e.g., T4 SM 7.5, V100 SM 7.0, Ampere SM 8.0/8.6)** or unsupported environments: cleanly falls back to `"bf16"` when native hardware BF16 is supported (`is_bf16_supported()` checking `torch.cuda.is_bf16_supported(including_emulation=False)`, i.e., SM >= 8.0) or `"fp16"` on pre-Ampere GPUs such as T4/V100 (`GradScaler` enabled only when `amp_dtype == torch.float16`).
  - **CPU**: resolves to `fp32`.

### Training (`prepare_model_for_fp8_training`)
- Converts 16-aligned `nn.Linear` layers (`in_features % 16 == 0 and out_features % 16 == 0`) via `torchao.float8.convert_to_float8_training(..., Float8LinearConfig(pad_inner_dim=True))` with `Float8Linear`.
- Non-16-aligned Linear layers and `Conv2d` / `BatchNorm` / `LayerNorm` modules run in `amp_dtype` (`bfloat16`) autocast.
- `ForensicVectorDBRetrieval` (`sid_unet.models.noise_rag`) activates FP8 similarity search via cached transposed `float8_e4m3fn` keys + `torch._scaled_mm`.
- `GradScaler` is disabled under `bfloat16` / `fp8` compute (`scaler_enabled = use_amp and amp_dtype == torch.float16`).

### Inference (`prepare_model_for_fp8_inference`)
- Called across `sid-eval`, `sid-cross-eval`, and `sid-predict`.
- Aligns only `nn.Linear` and `nn.MultiheadAttention` parameters to `compute_dtype` (`bfloat16`) before TorchAO quantization—leaving `Conv2d` / `BatchNorm` / `NoisePrint++` residual filters untouched in `float32` so accuracy/calibration matches BF16 closely.
- Applies `Float8DynamicActivationFloat8WeightConfig` to 16-aligned `Linear` layers and, when `fp8_all_parameters=True`, `Float8WeightOnlyConfig` to non-16-aligned `Linear` layers plus FP8 retrieval on `ForensicVectorDBRetrieval`.

---

## 2. NVIDIA L4 (`SM 8.9`, 22.5 GiB) Benchmarks on `noise_rag_prototype_v1`

Using `configs/experiments/noise_rag/noise_rag_prototype_v1.yaml`, checkpoint pulled from `KhangTruong/noise-rag-prototype-v1-Casia-2.0`, vector DB `KhangTruong/mask-clip-db-mae-cache` (`1,000,000 x 256`), and dataset `KhangTruong/Casia-v2.0-Fake` (`512x512`):

| Benchmark | BF16 | FP8 (`e4m3`) | Relative Speedup & Notes |
|---|---|---|---|
| **Isolated VectorDB Matmul (`32768 x 256 @ 256 x 1M`, `chunk=1024`)** | `587.8 ms` (`621.5 ms` FP16) | `459.8 ms` (`_scaled_mm`) | **~22–26% faster** GEMM |
| **Isolated Forward Pass (fixed batch `B=8`)** | `1991.4 ms` (`4.02 img/s`) | `1837.3 ms` (`4.35 img/s`) | **+8.2% faster** compute (`mean_abs_diff = 0.0067` prob) |
| **End-to-End `sid-eval` (`64` val samples, `BS=8`, streaming)** | `37.37s` (`IoU=0.0789`) | `36.34s` (`IoU=0.0803 [0.0420, 0.1186]`) | **+2.8% faster** end-to-end (bounded by HF HTTP streaming + `topk`) |
| **End-to-End `sid-cross-eval` (`64` val samples, `BS=8`, streaming)** | `39.79s` | `37.06s` | **+7.4% faster** wall time |
| **Sustained `sid-train` (`BS=16`, ~90% avg / 100% peak GPU util, `145s`)** | `26` steps (`4.29s/it`, `3.73 img/s`, `7412 MiB`) | `26` steps (`4.19s/it` avg, `3.82 img/s`, `7768 MiB`) | **~2.3% faster** (speed parity on Conv/topk-heavy Noise-RAG) |

**Key takeaway**: On `Noise-RAG`, FP8 accelerates Tensor Core matmuls (`Float8Linear` and especially `ForensicVectorDBRetrieval` via `torch._scaled_mm`), yielding **+8% to +26%** speedup in pure compute/VectorDB kernels and maintaining numerical parity (`mean_abs_diff = 0.0067`). End-to-end `sid-train` throughput is within **~2.3%** of BF16 because ~88% of Noise-RAG compute is inside VectorDB similarity + `topk(4)` selection plus ConvNet backprop (`NoisePrint++` / `ConvViT`), where `topk` memory bandwidth dominates.

---

## 3. 95% Confidence Interval (CI) Metrics (`sid_unet.metrics.confidence`)

All primary evaluation, cross-evaluation, training validation, and Markdown/JSON reports now emit **95% confidence intervals** (`confidence_level=0.95`):

- **Segmentation Metrics (`SegmentationMetricTracker`)**:
  - Uses exact **Welford / Chan parallel online variance accumulation** (`OnlineVarianceTracker`) across sample-level scores (`iou`, `dice`, `f1`, `pixel_f1`, `macro_f1`, `auroc`, `pixel_auroc`, `pixel_acc`, `precision`, `recall`, `specificity`) globally, for positive-only masks (`*_positive`), per label (`per_label`), and per sub-dataset (`per_dataset`).
  - Computes two-sided Student's $t$ confidence intervals $\bar{x} \pm t_{\alpha/2,\,n-1} \cdot \frac{s}{\sqrt{n}}$ clamped to $[0, 1]$ (`compute_mean_confidence_interval`).
- **Auxiliary Classification Metrics (`ClassificationMetricTracker`)**:
  - `aux_accuracy`: Wilson score 95% CI (`compute_wilson_confidence_interval`).
  - `aux_macro_f1` and `aux_auroc`: deterministic percentile bootstrap 95% CI (`seed=42`, `n_bootstrap=300`, `compute_bootstrap_confidence_interval`).
- **Schema & Reporting (`sid_unet.utils.report`)**:
  - Emits both flat keys (`<metric>_ci_low`, `<metric>_ci_high`, `<metric>_ci_margin`, `<metric>_std`, `<metric>_ci95_low`, `<metric>_ci95_high`, `<metric>_ci95_margin`) and structured `confidence_intervals`.
  - Markdown tables (`format_metrics_table`, `generate_evaluation_report`, `generate_checkpoint_cross_eval_report`, `generate_master_cross_evaluation_report`) format `[low, high] (±margin)` cleanly without redundant suffix rows.
