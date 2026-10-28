# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-10-28

### Added
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
