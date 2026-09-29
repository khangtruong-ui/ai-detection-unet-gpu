"""
Unit and integration tests for DiffusionDiffMinimizedModel (Diffusion-Diff-Minimized).

Verifies:
1. 2 lines of computation:
   - Line 1: from the real (z0 clean latent via frozen VAE encoder).
   - Line 2: pick one noisy latent from the diffusion model (single chosen timestep, e.g. 250).
   - Z channel verification (84 channels vs 244 channels in v2).
2. Base model: Segmind Tiny-SD (segmind/tiny-sd) loaded via DiffusionPipeline.
3. Strict component freezing: diffuser UNet strictly frozen, parallel VAE decoder strictly frozen,
   VAE encoder frozen by default.
4. Perpendicular skip connection injection from parallel frozen decoder into trainable decoder stages.
5. Backward pass gradient flow: only trainable decoder and classifier receive gradients.
6. Forward shapes with and without auxiliary classifier head.
7. Mask prediction with sigmoid thresholding.
8. Model building from config dict and checkpoint serialization/loading.
9. Optimization step updating only trainable weights.
10. Real segmind/tiny-sd pipeline loading and execution on CUDA if available.
"""

import time
import tempfile
import pytest
import torch
import torch.nn.functional as F

from sid_unet.models.diffusion_diff import sinusoidal_embedding, expand_to_spatial
from sid_unet.models.diffusion_diff_minimized import (
    DiffusionDiffMinimizedModel,
    DiffusionDiffMinimized,
    DEFAULT_DIFFUSION_CHECKPOINT,
)
from sid_unet.models.diffusion_diff_v2 import DiffusionDiffV2Model
from sid_unet.models.unet import build_model, UNet


def test_diffusion_diff_minimized_two_lines_of_computation():
    """
    CRITICAL SPECIFICATION TEST:
    Verify that Diffusion-Diff-Minimized uses strictly 2 lines of computation:
    1. Line 1: from the real (z0 clean latent, 4 channels)
    2. Line 2: pick one noisy latent from the diffusion model (4 noisy + 4 eps + 4 pred_eps + 4 diff + 32 t_emb + 32 sigma_emb = 80 channels)
    Total Z channels = 4 + 80 = 84 channels (vs 244 in v2 with 3 noisy latents).
    """
    model = DiffusionDiffMinimizedModel(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timestep_embed_dim=32,
        sigma_embed_dim=32,
    )
    # Default timesteps must contain exactly 1 noisy latent
    assert len(model.timesteps) == 1
    assert model.timesteps == [250]
    # Total channels: 4 (z0) + 1 * (4 + 4 + 4 + 4 + 32 + 32) = 84
    assert model.total_z_channels == 84

    # Test custom single timestep argument
    model_t = DiffusionDiffMinimizedModel(
        use_dummy=True,
        timestep=150,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
    )
    assert model_t.timesteps == [150]
    assert model_t.total_z_channels == 84

    # Test passing a list of timesteps with enforce_single_noisy=True (picks one)
    model_pick = DiffusionDiffMinimizedModel(
        use_dummy=True,
        timesteps=[100, 250, 500],
        enforce_single_noisy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
    )
    assert len(model_pick.timesteps) == 1
    assert model_pick.timesteps == [250]
    assert model_pick.total_z_channels == 84


def test_diffusion_diff_minimized_shapes_with_aux_classifier():
    """Verify output shapes with auxiliary classifier head."""
    model = DiffusionDiffMinimizedModel(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[250],
        timestep_embed_dim=16,
        sigma_embed_dim=16,
        decoder_config={
            "channels": [64, 32, 16],
            "upsample_mode": "bilinear",
        },
        aux_classifier=True,
        num_classes=3,
    )
    x = torch.randn(2, 3, 64, 64)
    mask_logits, class_logits = model(x)

    assert mask_logits.shape == (2, 1, 64, 64)
    assert class_logits.shape == (2, 3)


def test_diffusion_diff_minimized_shapes_without_aux_classifier():
    """Verify output shapes without auxiliary classifier."""
    model = DiffusionDiffMinimizedModel(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[250],
        timestep_embed_dim=16,
        sigma_embed_dim=16,
        decoder_config={
            "channels": [64, 32, 16],
            "upsample_mode": "bilinear",
        },
        aux_classifier=False,
    )
    x = torch.randn(2, 3, 64, 64)
    mask_logits = model(x)

    assert mask_logits.shape == (2, 1, 64, 64)
    assert not isinstance(mask_logits, tuple)


def test_diffusion_diff_minimized_default_frozen_encoder_and_components():
    """
    Verify:
    - Encoder is frozen by default (freeze_encoder=True).
    - Diffuser UNet is strictly frozen.
    - Parallel VAE decoder is strictly frozen.
    - Only TrainableLatentDecoder and ClassifierHead are trainable.
    """
    model = DiffusionDiffMinimizedModel(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[250],
        timestep_embed_dim=16,
        sigma_embed_dim=16,
        decoder_config={
            "channels": [64, 32, 16],
            "upsample_mode": "bilinear",
        },
        aux_classifier=True,
        num_classes=3,
    )
    assert model.freeze_encoder is True

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]

    assert any(n.startswith("decoder.") for n in trainable)
    assert any(n.startswith("classifier_head.") for n in trainable)

    assert all(not n.startswith("vae.encoder.") for n in trainable)
    assert all(not n.startswith("vae.decoder.") for n in trainable)
    assert all(not n.startswith("diffuser.") for n in trainable)

    assert any(n.startswith("vae.encoder.") for n in frozen)
    assert any(n.startswith("vae.decoder.") for n in frozen)
    assert any(n.startswith("diffuser.") for n in frozen)

    # Backward pass verification
    x = torch.randn(2, 3, 64, 64)
    mask_logits, class_logits = model(x)
    loss = mask_logits.sum() + class_logits.sum()
    loss.backward()

    for n, p in model.named_parameters():
        if n.startswith(("diffuser.", "vae.encoder.", "vae.decoder.")):
            assert p.grad is None, f"Frozen component {n} unexpectedly received gradients!"

    assert model.decoder.out_conv.weight.grad is not None
    assert any(p.grad is not None for p in model.classifier_head.parameters())
    perp_grads = [
        f.conv.weight.grad is not None
        for f in model.decoder.perp_fusions
        if hasattr(f, "conv")
    ]
    assert len(perp_grads) > 0 and all(perp_grads)


def test_diffusion_diff_minimized_unfreeze_encoder_option():
    """Verify that when freeze_encoder=False, encoder is trainable while diffuser remains frozen."""
    model = DiffusionDiffMinimizedModel(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[250],
        freeze_encoder=False,
    )
    assert model.freeze_encoder is False
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]

    assert any(n.startswith("vae.encoder.") for n in trainable)
    assert all(not n.startswith("vae.decoder.") for n in trainable)
    assert all(not n.startswith("diffuser.") for n in trainable)
    assert any(n.startswith("vae.decoder.") for n in frozen)
    assert any(n.startswith("diffuser.") for n in frozen)


def test_diffusion_diff_minimized_default_no_encoder_skips():
    """Verify that encoder-to-trainable-decoder skips are False by default."""
    model = DiffusionDiffMinimizedModel(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[250],
    )
    assert model.use_encoder_skips is False
    assert len(model.decoder.encoder_skip_fusions) == 0

    model_with_skips = DiffusionDiffMinimizedModel(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[250],
        use_encoder_skips=True,
    )
    assert model_with_skips.use_encoder_skips is True
    assert len(model_with_skips.decoder.encoder_skip_fusions) > 0


def test_diffusion_diff_minimized_perpendicular_skips():
    """Verify parallel frozen decoder and perpendicular skip injection."""
    model = DiffusionDiffMinimizedModel(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[250],
        use_perpendicular_skips=True,
        decoder_config={"channels": [32, 16]},
        aux_classifier=False,
    )
    assert model.use_perpendicular_skips is True
    assert len(model.decoder.perp_fusions) > 0
    assert len(model.perp_mapping) > 0

    x = torch.randn(2, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (2, 1, 64, 64)


def test_diffusion_diff_minimized_decoder_config_options():
    """Test custom decoder configuration options."""
    model = DiffusionDiffMinimizedModel(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[250],
        timestep_embed_dim=16,
        sigma_embed_dim=16,
        decoder_config={
            "channels": [128, 64, 32],
            "upsample_mode": "transpose",
            "norm_layer": "groupnorm",
            "activation": "leaky_relu",
            "dropout": 0.1,
            "num_res_blocks": 2,
        },
        aux_classifier=False,
    )
    x = torch.randn(1, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (1, 1, 64, 64)


def test_diffusion_diff_minimized_predict_mask():
    """Verify predict_mask binary output."""
    model = DiffusionDiffMinimizedModel(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[250],
        aux_classifier=True,
    )
    x = torch.randn(2, 3, 64, 64)
    mask = model.predict_mask(x, threshold=0.5)

    assert mask.shape == (2, 1, 64, 64)
    unique_vals = torch.unique(mask)
    for val in unique_vals:
        assert val.item() in (0.0, 1.0)


def test_diffusion_diff_minimized_arbitrary_image_resolution():
    """Verify non-divisible by 8 padding and reconstruction."""
    model = DiffusionDiffMinimizedModel(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[250],
        aux_classifier=False,
    )
    x = torch.randn(1, 3, 70, 75)
    logits = model(x)
    assert logits.shape == (1, 1, 70, 75)


def test_diffusion_diff_minimized_build_model_and_checkpoint():
    """Verify build_model integration and checkpoint saving/loading."""
    config_dict = {
        "model": {
            "name": "diffusion_diff_minimized",
            "use_dummy": True,
            "dummy_vae_channels": [32, 64],
            "dummy_unet_channels": [32, 64],
            "timesteps": [250],
            "timestep_embed_dim": 16,
            "sigma_embed_dim": 16,
            "decoder": {"channels": [64, 32, 16]},
            "aux_classifier": True,
            "num_classes": 3,
        }
    }
    model = build_model(config_dict)
    assert isinstance(model, DiffusionDiffMinimizedModel)
    assert len(model.timesteps) == 1

    x = torch.randn(2, 3, 64, 64)
    model.eval()
    with torch.no_grad():
        out1 = model(x)

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": config_dict,
        }
        torch.save(checkpoint, f.name)

        loaded_model = DiffusionDiffMinimizedModel.from_checkpoint(f.name)
        assert isinstance(loaded_model, DiffusionDiffMinimizedModel)
        assert len(loaded_model.timesteps) == 1

        with torch.no_grad():
            out2 = loaded_model(x)

        assert torch.allclose(out1[0], out2[0], atol=1e-5)
        assert torch.allclose(out1[1], out2[1], atol=1e-5)


def test_diffusion_diff_minimized_optimization_step():
    """Verify optimizer step updates only trainable parameters."""
    model = DiffusionDiffMinimizedModel(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[250],
        timestep_embed_dim=16,
        sigma_embed_dim=16,
        decoder_config={"channels": [64, 32, 16]},
        aux_classifier=True,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    diffuser_param_before = list(model.diffuser.parameters())[0].clone()
    vae_enc_param_before = list(model.vae.encoder.parameters())[0].clone()
    vae_dec_param_before = list(model.vae.decoder.parameters())[0].clone()
    dec_param_before = model.decoder.out_conv.weight.clone()

    x = torch.randn(2, 3, 64, 64)
    model.train()
    optimizer.zero_grad()
    mask_logits, class_logits = model(x)
    loss = mask_logits.sum() + class_logits.sum()
    loss.backward()
    optimizer.step()

    # Frozen components remain completely unchanged
    assert torch.equal(list(model.diffuser.parameters())[0], diffuser_param_before)
    assert torch.equal(list(model.vae.encoder.parameters())[0], vae_enc_param_before)
    assert torch.equal(list(model.vae.decoder.parameters())[0], vae_dec_param_before)

    # Trainable decoder parameters are updated
    assert not torch.equal(model.decoder.out_conv.weight, dec_param_before)


def test_diffusion_diff_minimized_compute_reduction_vs_v2():
    """
    Empirically verify compute time reduction:
    Diffusion-Diff-Minimized evaluates UNet only 1 time (1 noisy latent),
    whereas Diffusion-Diff-V2 evaluates UNet 3 times (3 noisy latents).
    """
    # Minimized model (1 noisy latent)
    model_min = DiffusionDiffMinimizedModel(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[250],
        decoder_config={"channels": [64, 32, 16]},
        aux_classifier=False,
    )
    # V2 model (3 noisy latents)
    model_v2 = DiffusionDiffV2Model(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[100, 250, 500],
        decoder_config={"channels": [64, 32, 16]},
        aux_classifier=False,
    )

    x = torch.randn(2, 3, 64, 64)

    # Warmup
    _ = model_min(x)
    _ = model_v2(x)

    iters = 10

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model_min(x)
    t_min = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model_v2(x)
    t_v2 = time.perf_counter() - t0

    # Minimized must be faster than V2
    assert t_min < t_v2, f"Minimized time ({t_min:.4f}s) should be less than V2 ({t_v2:.4f}s)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for segmind/tiny-sd GPU test")
def test_diffusion_diff_minimized_real_tiny_sd_cuda():
    """
    Verify real segmind/tiny-sd loading via:
    DiffusionPipeline.from_pretrained("segmind/tiny-sd", dtype=torch.float16, device_map="cuda")
    and executing end-to-end forward pass on CUDA.
    """
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        DEFAULT_DIFFUSION_CHECKPOINT,
        dtype=torch.float16,
        device_map="cuda",
    )
    model = DiffusionDiffMinimizedModel(
        pipeline=pipe,
        timesteps=[250],
        decoder_config={"channels": [128, 64, 32, 16]},
        aux_classifier=True,
        num_classes=3,
        diffuser_fp16=True,
    )
    # The pipeline is on CUDA, so model parameters for decoder can be moved to cuda
    model.to("cuda")

    x = torch.randn(1, 3, 256, 256, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        mask_logits, class_logits = model(x)

    assert mask_logits.device.type == "cuda"
    assert mask_logits.shape == (1, 1, 256, 256)
    assert class_logits.shape == (1, 3)
