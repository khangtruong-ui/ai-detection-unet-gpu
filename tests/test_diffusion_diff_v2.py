"""
Unit and integration tests for DiffusionDiffV2Model (Diffusion Multi-Noise Latent Feature Decoder V2).

Verifies:
1. Frozen VAE encoder by default and strictly frozen diffuser and parallel frozen decoder.
2. No encoder skip connections by default (use_encoder_skips=False).
3. Parallel pretrained, frozen decoder with perpendicular skip connections injected into trainable decoder.
4. Output tensor shapes with and without auxiliary classifier.
5. Gradient flow: only trainable decoder and classifier (and encoder if unfrozen) receive gradients.
6. Checkpoint save/load round-tripping.
7. Optimizer step updates only trainable parameters.
"""

import tempfile
import pytest
import torch
import torch.nn.functional as F

from sid_unet.models.diffusion_diff import sinusoidal_embedding, expand_to_spatial
from sid_unet.models.diffusion_diff_v2 import (
    DiffusionDiffV2Model,
    DiffusionDiffV2,
    TrainableLatentDecoderV2,
    PerpendicularSkipFusion,
)
from sid_unet.models.unet import build_model, UNet


def test_sinusoidal_embedding():
    timesteps = torch.tensor([10.0, 100.0, 500.0])
    emb = sinusoidal_embedding(timesteps, dim=32)
    assert emb.shape == (3, 32)
    assert not torch.allclose(emb[0], emb[1])

    spatial = expand_to_spatial(emb, height=16, width=16)
    assert spatial.shape == (3, 32, 16, 16)
    assert torch.equal(spatial[:, :, 0, 0], emb)


def test_diffusion_diff_v2_shapes_with_aux_classifier():
    model = DiffusionDiffV2Model(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[100, 250],
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


def test_diffusion_diff_v2_shapes_without_aux_classifier():
    model = DiffusionDiffV2Model(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[100, 250],
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


def test_diffusion_diff_v2_default_frozen_encoder_and_components():
    """
    CRITICAL ARCHITECTURAL TEST:
    In Diffusion-Diff-V2:
    - Encoder is frozen by default (freeze_encoder=True).
    - Diffuser UNet is strictly frozen.
    - Parallel VAE decoder is strictly frozen.
    - Only TrainableLatentDecoderV2 and ClassifierHead are trainable.
    """
    model = DiffusionDiffV2Model(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[100, 250],
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

    # Decoder and ClassifierHead must be trainable
    assert any(n.startswith("decoder.") for n in trainable)
    assert any(n.startswith("classifier_head.") for n in trainable)

    # VAE Encoder, VAE Decoder, and Diffuser UNet MUST all be strictly frozen
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

    # Diffuser, VAE encoder, and VAE decoder MUST NOT have gradients
    for n, p in model.named_parameters():
        if n.startswith(("diffuser.", "vae.encoder.", "vae.decoder.")):
            assert p.grad is None, f"Frozen component {n} unexpectedly received gradients!"

    # Trainable decoder parameters MUST have valid gradients
    assert model.decoder.out_conv.weight.grad is not None
    assert any(p.grad is not None for p in model.classifier_head.parameters())
    # Perpendicular skip fusions must have valid gradients
    perp_grads = [f.conv.weight.grad is not None for f in model.decoder.perp_fusions if hasattr(f, "conv")]
    assert len(perp_grads) > 0 and all(perp_grads)


def test_diffusion_diff_v2_unfreeze_encoder_option():
    """Verify that when freeze_encoder=False, encoder is trainable while diffuser and frozen decoder remain frozen."""
    model = DiffusionDiffV2Model(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[100],
        freeze_encoder=False,
    )
    assert model.freeze_encoder is False
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]

    # Encoder is trainable
    assert any(n.startswith("vae.encoder.") for n in trainable)
    # Decoder and Diffuser remain frozen
    assert all(not n.startswith("vae.decoder.") for n in trainable)
    assert all(not n.startswith("diffuser.") for n in trainable)
    assert any(n.startswith("vae.decoder.") for n in frozen)
    assert any(n.startswith("diffuser.") for n in frozen)


def test_diffusion_diff_v2_default_no_encoder_skips():
    """Verify that encoder-to-trainable-decoder skip connection is False by default."""
    model = DiffusionDiffV2Model(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[100],
    )
    assert model.use_encoder_skips is False
    assert len(model.decoder.encoder_skip_fusions) == 0

    # Test explicitly enabling encoder skips
    model_with_enc_skips = DiffusionDiffV2Model(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[100],
        use_encoder_skips=True,
    )
    assert model_with_enc_skips.use_encoder_skips is True
    assert len(model_with_enc_skips.decoder.encoder_skip_fusions) > 0


def test_diffusion_diff_v2_perpendicular_skips():
    """Verify parallel frozen decoder and perpendicular skip injection."""
    # 1. Perpendicular skips enabled (default)
    model = DiffusionDiffV2Model(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[100],
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

    # 2. Perpendicular skips disabled
    model_no_perp = DiffusionDiffV2Model(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[100],
        use_perpendicular_skips=False,
        decoder_config={"channels": [32, 16]},
        aux_classifier=False,
    )
    assert model_no_perp.use_perpendicular_skips is False
    assert len(model_no_perp.decoder.perp_fusions) == 0
    logits_no_perp = model_no_perp(x)
    assert logits_no_perp.shape == (2, 1, 64, 64)


def test_diffusion_diff_v2_decoder_config_options():
    """Test custom decoder architectures (transpose upsampling, groupnorm, leaky_relu)."""
    model = DiffusionDiffV2Model(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[100, 300],
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


def test_diffusion_diff_v2_predict_mask():
    model = DiffusionDiffV2Model(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[100],
        decoder_config={"channels": [32, 16]},
    )
    x = torch.randn(1, 3, 64, 64)
    mask = model.predict_mask(x, threshold=0.5)

    assert mask.shape == (1, 1, 64, 64)
    assert torch.all((mask == 0.0) | (mask == 1.0))


def test_diffusion_diff_v2_arbitrary_image_resolution():
    model = DiffusionDiffV2Model(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[100],
        decoder_config={"channels": [32, 16]},
    )
    # Non-multiple of 8 (71 x 83)
    x = torch.randn(1, 3, 71, 83)
    mask_logits, _ = model(x)
    assert mask_logits.shape == (1, 1, 71, 83)


def test_diffusion_diff_v2_build_model_and_checkpoint():
    config = {
        "model": {
            "name": "diffusion_diff_v2",
            "use_dummy": True,
            "dummy_vae_channels": [32, 64],
            "dummy_unet_channels": [32, 64],
            "timesteps": [100, 250],
            "timestep_embed_dim": 16,
            "sigma_embed_dim": 16,
            "decoder": {
                "channels": [64, 32, 16],
                "upsample_mode": "bilinear",
            },
            "aux_classifier": True,
            "num_classes": 3,
        }
    }
    model = build_model(config)
    assert isinstance(model, DiffusionDiffV2Model)

    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        torch.save({"model_state_dict": model.state_dict(), "config": config}, tmp.name)
        loaded = UNet.from_checkpoint(tmp.name)
        assert isinstance(loaded, DiffusionDiffV2Model)


def test_diffusion_diff_v2_optimization_step():
    """Verify optimizer step updates decoder and classifier while frozen components remain untouched."""
    model = DiffusionDiffV2Model(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[100],
        decoder_config={"channels": [32, 16]},
        aux_classifier=False,
    )
    assert model.freeze_encoder is True
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

    initial_decoder_weight = model.decoder.out_conv.weight.clone()
    initial_vae_encoder_weight = list(model.vae.encoder.parameters())[0].clone()
    initial_vae_decoder_weight = list(model.vae.decoder.parameters())[0].clone()
    initial_diffuser_weight = list(model.diffuser.parameters())[0].clone()

    x = torch.randn(2, 3, 64, 64)
    target = torch.ones(2, 1, 64, 64)
    logits = model(x)
    loss = F.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    optimizer.step()

    # Trainable decoder weight updated
    assert not torch.equal(model.decoder.out_conv.weight, initial_decoder_weight)
    # Frozen VAE encoder weight untouched
    assert torch.equal(list(model.vae.encoder.parameters())[0], initial_vae_encoder_weight)
    # Frozen parallel VAE decoder weight untouched
    assert torch.equal(list(model.vae.decoder.parameters())[0], initial_vae_decoder_weight)
    # Frozen diffuser weight untouched
    assert torch.equal(list(model.diffuser.parameters())[0], initial_diffuser_weight)


def test_diffusion_diff_v2_z_normalization():
    """Verify that z_norm normalizes high-dimensional representation Z properly in v2."""
    model = DiffusionDiffV2Model(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[100, 250],
        timestep_embed_dim=16,
        sigma_embed_dim=16,
        decoder_config={"channels": [32, 16], "norm_layer": "groupnorm"},
    )
    assert hasattr(model, "z_norm")
    assert isinstance(model.z_norm, torch.nn.GroupNorm)

    x = torch.randn(2, 3, 64, 64)
    logits = model(x)
    mask_logits = logits[0] if isinstance(logits, tuple) else logits
    assert torch.all(torch.isfinite(mask_logits))


def test_diffusion_diff_v2_eval_determinism_and_contiguity():
    """Verify that v2 evaluation mode is deterministic across calls and outputs are contiguous."""
    model = DiffusionDiffV2Model(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
    )
    model.eval()

    # Non-multiple of 8 triggers reflection padding and slicing
    x = torch.randn(2, 3, 70, 70)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    mask1 = out1[0] if isinstance(out1, tuple) else out1
    mask2 = out2[0] if isinstance(out2, tuple) else out2

    assert mask1.is_contiguous(), "Mask logits must be contiguous in memory"
    assert torch.allclose(mask1, mask2, atol=1e-6), "Repeated evaluation passes on identical inputs must be deterministic"

