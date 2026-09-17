"""
Unit and integration tests for DiffusionDiffModel (diffusion multi-noise latent feature decoder).
"""

import math
import tempfile
import pytest
import torch

from sid_unet.models.diffusion_diff import (
    DiffusionDiffModel,
    TrainableLatentDecoder,
    sinusoidal_embedding,
    expand_to_spatial,
)
from sid_unet.models.unet import build_model, UNet
from sid_unet.losses.auxiliary import SIDTotalLoss


def test_sinusoidal_embedding():
    # Test scalar values
    timesteps = torch.tensor([10.0, 100.0, 500.0])
    emb = sinusoidal_embedding(timesteps, dim=32)
    assert emb.shape == (3, 32)
    # Check that different inputs yield distinct embeddings
    assert not torch.allclose(emb[0], emb[1])

    # Test spatial expansion
    spatial = expand_to_spatial(emb, height=16, width=16)
    assert spatial.shape == (3, 32, 16, 16)
    assert torch.equal(spatial[:, :, 0, 0], emb)


def test_diffusion_diff_shapes_with_aux_classifier():
    model = DiffusionDiffModel(
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


def test_diffusion_diff_shapes_without_aux_classifier():
    model = DiffusionDiffModel(
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


def test_diffusion_diff_trainable_parameters_strictly_decoder():
    """
    CRITICAL TEST:
    Verify that ONLY the decoder (and optional aux head) has trainable parameters,
    and neither the VAE encoder nor the diffuser receives gradients.
    """
    model = DiffusionDiffModel(
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

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]

    # Every trainable param MUST belong to decoder or classifier_head
    for name in trainable:
        assert name.startswith(("decoder.", "classifier_head.")), f"Unexpected trainable parameter: {name}"

    # Base models MUST be frozen
    assert any(n.startswith("vae.") for n in frozen)
    assert any(n.startswith("diffuser.") for n in frozen)

    x = torch.randn(2, 3, 64, 64)
    mask_logits, class_logits = model(x)
    loss = mask_logits.sum() + class_logits.sum()
    loss.backward()

    # VAE and Diffuser parameters MUST NOT have gradients
    for n, p in model.named_parameters():
        if n.startswith(("vae.", "diffuser.")):
            assert p.grad is None, f"Frozen parameter {n} unexpectedly received gradients!"

    # Decoder and classifier head parameters MUST have valid gradients
    for n, p in model.named_parameters():
        if n.startswith(("decoder.out_conv.", "classifier_head.")):
            assert p.grad is not None, f"Trainable parameter {n} missing gradient!"


def test_diffusion_diff_decoder_config_options():
    """Test custom decoder architectures defined by configs (transpose upsampling, norm, activation)."""
    model = DiffusionDiffModel(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[100, 300, 600],
        timestep_embed_dim=32,
        sigma_embed_dim=32,
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


def test_diffusion_diff_predict_mask():
    model = DiffusionDiffModel(
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


def test_diffusion_diff_arbitrary_image_resolution():
    model = DiffusionDiffModel(
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


def test_diffusion_diff_build_model_and_checkpoint():
    config = {
        "model": {
            "name": "diffusion_diff",
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
    assert isinstance(model, DiffusionDiffModel)

    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        torch.save({"model_state_dict": model.state_dict(), "config": config}, tmp.name)
        loaded = UNet.from_checkpoint(tmp.name)
        assert isinstance(loaded, DiffusionDiffModel)


def test_diffusion_diff_optimization_step():
    """Verify optimizer step updates decoder weights while frozen weights are untouched."""
    model = DiffusionDiffModel(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[100],
        decoder_config={"channels": [32, 16]},
        aux_classifier=False,
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

    initial_decoder_weight = model.decoder.out_conv.weight.clone()
    initial_vae_weight = list(model.vae.parameters())[0].clone()

    x = torch.randn(2, 3, 64, 64)
    target = torch.ones(2, 1, 64, 64)
    logits = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    optimizer.step()

    # Decoder weight changed
    assert not torch.equal(model.decoder.out_conv.weight, initial_decoder_weight)
    # Frozen VAE weight completely unchanged
    assert torch.equal(list(model.vae.parameters())[0], initial_vae_weight)


def test_diffusion_diff_skip_connections():
    """Verify UNet-style encoder skip connections in DiffusionDiffModel."""
    # 1. With skip connections enabled (default)
    model_with_skips = DiffusionDiffModel(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[100],
        decoder_config={"channels": [32, 16]},
        use_skip_connections=True,
        aux_classifier=False,
    )
    assert model_with_skips.use_skip_connections is True
    assert model_with_skips.decoder.skip_fusions is not None
    assert len(model_with_skips.decoder.skip_fusions) > 0

    x = torch.randn(2, 3, 64, 64)
    logits_skips = model_with_skips(x)
    assert logits_skips.shape == (2, 1, 64, 64)

    # Check that skip fusion parameters are trainable and receive gradients
    loss = logits_skips.sum()
    loss.backward()
    for fusion in model_with_skips.decoder.skip_fusions:
        assert fusion.conv.weight.grad is not None
    # Frozen VAE and UNet must still have no gradients
    for p in model_with_skips.vae.parameters():
        assert p.grad is None
    for p in model_with_skips.diffuser.parameters():
        assert p.grad is None

    # 2. With skip connections disabled
    model_no_skips = DiffusionDiffModel(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[100],
        decoder_config={"channels": [32, 16]},
        use_skip_connections=False,
        aux_classifier=False,
    )
    assert model_no_skips.use_skip_connections is False
    assert len(model_no_skips.decoder.skip_fusions) == 0
    logits_no_skips = model_no_skips(x)
    assert logits_no_skips.shape == (2, 1, 64, 64)
