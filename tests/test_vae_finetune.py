"""
Unit and integration tests for DiffusionVAEFinetune model.
"""

import tempfile
import pytest
import torch

from sid_unet.models.vae_finetune import DiffusionVAEFinetune
from sid_unet.models.unet import build_model, UNet
from sid_unet.losses.auxiliary import SIDTotalLoss


def test_vae_finetune_shapes_with_aux_classifier():
    model = DiffusionVAEFinetune(
        use_dummy=True,
        dummy_channels=(32, 64),
        aux_classifier=True,
        num_classes=3,
    )
    x = torch.randn(2, 3, 64, 64)
    mask_logits, class_logits = model(x)

    assert mask_logits.shape == (2, 1, 64, 64)
    assert class_logits.shape == (2, 3)


def test_vae_finetune_shapes_without_aux_classifier():
    model = DiffusionVAEFinetune(
        use_dummy=True,
        dummy_channels=(32, 64),
        aux_classifier=False,
    )
    x = torch.randn(2, 3, 64, 64)
    mask_logits = model(x)

    assert mask_logits.shape == (2, 1, 64, 64)
    assert not isinstance(mask_logits, tuple)


def test_vae_finetune_backward_end_to_end():
    model = DiffusionVAEFinetune(
        use_dummy=True,
        dummy_channels=(32, 64),
        freeze_encoder=False,
        aux_classifier=True,
        num_classes=3,
    )
    x = torch.randn(2, 3, 64, 64)
    mask_logits, class_logits = model(x)
    loss = mask_logits.sum() + class_logits.sum()
    loss.backward()

    # Verify both encoder and decoder receive gradients when not frozen
    assert model.vae.decoder.conv_out.weight.grad is not None
    assert any(p.grad is not None for p in model.vae.encoder.parameters())


def test_vae_finetune_freeze_encoder():
    model = DiffusionVAEFinetune(
        use_dummy=True,
        dummy_channels=(32, 64),
        freeze_encoder=True,
        aux_classifier=False,
    )
    x = torch.randn(2, 3, 64, 64)
    mask_logits = model(x)
    loss = mask_logits.sum()
    loss.backward()

    # Encoder parameters must have no gradients
    for p in model.vae.encoder.parameters():
        assert p.grad is None
    # Decoder output conv must have gradients
    assert model.vae.decoder.conv_out.weight.grad is not None


def test_vae_finetune_predict_mask():
    model = DiffusionVAEFinetune(
        use_dummy=True,
        dummy_channels=(32, 64),
    )
    x = torch.randn(1, 3, 64, 64)
    mask = model.predict_mask(x, threshold=0.5)

    assert mask.shape == (1, 1, 64, 64)
    assert torch.all((mask == 0.0) | (mask == 1.0))


def test_vae_finetune_arbitrary_image_resolution():
    model = DiffusionVAEFinetune(
        use_dummy=True,
        dummy_channels=(32, 64),
    )
    # Non-multiple of 8 (67 x 75)
    x = torch.randn(1, 3, 67, 75)
    mask_logits, _ = model(x)
    assert mask_logits.shape == (1, 1, 67, 75)


def test_vae_finetune_build_model_and_checkpoint():
    config = {
        "model": {
            "name": "vae_finetune",
            "use_dummy": True,
            "dummy_channels": [32, 64],
            "freeze_encoder": False,
            "aux_classifier": True,
            "num_classes": 3,
        }
    }
    model = build_model(config)
    assert isinstance(model, DiffusionVAEFinetune)

    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        torch.save({"model_state_dict": model.state_dict(), "config": config}, tmp.name)
        loaded = UNet.from_checkpoint(tmp.name)
        assert isinstance(loaded, DiffusionVAEFinetune)


def test_vae_finetune_loss_integration():
    model = DiffusionVAEFinetune(
        use_dummy=True,
        dummy_channels=(32, 64),
        aux_classifier=True,
        num_classes=3,
    )
    loss_fn = SIDTotalLoss(mask_loss_type="combined", aux_classifier=True)
    images = torch.randn(2, 3, 64, 64)
    masks = torch.randint(0, 2, (2, 1, 64, 64)).float()
    labels = torch.tensor([0, 2])

    outputs = model(images)
    total_loss, metrics = loss_fn(outputs, masks, labels)

    assert total_loss.item() > 0.0
    assert "mask_loss" in metrics
    assert "aux_loss" in metrics


def test_vae_finetune_skip_connections():
    """Verify UNet-style encoder-to-decoder skip connections in DiffusionVAEFinetune."""
    # 1. Model with skip connections enabled (default)
    model_with_skips = DiffusionVAEFinetune(
        use_dummy=True,
        dummy_channels=(32, 64),
        use_skip_connections=True,
        freeze_encoder=False,
    )
    assert model_with_skips.use_skip_connections is True
    assert len(model_with_skips.skip_fusions) > 0

    x = torch.randn(2, 3, 64, 64)
    out_skips, _ = model_with_skips(x)
    assert out_skips.shape == (2, 1, 64, 64)

    # Test backward pass through skip fusions
    loss = out_skips.sum()
    loss.backward()
    for fusion in model_with_skips.skip_fusions:
        assert fusion.conv.weight.grad is not None

    # 2. Model with skip connections disabled
    model_no_skips = DiffusionVAEFinetune(
        use_dummy=True,
        dummy_channels=(32, 64),
        use_skip_connections=False,
    )
    assert model_no_skips.use_skip_connections is False
    assert len(model_no_skips.skip_fusions) == 0

    out_no_skips, _ = model_no_skips(x)
    assert out_no_skips.shape == (2, 1, 64, 64)
