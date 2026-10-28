"""
Unit tests for IMLMaskformer (RefactorMaskformer) integration in sid_unet.
"""

import pytest
import torch
from sid_unet.models.iml_maskformer import (
    IMLMaskformer,
    RefactorMaskformer,
    RefectorMaskformer,
    HungarianMatcher,
    RefactorMaskformerLoss,
)
from sid_unet.models.unet import build_model


def test_iml_maskformer_initialization_dummy():
    """Verify IMLMaskformer initializes cleanly with dummy backbone on CPU."""
    model = IMLMaskformer(
        variant="large",
        embed_dim=64,
        num_queries=10,
        pixel_num_layers=2,
        decoder_num_layers=3,
        use_dummy=True,
        device="cpu",
    )
    assert model is not None
    assert isinstance(model, IMLMaskformer)
    assert isinstance(model, RefactorMaskformer)
    assert isinstance(model, RefectorMaskformer)


def test_iml_maskformer_forward_shape():
    """Verify forward pass produces expected dense mask logit shape [B, 1, H, W]."""
    model = IMLMaskformer(
        variant="large",
        embed_dim=64,
        num_queries=10,
        pixel_num_layers=2,
        decoder_num_layers=3,
        target_size=(384, 384),
        use_dummy=True,
        device="cpu",
    )
    x = torch.randn(2, 3, 384, 384)
    out = model(x)
    assert out.shape == (2, 1, 384, 384)
    assert torch.isfinite(out).all()


def test_iml_maskformer_predict_mask():
    """Verify predict_mask produces binary tensor values {0.0, 1.0}."""
    model = IMLMaskformer(
        variant="large",
        embed_dim=64,
        num_queries=10,
        pixel_num_layers=2,
        decoder_num_layers=3,
        use_dummy=True,
        device="cpu",
    )
    x = torch.randn(1, 3, 384, 384)
    mask = model.predict_mask(x, threshold=0.5)
    assert mask.shape == (1, 1, 384, 384)
    unique_vals = torch.unique(mask)
    for val in unique_vals:
        assert val.item() in (0.0, 1.0)


def test_iml_maskformer_return_dict():
    """Verify return_dict=True provides complete dictionary with intermediate tensors."""
    model = IMLMaskformer(
        variant="large",
        embed_dim=64,
        num_queries=10,
        pixel_num_layers=2,
        decoder_num_layers=3,
        use_dummy=True,
        device="cpu",
    )
    x = torch.randn(1, 3, 384, 384)
    res = model(x, return_dict=True)
    assert isinstance(res, dict)
    assert "mask_logits" in res
    assert "forgery_mask" in res
    assert "cls_logits" in res
    assert "query_mask_logits" in res
    assert res["cls_logits"].shape == (1, 10, 3)
    assert res["query_mask_logits"].shape == (1, 10, 96, 96)


def test_iml_hungarian_matcher():
    """Verify HungarianMatcher returns valid matching indices on CPU."""
    matcher = HungarianMatcher(mask_weight=5.0, dice_weight=5.0, num_points=128)
    cls_logits = torch.randn(2, 10, 3)
    mask_logits = torch.randn(2, 10, 48, 48)

    cls_labels = [torch.tensor([0, 1], dtype=torch.long), torch.tensor([0], dtype=torch.long)]
    mask_labels = [torch.zeros(2, 48, 48), torch.ones(1, 48, 48)]
    mask_labels[0][0, :24, :] = 1.0
    mask_labels[0][1, 24:, :] = 1.0

    matched = matcher(cls_logits, mask_logits, cls_labels, mask_labels)
    assert len(matched) == 2
    for pred_idx, tgt_idx in matched:
        assert len(pred_idx) == len(tgt_idx)


def test_iml_hungarian_loss():
    """Verify RefactorMaskformerLoss returns finite loss values."""
    loss_fn = RefactorMaskformerLoss(num_points=128)
    cls_logits = torch.randn(2, 10, 3)
    mask_logits = torch.randn(2, 10, 48, 48)

    cls_labels = [torch.tensor([0, 1], dtype=torch.long), torch.tensor([0], dtype=torch.long)]
    mask_labels = [torch.zeros(2, 48, 48), torch.ones(1, 48, 48)]
    mask_labels[0][0, :24, :] = 1.0
    mask_labels[0][1, 24:, :] = 1.0

    total_loss, ce, dice, mask_l = loss_fn(cls_logits, mask_logits, cls_labels, mask_labels)
    assert torch.isfinite(total_loss)
    assert torch.isfinite(ce)
    assert torch.isfinite(dice)
    assert torch.isfinite(mask_l)
    assert total_loss > 0


def test_iml_build_model():
    """Verify build_model instantiates IMLMaskformer from configuration dict."""
    cfg = {
        "model": {
            "name": "iml",
            "variant": "large",
            "embed_dim": 64,
            "num_queries": 10,
            "pixel_num_layers": 2,
            "decoder_num_layers": 3,
            "use_dummy": True,
        },
        "project": {"device": "cpu"},
    }
    model = build_model(cfg)
    assert isinstance(model, IMLMaskformer)
