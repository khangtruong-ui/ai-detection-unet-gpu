import pytest
import torch
from sid_unet.models.unet import UNet, build_model
from sid_unet.utils.config import load_config


def test_unet_shapes_with_aux_classifier():
    model = UNet(
        in_channels=3,
        out_channels=1,
        features=[16, 32, 64],
        bilinear=True,
        aux_classifier=True,
        num_classes=3,
    )
    x = torch.randn(2, 3, 128, 128)
    mask_logits, class_logits = model(x)

    assert mask_logits.shape == (2, 1, 128, 128)
    assert class_logits.shape == (2, 3)


def test_unet_shapes_without_aux_classifier():
    model = UNet(
        in_channels=3,
        out_channels=1,
        features=[16, 32, 64],
        bilinear=True,
        aux_classifier=False,
    )
    x = torch.randn(2, 3, 128, 128)
    mask_logits = model(x)

    assert mask_logits.shape == (2, 1, 128, 128)


def test_unet_backward_gradients():
    model = UNet(
        in_channels=3,
        out_channels=1,
        features=[16, 32, 64],
        aux_classifier=True,
        num_classes=3,
    )
    x = torch.randn(2, 3, 64, 64)
    mask_logits, class_logits = model(x)

    loss = mask_logits.sum() + class_logits.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient is None for parameter {name}"


def test_unet_predict_mask():
    model = UNet(in_channels=3, out_channels=1, features=[16, 32], aux_classifier=True)
    x = torch.randn(1, 3, 64, 64)
    pred_mask = model.predict_mask(x, threshold=0.5)

    assert pred_mask.shape == (1, 1, 64, 64)
    unique_vals = torch.unique(pred_mask).tolist()
    for v in unique_vals:
        assert v in (0.0, 1.0)
