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


def test_unet_from_checkpoint_with_embedded_config(tmp_path):
    # Create a custom architecture model
    original_model = UNet(
        in_channels=3,
        out_channels=1,
        features=[8, 16, 32],
        bilinear=False,
        aux_classifier=False,
    )
    custom_cfg = {
        "model": {
            "name": "unet",
            "in_channels": 3,
            "out_channels": 1,
            "features": [8, 16, 32],
            "bilinear": False,
            "aux_classifier": False,
        }
    }
    ckpt_path = tmp_path / "checkpoint_custom.pt"
    state = {
        "epoch": 5,
        "model_state_dict": original_model.state_dict(),
        "config": custom_cfg,
    }
    torch.save(state, str(ckpt_path))

    # Load via from_checkpoint
    loaded_model, loaded_cfg = UNet.from_checkpoint(str(ckpt_path), return_config=True)

    assert loaded_model.features == [8, 16, 32]
    assert loaded_model.bilinear is False
    assert loaded_model.aux_classifier is False
    assert loaded_cfg.model.features == [8, 16, 32]

    # Verify predictions match
    original_model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out_orig = original_model(x)
        out_loaded = loaded_model(x)
    assert torch.allclose(out_orig, out_loaded, atol=1e-6)


def test_unet_from_checkpoint_raw_state_dict(tmp_path):
    model = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512])
    ckpt_path = tmp_path / "raw_weights.pt"
    torch.save(model.state_dict(), str(ckpt_path))

    loaded_model = UNet.from_checkpoint(str(ckpt_path))
    assert loaded_model.features == [64, 128, 256, 512]


def test_unet_from_checkpoint_file_not_found():
    with pytest.raises(FileNotFoundError):
        UNet.from_checkpoint("/nonexistent/checkpoint.pt")


def test_efficientnet_unet_shapes_and_backward():
    from sid_unet.models.efficientnet import EfficientNetSegmentation
    model = EfficientNetSegmentation(
        backbone="efficientnet_b0",
        pretrained=False,
        sacrifice_of_pixel=False,
        aux_classifier=True,
        num_classes=3,
    )
    x = torch.randn(2, 3, 128, 128, requires_grad=True)
    mask_logits, cls_logits = model(x)

    assert mask_logits.shape == (2, 1, 128, 128)
    assert cls_logits.shape == (2, 3)

    loss = mask_logits.sum() + cls_logits.sum()
    loss.backward()
    assert x.grad is not None


def test_efficientnet_sacrifice_of_pixel():
    from sid_unet.models.efficientnet import EfficientNetSegmentation
    model = EfficientNetSegmentation(
        backbone="efficientnet_b0",
        pretrained=False,
        sacrifice_of_pixel=True,
        aux_classifier=True,
        num_classes=3,
    )
    x = torch.randn(2, 3, 256, 256, requires_grad=True)
    mask_logits, cls_logits = model(x)

    # Output matches full image resolution
    assert mask_logits.shape == (2, 1, 256, 256)
    assert cls_logits.shape == (2, 3)

    # Predict mask returns binary values
    pred = model.predict_mask(x[:1], threshold=0.5)
    assert pred.shape == (1, 1, 256, 256)
    assert set(torch.unique(pred).tolist()).issubset({0.0, 1.0})

    loss = mask_logits.sum() + cls_logits.sum()
    loss.backward()
    assert x.grad is not None


def test_efficientnet_checkpoint_save_and_load(tmp_path):
    from sid_unet.models.efficientnet import EfficientNetSegmentation
    cfg = {
        "model": {
            "name": "efficientnet",
            "backbone": "efficientnet_b0",
            "pretrained": False,
            "sacrifice_of_pixel": True,
            "aux_classifier": False,
            "in_channels": 3,
            "out_channels": 1,
        }
    }
    orig_model = build_model(cfg)
    assert isinstance(orig_model, EfficientNetSegmentation)
    assert orig_model.sacrifice_of_pixel is True

    ckpt_path = tmp_path / "eff_sac_ckpt.pt"
    torch.save({"model_state_dict": orig_model.state_dict(), "config": cfg}, str(ckpt_path))

    # Test loading via UNet.from_checkpoint and EfficientNetSegmentation.from_checkpoint
    loaded_model = UNet.from_checkpoint(str(ckpt_path))
    assert isinstance(loaded_model, EfficientNetSegmentation)
    assert loaded_model.sacrifice_of_pixel is True

