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
