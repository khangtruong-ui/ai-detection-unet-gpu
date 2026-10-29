"""
Unit and integration tests for Detective SAM (ICLR 2026) model integration.
Covers perturbation operator, feature adapters, mask adapter, coarse transformer,
spatial gating, full model forward pass, gradient flow, and loss functions.
"""

import pytest
import torch
import torch.nn as nn
from sid_unet.models.detective_sam import (
    DetectiveSAM,
    ImagePerturbation,
    FeatureAdapter,
    MaskAdapter,
    DetectiveSAMLoss,
)
from sid_unet.models.unet import build_model
from sid_unet.utils.config import ConfigDict


def test_image_perturbation():
    """Verify ImagePerturbation generates valid blurred and noisy images."""
    pert = ImagePerturbation(perturbation_type="blur_noise", noise_intensity=0.75)
    x = torch.rand(2, 3, 64, 64)
    out = pert(x)
    assert out.shape == x.shape
    assert not torch.allclose(out, x), "Perturbed image should differ from original"
    assert torch.isfinite(out).all()

    # Identity / none
    pert_none = ImagePerturbation(perturbation_type="none")
    out_none = pert_none(x)
    assert torch.equal(out_none, x)


def test_feature_adapter():
    """Verify FeatureAdapter residual delta correction and gradient propagation."""
    adapter = FeatureAdapter(in_channels=32, width=64)
    f_i = torch.randn(2, 32, 16, 16)
    f_pert = torch.randn(2, 32, 16, 16)

    f_tilde = adapter(f_i, f_pert)
    assert f_tilde.shape == (2, 32, 16, 16)

    loss = f_tilde.sum()
    loss.backward()
    trainable_grads = [p.grad for p in adapter.parameters() if p.grad is not None]
    assert len(trainable_grads) > 0, "FeatureAdapter parameters should receive gradients"


def test_mask_adapter_shapes_and_gating():
    """Verify MaskAdapter cross-scale fusion, coarse transformer, and spatial gating."""
    mask_adapter = MaskAdapter(
        in_channels_list=[32, 64, 256],
        d=64,
        downscale_factor=8,
        transformer_layers=1,
        num_heads=4,
        dropout=0.15,
    )
    b = 2
    f_i = [torch.randn(b, 32, 32, 32), torch.randn(b, 64, 16, 16), torch.randn(b, 256, 8, 8)]
    f_pert = [torch.randn(b, 32, 32, 32), torch.randn(b, 64, 16, 16), torch.randn(b, 256, 8, 8)]
    f_tilde = [torch.randn(b, 32, 32, 32), torch.randn(b, 64, 16, 16), torch.randn(b, 256, 8, 8)]

    m, l_coarse, l_refine, u = mask_adapter(f_i, f_pert, f_tilde)

    assert m.shape == (b, 1, 32, 32)
    assert l_coarse.shape == (b, 1, 32, 32)
    assert l_refine.shape == (b, 1, 32, 32)
    assert u.shape == (b, 1, 32, 32)
    assert torch.isfinite(m).all()

    # Backward pass
    loss = m.sum()
    loss.backward()
    grads = [p.grad for p in mask_adapter.parameters() if p.grad is not None]
    assert len(grads) > 0, "MaskAdapter parameters should receive gradients"


def test_detective_sam_dummy_init_and_forward():
    """Verify DetectiveSAM initializes cleanly in dummy mode and processes forward pass."""
    model = DetectiveSAM(
        variant="small",
        layer_width=64,
        target_size=(256, 256),
        use_dummy=True,
        device="cpu",
    )
    assert model is not None
    assert model.classifier_head is None

    x = torch.randn(2, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (2, 1, 64, 64)
    assert torch.isfinite(logits).all()


def test_detective_sam_return_dict():
    """Verify return_dict=True provides complete dictionary with all intermediate maps."""
    model = DetectiveSAM(
        variant="small",
        layer_width=64,
        target_size=(256, 256),
        use_dummy=True,
        device="cpu",
    )
    x = torch.randn(1, 3, 64, 64)
    res = model(x, return_dict=True)
    assert isinstance(res, dict)
    assert "mask_logits" in res
    assert "heatmap_prompt" in res
    assert "l_coarse" in res
    assert "l_refine" in res
    assert "uncertainty" in res
    assert "perturbed_images" in res
    assert res["mask_logits"].shape == (1, 1, 64, 64)


def test_detective_sam_with_aux_classifier():
    """Verify auxiliary classifier produces [B, num_classes] logits."""
    model = DetectiveSAM(
        variant="small",
        layer_width=64,
        aux_classifier=True,
        num_classes=3,
        target_size=(256, 256),
        use_dummy=True,
        device="cpu",
    )
    assert model.classifier_head is not None

    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert isinstance(out, tuple)
    mask_logits, cls_logits = out
    assert mask_logits.shape == (2, 1, 64, 64)
    assert cls_logits.shape == (2, 3)


def test_detective_sam_predict_mask():
    """Verify predict_mask produces binary output {0.0, 1.0}."""
    model = DetectiveSAM(
        variant="small",
        target_size=(256, 256),
        use_dummy=True,
        device="cpu",
    )
    x = torch.randn(1, 3, 64, 64)
    pred = model.predict_mask(x, threshold=0.5)
    assert pred.shape == (1, 1, 64, 64)
    unique_vals = set(pred.unique().cpu().numpy())
    assert unique_vals.issubset({0.0, 1.0})


def test_detective_sam_build_model_from_config():
    """Verify build_model instantiates DetectiveSAM from YAML-style configuration."""
    cfg = ConfigDict({
        "project": {"device": "cpu"},
        "model": {
            "name": "detective_sam",
            "variant": "small",
            "layer_width": 64,
            "noise_intensity": 0.75,
            "perturbation_type": "blur_noise",
            "use_dummy": True,
            "target_size": [256, 256],
        }
    })
    model = build_model(cfg)
    assert isinstance(model, DetectiveSAM)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert out.shape == (1, 1, 64, 64)


def test_detective_sam_loss():
    """Verify DetectiveSAMLoss computes focal, dice, and IoU loss components."""
    loss_fn = DetectiveSAMLoss(focal_weight=20.0, dice_weight=1.0, iou_weight=1.0, alpha=0.6, gamma=1.0)
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    targets = (torch.rand(2, 1, 32, 32) > 0.5).float()

    loss, loss_dict = loss_fn(logits, targets)
    assert torch.isfinite(loss)
    assert "loss_focal" in loss_dict
    assert "loss_dice" in loss_dict
    assert "loss_iou" in loss_dict
    assert "iou" in loss_dict
    assert 0.0 <= loss_dict["iou"] <= 1.0

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_detective_sam_gpu_gradient_flow():
    """Verify DetectiveSAM on GPU with SAM2 backbone has trainable adapters and frozen backbone."""
    model = DetectiveSAM(
        variant="small",
        layer_width=64,
        target_size=(512, 512),
        freeze_sam2=True,
        freeze_decoder=True,
        use_dummy=False,
        device="cuda",
    ).cuda()

    # Trainable vs frozen
    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen = [p for p in model.parameters() if not p.requires_grad]

    assert len(trainable) > 0, "Adapters must be trainable"
    assert len(frozen) > 0, "Base SAM2 backbone must be frozen"

    # Forward + backward pass on GPU
    x = torch.randn(1, 3, 128, 128, device="cuda")
    logits = model(x)
    loss = logits.mean()
    loss.backward()

    active_grads = [p.grad for p in trainable if p.grad is not None]
    assert len(active_grads) > 0, "Trainable adapters must receive gradients"
    for p in frozen:
        assert p.grad is None, "Frozen backbone parameter unexpectedly received gradient"


def test_detective_sam_default_target_size():
    """Verify DetectiveSAM defaults to 512x512 resolution for optimal speed and memory."""
    model = DetectiveSAM(use_dummy=True, device="cpu")
    assert model.target_size == (512, 512), f"Expected (512, 512), got {model.target_size}"

    # Verify build_model without target_size defaults to (512, 512)
    cfg = ConfigDict({
        "project": {"device": "cpu"},
        "model": {
            "name": "detective_sam",
            "use_dummy": True,
        }
    })
    model_cfg_inst = build_model(cfg)
    assert model_cfg_inst.target_size == (512, 512), f"Expected build_model to use (512, 512), got {model_cfg_inst.target_size}"


def test_sam2_query_slot_parameter_sizes():
    """Verify learnable parameter sizes in the query slot of SAM2's mask decoder."""
    from transformers import Sam2Model
    model = Sam2Model.from_pretrained("facebook/sam2-hiera-tiny")
    mask_decoder = model.mask_decoder

    # Query tokens in mask decoder:
    # 1. mask_tokens: [4, 256] = 1,024 params (4 mask proposal query slots)
    # 2. iou_token: [1, 256] = 256 params (IoU prediction query slot)
    # 3. obj_score_token: [1, 256] = 256 params (object presence query slot)
    assert mask_decoder.mask_tokens.weight.shape == (4, 256)
    assert mask_decoder.mask_tokens.weight.numel() == 1024

    assert mask_decoder.iou_token.weight.shape == (1, 256)
    assert mask_decoder.iou_token.weight.numel() == 256

    assert mask_decoder.obj_score_token.weight.shape == (1, 256)
    assert mask_decoder.obj_score_token.weight.numel() == 256

    total_query_params = (
        mask_decoder.mask_tokens.weight.numel()
        + mask_decoder.iou_token.weight.numel()
        + mask_decoder.obj_score_token.weight.numel()
    )
    assert total_query_params == 1536

