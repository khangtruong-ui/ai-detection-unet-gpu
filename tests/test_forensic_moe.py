"""
Unit and integration tests for ForensicMoE Prototype v1, SparseViT, and unified TruFor.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from sid_unet.losses.auxiliary import build_loss
from sid_unet.models import ForensicMoE, SparseViT, TruFor, UNet, build_model
from sid_unet.utils.config import ConfigDict, DEFAULT_CONFIG, deep_merge, load_config


def test_zero_random_init_policy():
    """Random initialization must be rejected unless allow_random_init=True is explicit."""
    with pytest.raises(ValueError, match="allow_random_init"):
        SparseViT(pretrained=False)
    with pytest.raises(ValueError, match="allow_random_init"):
        TruFor(np_out_channels=1, pretrained=False)
    with pytest.raises(ValueError, match="allow_random_init"):
        TruFor(np_out_channels=64, pretrained=False)
    with pytest.raises(ValueError, match="allow_random_init"):
        ForensicMoE(pretrained=False)


def test_sparsevit_and_trufor_forward_shapes():
    """Verify SparseViT and TruFor (1-ch and 64-ch) shapes and return_dict contracts."""
    x_raw = torch.rand(1, 3, 64, 64)

    sv = SparseViT(img_size=64, pretrained=False, allow_random_init=True, freeze=True)
    out_sv = sv(x_raw, return_dict=True)
    assert out_sv["mask_logits"].shape == (1, 1, 64, 64)
    assert out_sv["pred_mask"].shape == (1, 1, 64, 64)
    assert sv(x_raw, return_dict=False).shape == (1, 1, 64, 64)

    for ch in (1, 64):
        tf = TruFor(np_out_channels=ch, pretrained=False, allow_random_init=True, freeze=True)
        out_tf = tf(x_raw, return_dict=True)
        assert out_tf["mask_logits"].shape == (1, 1, 64, 64)
        assert out_tf["raw_logits"].shape == (1, 2, 64, 64)
        assert out_tf["conf_map"].shape == (1, 1, 64, 64)
        assert out_tf["conf_prob"].shape == (1, 1, 64, 64)
        assert out_tf["det_logit"].shape == (1, 1)
        assert (ch == 64) == (tf.np_adapter is not None)


def test_forensic_moe_param_budget_and_frozen_experts():
    """Verify <1.5M trainable params, strictly frozen experts, and train() eval-lock."""
    model = ForensicMoE(img_size=64, pretrained=False, allow_random_init=True)
    trainable = model.get_trainable_param_count()
    total = model.get_total_param_count()

    assert 300_000 <= trainable < 1_500_000, f"Unexpected trainable params: {trainable}"
    assert total > 100_000_000

    for expert in (model.expert_sparsevit, model.expert_trufor, model.expert_diffusionprint):
        assert all(not p.requires_grad for p in expert.parameters())

    # Switching ForensicMoE to train() must keep all 3 frozen experts in eval()
    model.train()
    assert model.gate.training is True
    assert model.expert_sparsevit.training is False
    assert model.expert_trufor.training is False
    assert model.expert_diffusionprint.training is False


def test_forensic_moe_preprocessing_routing_and_scale_invariance():
    """
    Verify raw [0, 1] RGB reaches TruFor / TruFor-DP while ImageNet-standardized RGB
    reaches SparseViT, and both raw and pre-normalized inputs produce identical outputs.
    """
    model = ForensicMoE(img_size=64, pretrained=False, allow_random_init=True)
    model.eval()

    captured: dict[str, torch.Tensor] = {}

    orig_sv = model.expert_sparsevit.forward
    orig_tf = model.expert_trufor.forward
    orig_dp = model.expert_diffusionprint.forward

    def spy_sv(x, *args, **kwargs):
        captured["sv_in"] = x.detach().clone()
        return orig_sv(x, *args, **kwargs)

    def spy_tf(x, *args, **kwargs):
        captured["tf_in"] = x.detach().clone()
        return orig_tf(x, *args, **kwargs)

    def spy_dp(x, *args, **kwargs):
        captured["dp_in"] = x.detach().clone()
        return orig_dp(x, *args, **kwargs)

    model.expert_sparsevit.forward = spy_sv  # type: ignore[method-assign]
    model.expert_trufor.forward = spy_tf  # type: ignore[method-assign]
    model.expert_diffusionprint.forward = spy_dp  # type: ignore[method-assign]

    torch.manual_seed(0)
    x_raw = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        out_raw = model(x_raw, return_dict=True)

    # Experts 2 & 3 must receive [0, 1] raw RGB; Expert 1B must receive ImageNet-normalized RGB
    assert float(captured["tf_in"].min()) >= 0.0 and float(captured["tf_in"].max()) <= 1.0
    assert float(captured["dp_in"].min()) >= 0.0 and float(captured["dp_in"].max()) <= 1.0
    assert float(captured["sv_in"].min()) < -0.5 and float(captured["sv_in"].max()) > 1.2

    # Passing ImageNet-standardized input must trigger _ensure_raw_rgb() and match raw output
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x_norm = (x_raw - mean) / std
    with torch.no_grad():
        out_norm = model(x_norm, return_dict=True)

    assert torch.allclose(out_raw["mask_logits"], out_norm["mask_logits"], atol=1e-5)


def test_forensic_moe_gate_simplex_and_backward_with_sid_loss():
    """Verify gate outputs lie on simplex, spatial init matches alpha_img, and backward works."""
    model = ForensicMoE(img_size=64, pretrained=False, allow_random_init=True)
    model.train()

    x = torch.rand(2, 3, 64, 64)
    gt = torch.zeros(2, 1, 64, 64)
    gt[0, :, 16:40, 16:40] = 1.0

    out = model(x, gt_masks=gt, return_dict=True)
    assert torch.allclose(out["alpha_img"].sum(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(out["spatial_weights"].sum(dim=1), torch.ones(2, 64, 64), atol=1e-5)

    # Zero-initialized spatial_out conv means initial spatial weights equal alpha_img broadcast
    expected_w = out["alpha_img"].view(2, 3, 1, 1).expand(2, 3, 64, 64)
    assert torch.allclose(out["spatial_weights"], expected_w, atol=1e-5)

    cfg = ConfigDict(
        deep_merge(
            DEFAULT_CONFIG,
            {
                "model": {"name": "forensic_moe", "aux_classifier": False},
                "loss": {"mask_loss_type": "combined", "bce_weight": 0.5, "dice_weight": 0.5, "align_weight": 1.0},
            },
        )
    )
    loss_fn = build_loss(cfg)
    loss, metrics = loss_fn(out, gt)
    assert torch.isfinite(loss)
    assert "artifact_loss" in metrics
    loss.backward()

    assert model.gate.logit_gamma.grad is not None
    assert model.gate.logit_beta.grad is not None
    assert model.gate.spatial_out.weight.grad is not None
    assert all(p.grad is None for p in model.expert_sparsevit.parameters())
    assert all(p.grad is None for p in model.expert_trufor.parameters())
    assert all(p.grad is None for p in model.expert_diffusionprint.parameters())


def test_build_model_and_checkpoint_roundtrip_and_yaml_config():
    """Verify build_model(), UNet.from_checkpoint(), and configs/experiments/moe/forensic_moe_v1.yaml."""
    yaml_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "experiments",
        "moe",
        "forensic_moe_v1.yaml",
    )
    yaml_cfg = load_config(yaml_path)
    assert yaml_cfg.model.name == "forensic_moe"
    assert yaml_cfg.data.dataset_name == "KhangTruong/IMD2020-CASIAv2-Joint"
    assert yaml_cfg.data.normalize is False
    assert yaml_cfg.data.balance_domains is True

    test_cfg = ConfigDict(
        deep_merge(
            DEFAULT_CONFIG,
            {
                "data": {"image_size": [64, 64], "normalize": False},
                "model": {
                    "name": "forensic_moe",
                    "pretrained": False,
                    "allow_random_init": True,
                    "img_size": 64,
                },
            },
        )
    )
    model = build_model(test_cfg)
    assert isinstance(model, ForensicMoE)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "moe_ckpt.pt")
        torch.save({"model_state_dict": model.state_dict(), "config": test_cfg.to_dict()}, ckpt_path)
        reloaded = UNet.from_checkpoint(ckpt_path, device="cpu")
        assert isinstance(reloaded, ForensicMoE)
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            y1 = model.eval()(x, return_dict=False)
            y2 = reloaded.eval()(x, return_dict=False)
        assert torch.allclose(y1, y2, atol=1e-5)
