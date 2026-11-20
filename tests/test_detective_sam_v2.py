"""
Unit and integration tests for DetectiveSAMv2 (`Gertlek/DetectiveSAMv2` integration),
L2-SP anti-forgetting anchor regularization, checkpoint round-trip, and stratified
sub-dataset evaluation sampling.
"""

from __future__ import annotations

import pytest
import torch

from sid_unet.dataset.loader import SIDMapDataset
from sid_unet.losses.auxiliary import build_loss
from sid_unet.models.detective_sam_v2 import (
    DetectiveSAMv2,
    SpatialCrossAttentionSharedAdapter,
    StreamEvidenceBuilder,
    TransformerEvidenceMaskAdapter,
    compute_perturbation_params,
    parse_perturbation_types,
)
from sid_unet.models.unet import UNet, build_model
from sid_unet.utils.config import ConfigDict, load_config


def test_parse_perturbations_and_params():
    streams = parse_perturbation_types("gaussian_blur+jpeg_compression+gaussian_noise")
    assert streams == ["gaussian_blur", "jpeg_compression", "gaussian_noise"]
    assert parse_perturbation_types("none") == []

    params = compute_perturbation_params(0.5)
    assert params["blur_sigma"] == pytest.approx(1.0)
    assert params["jpeg_quality"] == 66
    assert params["noise_std"] == pytest.approx(0.1)


def test_detective_sam_v2_modules_forward_backward():
    b, h, w = 2, 128, 128
    unadapted = [
        torch.randn(b, 256, h // 16, w // 16),
        torch.randn(b, 32, h // 4, w // 4),
        torch.randn(b, 64, h // 8, w // 8),
    ]
    streams = [[torch.randn_like(u) for _ in range(3)] for u in unadapted]

    ev_builder = StreamEvidenceBuilder([256, 32, 64], hidden_dim=96, max_streams=3)
    adapter = SpatialCrossAttentionSharedAdapter([256, 32, 64], hidden_dim=96, max_streams=3)
    mask_adapter = TransformerEvidenceMaskAdapter(hidden_dim=96, output_resolution=(h // 4, w // 4))

    evidence = ev_builder(unadapted, streams)
    assert len(evidence) == 3
    adapted = [adapter(streams[i], unadapted[i], i, evidence=evidence[i]) for i in range(3)]
    mask_prompt, coarse_mask, extra = mask_adapter(adapted, streams, unadapted, evidence=evidence)
    assert mask_prompt.shape == (b, 1, h // 4, w // 4)
    assert coarse_mask.shape[0] == b
    assert extra is None

    loss = mask_prompt.sum() + coarse_mask.sum()
    loss.backward()
    assert any(p.grad is not None for p in ev_builder.parameters())
    assert any(p.grad is not None for p in adapter.parameters())
    assert any(p.grad is not None for p in mask_adapter.parameters())


def test_detective_sam_v2_dummy_forward_and_l2sp():
    model = DetectiveSAMv2(
        use_dummy=True,
        target_size=(128, 128),
        l2sp_weight=1.0,
        multiscale_supervision=True,
        device="cpu",
    )
    model.train()
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    assert isinstance(out, dict)
    assert out["mask_logits"].shape == (2, 1, 128, 128)
    assert len(out["multiscale_logits"]) == 1
    # Anchor penalty is 0 immediately after init before optimizer steps
    assert float(out["prior_anchor_loss"].item()) == pytest.approx(0.0, abs=1e-7)

    # Perturb a trainable weight and verify L2-SP penalty becomes positive
    with torch.no_grad():
        for p in model.mask_adapter.parameters():
            if p.requires_grad:
                p.add_(0.1)
                break
    out2 = model(x)
    assert float(out2["prior_anchor_loss"].item()) > 0.0

    # Eval mode returns tensor directly unless return_dict=True
    model.eval()
    logits = model(x)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 1, 128, 128)


def test_detective_sam_v2_trainable_module_selection():
    model_head = DetectiveSAMv2(use_dummy=True, trainable_modules="mask_head", device="cpu")
    trainable_names = [n for n, p in model_head.named_parameters() if p.requires_grad]
    assert len(trainable_names) > 0
    assert all("mask_adapter.mask_head" in n or "mask_adapter.aux_head" in n for n in trainable_names)


def test_detective_sam_v2_configs_and_checkpoint_roundtrip(tmp_path):
    cfg = load_config("configs/experiments/sam2/detective_sam_v2.yaml")
    assert cfg.model.name == "detective_sam_v2"
    cfg.model.use_dummy = True
    cfg.model.target_size = [128, 128]
    model = build_model(cfg)
    assert isinstance(model, DetectiveSAMv2)

    cfg_casia = load_config("configs/experiments/sam2/detective_sam_v2_casia.yaml")
    assert cfg_casia.data.dataset_name == "KhangTruong/Casia-v2.0-Fake"
    assert cfg_casia.model.l2sp_weight > 0

    # Save & reload via UNet.from_checkpoint
    ckpt_path = tmp_path / "detective_sam_v2_test.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": cfg.to_dict()}, ckpt_path)
    reloaded, loaded_cfg = UNet.from_checkpoint(str(ckpt_path), device="cpu", return_config=True)
    assert isinstance(reloaded, DetectiveSAMv2)
    assert loaded_cfg.model.name == "detective_sam_v2"


def test_detective_sam_v2_sid_total_loss_integration():
    cfg = ConfigDict({
        "project": {"device": "cpu"},
        "model": {"name": "detective_sam_v2", "use_dummy": True, "target_size": [128, 128], "l2sp_weight": 0.5, "multiscale_supervision": True, "aux_classifier": False},
        "loss": {"mask_loss_type": "combined", "bce_weight": 0.25, "dice_weight": 0.5, "focal_weight": 0.5, "multiscale_weight": 0.2},
    })
    model = build_model(cfg)
    loss_fn = build_loss(cfg)
    model.train()
    x = torch.randn(2, 3, 128, 128)
    y = (torch.rand(2, 1, 128, 128) > 0.5).float()
    out = model(x)
    loss, loss_dict = loss_fn(out, y)
    assert torch.isfinite(loss)
    assert "prior_anchor_loss" in loss_dict
    assert "multiscale_loss" in loss_dict
    loss.backward()


def test_mock_stratified_samples_per_dataset():
    ds = SIDMapDataset(
        dataset_name="mock",
        split="train",
        max_samples=40,
        samples_per_dataset=5,
        target_image_size=(64, 64),
        seed=42,
    )
    assert len(ds) == 40
    item = ds[0]
    assert item["image"].shape == (3, 64, 64)
