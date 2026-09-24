import pytest
import torch
from sid_unet.losses.bce import BCELoss
from sid_unet.losses.dice import DiceLoss
from sid_unet.losses.focal import FocalLoss
from sid_unet.losses.combined import CombinedMaskLoss
from sid_unet.losses.auxiliary import SIDTotalLoss


def test_bce_loss():
    loss_fn = BCELoss()
    logits = torch.zeros(2, 1, 32, 32)
    targets = torch.zeros(2, 1, 32, 32)
    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)


def test_dice_loss():
    loss_fn = DiceLoss()
    logits = torch.ones(2, 1, 32, 32) * 10.0 # High positive logits -> probs ~ 1
    targets = torch.ones(2, 1, 32, 32)
    loss = loss_fn(logits, targets)
    assert loss.item() < 0.05 # Near zero loss for perfect match


def test_focal_loss():
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.randn(2, 1, 32, 32)
    targets = torch.randint(0, 2, (2, 1, 32, 32)).float()
    loss = loss_fn(logits, targets)
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)


def test_combined_and_total_loss():
    total_loss_fn = SIDTotalLoss(
        mask_loss_type="combined",
        bce_weight=0.5,
        dice_weight=0.5,
        aux_classifier=True,
        aux_weight=0.2,
    )
    mask_logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    class_logits = torch.randn(2, 3, requires_grad=True)
    target_masks = torch.randint(0, 2, (2, 1, 32, 32)).float()
    target_labels = torch.tensor([0, 2], dtype=torch.long)

    loss, metrics_dict = total_loss_fn((mask_logits, class_logits), target_masks, target_labels)
    assert loss.item() > 0.0
    assert "mask_loss" in metrics_dict
    assert "aux_loss" in metrics_dict
    assert "total_loss" in metrics_dict
    assert "focal_loss" in metrics_dict

    loss.backward()
    assert mask_logits.grad is not None
    assert class_logits.grad is not None


def test_combined_mask_loss_with_focal():
    # Combined with BCE, Dice, and Focal
    loss_fn = CombinedMaskLoss(
        loss_type="combined",
        bce_weight=0.5,
        dice_weight=0.5,
        focal_weight=0.5,
    )
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 32, 32)).float()
    loss, metrics = loss_fn(logits, targets)

    assert loss.item() > 0.0
    assert "bce_loss" in metrics
    assert "dice_loss" in metrics
    assert "focal_loss" in metrics
    assert "mask_loss" in metrics
    loss.backward()
    assert logits.grad is not None


def test_combined_mask_loss_without_bce():
    # Combined with BCE completely removed (bce_weight=0.0)
    loss_fn = CombinedMaskLoss(
        loss_type="combined",
        bce_weight=0.0,
        dice_weight=0.5,
        focal_weight=0.5,
    )
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 32, 32)).float()
    loss, metrics = loss_fn(logits, targets)

    assert loss.item() > 0.0
    assert "bce_loss" not in metrics
    assert "dice_loss" in metrics
    assert "focal_loss" in metrics
    assert "mask_loss" in metrics
    loss.backward()
    assert logits.grad is not None


def test_build_loss_focal_weight():
    from sid_unet.losses.auxiliary import build_loss

    cfg = {
        "loss": {
            "mask_loss_type": "combined",
            "bce_weight": 0.0,
            "dice_weight": 0.7,
            "focal_weight": 0.8,
        },
        "model": {"aux_classifier": False},
    }
    loss_mod = build_loss(cfg)
    assert loss_mod.mask_loss_fn.bce_weight == 0.0
    assert loss_mod.mask_loss_fn.dice_weight == 0.7
    assert loss_mod.mask_loss_fn.focal_weight == 0.8


def test_loss_extreme_logits_stability():
    """Verify that extreme logits (which could cause FP16 overflow) remain finite and non-NaN."""
    bce = BCELoss()
    focal = FocalLoss()

    extreme_logits = torch.tensor([-1000.0, -100.0, -50.0, 0.0, 50.0, 100.0, 1000.0], requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

    loss_bce = bce(extreme_logits, targets)
    assert torch.isfinite(loss_bce)
    loss_bce.backward()
    assert torch.all(torch.isfinite(extreme_logits.grad))

    extreme_logits.grad = None
    loss_focal = focal(extreme_logits, targets)
    assert torch.isfinite(loss_focal)
    loss_focal.backward()
    assert torch.all(torch.isfinite(extreme_logits.grad))


def test_focal_alpha_foreground_gradient_weight():
    """Verify that focal alpha=0.75 properly weights minority foreground pixels."""
    focal = FocalLoss(alpha=0.75, gamma=2.0)
    logits = torch.zeros(2, 1, 10, 10, requires_grad=True)
    # 2% foreground, 98% background
    targets = torch.zeros(2, 1, 10, 10)
    targets[:, :, :1, :1] = 1.0  # 2 pixels positive

    loss = focal(logits, targets)
    loss.backward()

    # Logits on positive pixels should receive negative gradient (pulling logits towards positive values)
    pos_grad = logits.grad[targets == 1.0]
    neg_grad = logits.grad[targets == 0.0]
    assert (pos_grad < 0).all()
    # Magnitude of gradient on positive pixels should be significantly larger due to alpha=0.75 vs 0.25
    assert pos_grad.abs().mean() > neg_grad.abs().mean()



