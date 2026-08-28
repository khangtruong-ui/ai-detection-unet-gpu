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

    loss.backward()
    assert mask_logits.grad is not None
    assert class_logits.grad is not None
