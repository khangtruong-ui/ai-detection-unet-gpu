"""
Combined Mask Loss combining BCE, Dice, and/or Focal Loss.
"""

from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn

from sid_unet.losses.bce import BCELoss
from sid_unet.losses.dice import DiceLoss
from sid_unet.losses.focal import FocalLoss


class CombinedMaskLoss(nn.Module):
    """
    Weighted combination of BCE Loss, Soft Dice Loss, and Focal Loss.
    """

    def __init__(
        self,
        loss_type: str = "combined",  # 'bce', 'dice', 'focal', 'combined'
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute mask loss and breakdown dict.
        """
        metrics = {}
        if self.loss_type == "bce":
            loss = self.bce(logits, targets)
            metrics["bce_loss"] = loss.item()
        elif self.loss_type == "dice":
            loss = self.dice(logits, targets)
            metrics["dice_loss"] = loss.item()
        elif self.loss_type == "focal":
            loss = self.focal(logits, targets)
            metrics["focal_loss"] = loss.item()
        elif self.loss_type == "combined":
            loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            has_term = False
            if self.bce_weight > 0:
                l_bce = self.bce(logits, targets)
                loss = loss + self.bce_weight * l_bce
                metrics["bce_loss"] = l_bce.item()
                has_term = True
            if self.dice_weight > 0:
                l_dice = self.dice(logits, targets)
                loss = loss + self.dice_weight * l_dice
                metrics["dice_loss"] = l_dice.item()
                has_term = True
            if self.focal_weight > 0:
                l_focal = self.focal(logits, targets)
                loss = loss + self.focal_weight * l_focal
                metrics["focal_loss"] = l_focal.item()
                has_term = True
            if not has_term:
                l_bce = self.bce(logits, targets)
                loss = l_bce
                metrics["bce_loss"] = l_bce.item()
        else:
            raise ValueError(f"Unknown mask_loss_type: {self.loss_type}")

        metrics["mask_loss"] = loss.item()
        return loss, metrics
