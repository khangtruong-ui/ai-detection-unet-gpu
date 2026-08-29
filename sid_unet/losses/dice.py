"""
Soft Dice Loss for binary segmentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Soft Dice Loss with smooth factor for binary mask segmentation.
    Computes 1 - (2 * intersection + smooth) / (total_predicted + total_target + smooth).
    """

    def __init__(self, smooth: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, 1, H, W) or (B, H, W)
        targets: (B, 1, H, W) or (B, H, W)
        """
        probs = torch.sigmoid(logits)

        # Flatten batch elements across spatial dimensions: (B, N)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1).float()

        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
        dice_loss = 1.0 - dice_score
        return dice_loss.mean()
