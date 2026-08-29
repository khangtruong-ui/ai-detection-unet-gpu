"""
Binary Focal Loss for hard-example mining in segmentation masks.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, 1, H, W) or (B, H, W)
        targets: (B, 1, H, W) or (B, H, W)
        """
        if logits.shape != targets.shape:
            if logits.dim() == 4 and targets.dim() == 3:
                targets = targets.unsqueeze(1)
            elif logits.dim() == 3 and targets.dim() == 4:
                logits = logits.unsqueeze(1)

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        focal_weight = alpha_t * torch.pow((1.0 - p_t), self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
