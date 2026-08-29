"""
Binary Cross Entropy Loss with Logits for mask segmentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight != 1.0 else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, 1, H, W) or (B, H, W)
        targets: (B, 1, H, W) or (B, H, W) in {0, 1}
        """
        if logits.shape != targets.shape:
            if logits.dim() == 4 and targets.dim() == 3:
                targets = targets.unsqueeze(1)
            elif logits.dim() == 3 and targets.dim() == 4:
                logits = logits.unsqueeze(1)

        pos_wt = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=pos_wt, reduction=self.reduction
        )
