"""
Composite Loss Module computing both Mask Segmentation Loss and 3-Class Auxiliary Classification Loss.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from sid_unet.losses.combined import CombinedMaskLoss


class SIDTotalLoss(nn.Module):
    """
    Computes total training/validation loss:
      Total Loss = Mask_Loss + aux_weight * Auxiliary_Classification_Loss
    """

    def __init__(
        self,
        mask_loss_type: str = "combined",
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        aux_classifier: bool = True,
        aux_loss_type: str = "cross_entropy",
        aux_weight: float = 0.2,
    ):
        super().__init__()
        self.mask_loss_fn = CombinedMaskLoss(
            loss_type=mask_loss_type,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
        )
        self.aux_classifier = aux_classifier
        self.aux_loss_type = aux_loss_type
        self.aux_weight = aux_weight

    def forward(
        self,
        model_output: Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]],
        target_masks: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        model_output: mask_logits or (mask_logits, class_logits)
        target_masks: (B, 1, H, W)
        target_labels: (B,)
        """
        if isinstance(model_output, tuple):
            mask_logits, class_logits = model_output
        else:
            mask_logits = model_output
            class_logits = None

        mask_loss, loss_metrics = self.mask_loss_fn(mask_logits, target_masks)
        total_loss = mask_loss

        if (
            self.aux_classifier
            and class_logits is not None
            and target_labels is not None
            and self.aux_weight > 0
        ):
            aux_loss = F.cross_entropy(class_logits, target_labels)
            total_loss = total_loss + self.aux_weight * aux_loss
            loss_metrics["aux_loss"] = aux_loss.item()
            loss_metrics["total_loss"] = total_loss.item()
        else:
            loss_metrics["total_loss"] = total_loss.item()

        return total_loss, loss_metrics


def build_loss(config: Any) -> SIDTotalLoss:
    """Build total loss module from config dictionary."""
    loss_cfg = config.get("loss", {})
    model_cfg = config.get("model", {})
    return SIDTotalLoss(
        mask_loss_type=loss_cfg.get("mask_loss_type", "combined"),
        bce_weight=float(loss_cfg.get("bce_weight", 0.5)),
        dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
        focal_gamma=float(loss_cfg.get("focal_gamma", 2.0)),
        focal_alpha=float(loss_cfg.get("focal_alpha", 0.25)),
        aux_classifier=bool(model_cfg.get("aux_classifier", True)),
        aux_loss_type=loss_cfg.get("aux_loss_type", "cross_entropy"),
        aux_weight=float(loss_cfg.get("aux_weight", 0.2)),
    )
