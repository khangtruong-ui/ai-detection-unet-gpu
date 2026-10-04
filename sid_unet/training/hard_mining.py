"""
Hard Mining module for SID-UNet.

Enables hard example mining starting from epoch 2:
- First epoch trains on all batches (longest).
- Hard batches are identified as those whose loss on the previous full epoch was higher than the median.
- Subsequent epochs only train on these hard examples (same number of iterations).
- After `reset_epochs` (default: 5) epochs of hard mining, the state is forgotten and full training
  repeats just like the first epoch, followed by hard mining again.
- Memory & disk efficient: tracks batch indices without storing redundant tensors or duplicating datasets.
- Works with both Map-style datasets and Streaming IterableDatasets with known or unknown iterations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Set, Union
import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger("sid_unet.training.hard_mining")


class HardMiningBatchFilter:
    """
    Lightweight wrapper around an existing DataLoader or iterator that yields only batches
    marked as hard examples based on their batch index.
    """

    def __init__(self, dataloader: Any, hard_batch_indices: Set[int]):
        self.dataloader = dataloader
        self.hard_batch_indices = set(hard_batch_indices)

    def __iter__(self) -> Iterator[Any]:
        for batch_idx, batch in enumerate(self.dataloader):
            if batch_idx in self.hard_batch_indices:
                yield batch

    def __len__(self) -> int:
        return len(self.hard_batch_indices)

    @property
    def dataset(self) -> Any:
        return getattr(self.dataloader, "dataset", None)

    @property
    def batch_size(self) -> Optional[int]:
        return getattr(self.dataloader, "batch_size", None)

    def close(self) -> None:
        if hasattr(self.dataloader, "close") and callable(self.dataloader.close):
            try:
                self.dataloader.close()
            except Exception:
                pass


class HardMiner:
    """
    Manager for Hard Example Mining across training epochs.

    Logic:
    - Cycle length: (1 + reset_epochs) epochs.
      - Epoch 1 of cycle: Full epoch. Trains on all batches. Records loss per batch.
      - Epoch 2..(reset_epochs+1) of cycle: Hard mining epochs. Only trains on batches whose
        loss was >= median loss in the preceding full epoch.
      - Next epoch: Forgets previous hard selections, repeats full epoch, and repeats cycle.
    """

    def __init__(self, config: Any):
        self.config = config

        # Support configuration in top-level hard_mining or training.hard_mining or training.use_hard_mining
        cfg_hm = getattr(config, "hard_mining", {}) if hasattr(config, "hard_mining") else (
            config.get("hard_mining", {}) if hasattr(config, "get") else {}
        )
        if not isinstance(cfg_hm, dict):
            cfg_hm = cfg_hm.to_dict() if hasattr(cfg_hm, "to_dict") else {}

        training_cfg = getattr(config, "training", {}) if hasattr(config, "training") else (
            config.get("training", {}) if hasattr(config, "get") else {}
        )
        if not isinstance(training_cfg, dict):
            training_cfg = training_cfg.to_dict() if hasattr(training_cfg, "to_dict") else {}

        # Enabled flag
        enabled = cfg_hm.get(
            "enabled",
            cfg_hm.get(
                "use_hard_mining",
                training_cfg.get("use_hard_mining", training_cfg.get("hard_mining", False)),
            ),
        )
        self.enabled = bool(enabled)

        # Metric criterion: default "median"
        self.metric = str(cfg_hm.get("metric", training_cfg.get("hard_mining_metric", "median"))).lower()

        # Reset epochs: default 5
        self.reset_epochs = int(
            cfg_hm.get(
                "reset_epochs",
                cfg_hm.get(
                    "hard_mining_epochs",
                    training_cfg.get("hard_mining_reset_epochs", training_cfg.get("hard_mining_epochs", 5)),
                ),
            )
        )
        if self.reset_epochs < 1:
            self.reset_epochs = 5

        # Active state for current epoch
        self.is_active_epoch: bool = False
        self.current_epoch: int = 1
        self.cycle_epoch: int = 1

        # Historical / mined state
        self.previous_median: Optional[float] = None
        self.hard_batch_indices: Set[int] = set()
        self.total_batches_in_full_epoch: int = 0

        # Loss recorder for current epoch
        self._current_epoch_losses: List[float] = []

    def is_hard_mining_active(self, epoch: int) -> bool:
        """Return True if hard mining is enabled and the current epoch is a filtered hard-mining epoch."""
        if not self.enabled:
            return False
        # Epoch 1 is always full training
        if epoch <= 1:
            return False
        # Cycle length = 1 (full epoch) + reset_epochs (hard mining epochs)
        cycle_len = self.reset_epochs + 1
        cycle_step = (epoch - 1) % cycle_len
        # cycle_step == 0 corresponds to full epoch (repeat like first epoch)
        # cycle_step >= 1 corresponds to hard mining epochs
        return cycle_step > 0

    def start_epoch(self, epoch: int) -> bool:
        """
        Notify HardMiner that an epoch is starting.
        Returns True if this epoch should run in hard-mining mode, False if full training.
        """
        self.current_epoch = epoch
        self._current_epoch_losses = []

        if not self.enabled:
            self.is_active_epoch = False
            return False

        cycle_len = self.reset_epochs + 1
        cycle_step = (epoch - 1) % cycle_len
        self.cycle_epoch = cycle_step + 1

        if cycle_step == 0:
            # Full epoch: forget previous selections and repeat full training
            self.is_active_epoch = False
            self.hard_batch_indices.clear()
            logger.info(
                f"⛏️ [HARD MINING] Epoch {epoch}: Running FULL dataset training (Cycle reset / Epoch 1 of cycle). "
                f"Next {self.reset_epochs} epochs will train on hard examples (loss >= {self.metric})."
            )
        else:
            # Hard mining epoch
            if self.hard_batch_indices:
                self.is_active_epoch = True
                logger.info(
                    f"⛏️ [HARD MINING] Epoch {epoch}: Active ({cycle_step}/{self.reset_epochs} in cycle). "
                    f"Training on {len(self.hard_batch_indices)} hard batches (previous {self.metric}: "
                    f"{self.previous_median:.4f} if self.previous_median is not None else 0.0)."
                )
            else:
                # If no hard batches recorded yet (e.g. resumed directly at epoch > 1), run full epoch
                self.is_active_epoch = False
                logger.info(
                    f"⛏️ [HARD MINING] Epoch {epoch}: No prior batch history found. Running FULL training to bootstrap hard batches."
                )

        return self.is_active_epoch

    @property
    def threshold(self) -> Optional[float]:
        return self.previous_median

    @threshold.setter
    def threshold(self, val: Optional[float]) -> None:
        self.previous_median = val

    def on_epoch_start(self, epoch: int) -> bool:
        return self.start_epoch(epoch)

    def on_epoch_end(self, epoch: int) -> Dict[str, Any]:
        return self.end_epoch(epoch)

    def get_train_dataloader(self, dataloader: Any, epoch: Optional[int] = None) -> Union[DataLoader, HardMiningBatchFilter]:
        if epoch is not None:
            self.start_epoch(epoch)
        return self.wrap_dataloader(dataloader)

    def wrap_dataloader(self, dataloader: DataLoader) -> Union[DataLoader, HardMiningBatchFilter]:
        """
        If hard mining is active for the current epoch and hard batches are available,
        wrap the dataloader to only yield the hard batches. Otherwise, return the original dataloader.
        """
        if self.enabled and self.is_active_epoch and self.hard_batch_indices:
            return HardMiningBatchFilter(dataloader, self.hard_batch_indices)
        return dataloader

    def record_batch_loss(self, batch_idx: int, loss_val: float) -> None:
        """Record loss of a processed batch during training."""
        if not self.enabled:
            return
        if not self.is_active_epoch:
            # During full epochs, record losses to compute threshold and identify hard batches
            self._current_epoch_losses.append(float(loss_val))

    def end_epoch(self, epoch: int) -> Dict[str, Any]:
        """
        Notify HardMiner that an epoch has finished.
        Computes threshold (median) and sets hard batch indices for subsequent epochs.
        """
        summary: Dict[str, Any] = {
            "epoch": epoch,
            "hard_mining_enabled": self.enabled,
            "is_hard_mining_epoch": self.is_active_epoch,
            "hard_batch_count": len(self.hard_batch_indices),
        }

        if not self.enabled:
            return summary

        if not self.is_active_epoch and self._current_epoch_losses:
            # Full epoch finished: compute median and mark hard batches
            losses_arr = np.array(self._current_epoch_losses, dtype=np.float64)
            self.total_batches_in_full_epoch = len(losses_arr)

            if self.metric == "mean":
                threshold = float(np.mean(losses_arr))
            else:
                # Default: median
                threshold = float(np.median(losses_arr))

            self.previous_median = threshold

            # Mark all batches with loss >= threshold as hard examples
            hard_indices = {
                idx for idx, l in enumerate(self._current_epoch_losses)
                if l >= threshold
            }
            # Fallback in case of degenerate zero or single batch
            if not hard_indices and self.total_batches_in_full_epoch > 0:
                hard_indices = set(range(self.total_batches_in_full_epoch))

            self.hard_batch_indices = hard_indices
            summary["median_loss"] = threshold
            summary["hard_batch_count"] = len(hard_indices)
            summary["total_batches"] = self.total_batches_in_full_epoch

            logger.info(
                f"⛏️ [HARD MINING] Epoch {epoch} complete. Computed {self.metric} loss: {threshold:.4f}. "
                f"Marked {len(hard_indices)}/{self.total_batches_in_full_epoch} batches as hard examples. "
                f"Epochs {epoch + 1} to {epoch + self.reset_epochs} will train on these {len(hard_indices)} batches."
            )

        return summary

    def state_dict(self) -> Dict[str, Any]:
        """Serialize hard mining state for checkpointing."""
        return {
            "enabled": self.enabled,
            "metric": self.metric,
            "reset_epochs": self.reset_epochs,
            "current_epoch": self.current_epoch,
            "cycle_epoch": self.cycle_epoch,
            "is_active_epoch": self.is_active_epoch,
            "previous_median": self.previous_median,
            "threshold": self.previous_median,
            "hard_batch_indices": list(self.hard_batch_indices),
            "total_batches_in_full_epoch": self.total_batches_in_full_epoch,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore hard mining state from checkpoint."""
        if not state:
            return
        if "enabled" in state:
            self.enabled = bool(state["enabled"])
        if "metric" in state:
            self.metric = str(state["metric"])
        if "reset_epochs" in state:
            self.reset_epochs = int(state["reset_epochs"])
        if "current_epoch" in state:
            self.current_epoch = int(state["current_epoch"])
        if "cycle_epoch" in state:
            self.cycle_epoch = int(state["cycle_epoch"])
        if "is_active_epoch" in state:
            self.is_active_epoch = bool(state["is_active_epoch"])
        if "previous_median" in state:
            self.previous_median = float(state["previous_median"]) if state["previous_median"] is not None else None
        elif "threshold" in state:
            self.previous_median = float(state["threshold"]) if state["threshold"] is not None else None
        if "hard_batch_indices" in state:
            self.hard_batch_indices = set(int(x) for x in state["hard_batch_indices"])
        if "total_batches_in_full_epoch" in state:
            self.total_batches_in_full_epoch = int(state["total_batches_in_full_epoch"])
