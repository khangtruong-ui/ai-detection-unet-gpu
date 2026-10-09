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
    Bounded by the underlying dataloader length to prevent yielding or claiming iterations
    that do not exist when running on machines with different numbers of GPUs.
    """

    def __init__(self, dataloader: Any, hard_batch_indices: Set[int], anchor_seed: Optional[int] = None):
        self.dataloader = dataloader
        self.hard_batch_indices = set(hard_batch_indices)
        self.anchor_seed = anchor_seed

    def __iter__(self) -> Iterator[Any]:
        if not self.hard_batch_indices:
            return
        if self.anchor_seed is not None:
            torch.manual_seed(self.anchor_seed)

        loader_len = len(self.dataloader) if hasattr(self.dataloader, "__len__") else None
        if loader_len is not None:
            target = sum(1 for idx in self.hard_batch_indices if idx < loader_len)
        else:
            target = len(self.hard_batch_indices)

        if target == 0:
            return

        yielded = 0
        for batch_idx, batch in enumerate(self.dataloader):
            if batch_idx in self.hard_batch_indices:
                yield batch
                yielded += 1
                if yielded >= target:
                    break

    def __len__(self) -> int:
        if hasattr(self.dataloader, "__len__"):
            try:
                loader_len = len(self.dataloader)
                return sum(1 for idx in self.hard_batch_indices if idx < loader_len)
            except Exception:
                pass
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
                from sid_unet.utils.distributed import is_dist_avail_and_initialized, sync_scalar_min
                if is_dist_avail_and_initialized():
                    local_count = len(self.hard_batch_indices)
                    synced_k = sync_scalar_min(local_count)
                    if local_count > synced_k:
                        sorted_idx = sorted(self.hard_batch_indices)
                        self.hard_batch_indices = set(sorted_idx[:synced_k])

                self.is_active_epoch = True
                prev_med_val = float(self.previous_median) if self.previous_median is not None else 0.0
                logger.info(
                    f"⛏️ [HARD MINING] Epoch {epoch}: Active ({cycle_step}/{self.reset_epochs} in cycle). "
                    f"Training on {len(self.hard_batch_indices)} hard batches (previous {self.metric}: "
                    f"{prev_med_val:.4f})."
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

    def get_anchor_epoch(self, epoch: Optional[int] = None) -> int:
        """
        Return the cycle anchor epoch (the full-training epoch where hard batches were mined).
        All epochs within the same cycle share the exact same anchor epoch.
        """
        if not self.enabled:
            return epoch if epoch is not None else self.current_epoch
        ep = epoch if epoch is not None else self.current_epoch
        cycle_len = self.reset_epochs + 1
        cycle_step = (ep - 1) % cycle_len
        return max(1, ep - cycle_step)

    def get_sampler_epoch(self, epoch: int) -> int:
        """
        Return the epoch seed for DistributedSampler.
        During active hard mining epochs, returns the cycle's anchor epoch so the
        shuffled batch permutation matches the full epoch where losses were evaluated.
        When a new cycle begins, returns the new anchor epoch to shuffle anew.
        """
        if self.enabled and self.is_hard_mining_active(epoch):
            return self.get_anchor_epoch(epoch)
        return epoch

    def wrap_dataloader(self, dataloader: DataLoader) -> Union[DataLoader, HardMiningBatchFilter]:
        """
        If hard mining is active for the current epoch and hard batches are available,
        wrap the dataloader to only yield the hard batches. Otherwise, return the original dataloader.
        Validates dataloader length against recorded full-epoch batches to guard against
        multi-GPU topology mismatches.
        """
        if self.enabled and self.is_active_epoch and self.hard_batch_indices:
            if hasattr(dataloader, "__len__"):
                actual_loader_len = len(dataloader)
                # If dataloader length does not match total batches recorded during full epoch
                if self.total_batches_in_full_epoch > 0 and actual_loader_len != self.total_batches_in_full_epoch:
                    logger.warning(
                        f"⚠️ [HARD MINING] Dataloader iteration count ({actual_loader_len}) does not match "
                        f"full epoch batches ({self.total_batches_in_full_epoch}). Disabling hard mining filter for "
                        f"epoch {self.current_epoch} to ensure complete training across all GPUs."
                    )
                    self.is_active_epoch = False
                    self.hard_batch_indices.clear()
                    return dataloader

                # Verify valid indices
                valid_count = sum(1 for idx in self.hard_batch_indices if idx < actual_loader_len)
                if valid_count == 0 and actual_loader_len > 0:
                    logger.warning(
                        f"⚠️ [HARD MINING] None of the {len(self.hard_batch_indices)} hard batch indices fall within "
                        f"current dataloader bounds (0..{actual_loader_len - 1}). Disabling hard mining filter for epoch {self.current_epoch}."
                    )
                    self.is_active_epoch = False
                    self.hard_batch_indices.clear()
                    return dataloader

            base_seed = 42
            if hasattr(self.config, "project") and hasattr(self.config.project, "get"):
                base_seed = int(self.config.project.get("seed", 42))
            elif isinstance(self.config, dict) and "project" in self.config:
                base_seed = int(self.config["project"].get("seed", 42))
            anchor_seed = base_seed + self.get_anchor_epoch(self.current_epoch)
            return HardMiningBatchFilter(dataloader, self.hard_batch_indices, anchor_seed=anchor_seed)
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

            from sid_unet.utils.distributed import is_dist_avail_and_initialized, broadcast_scalar, sync_scalar_min
            if is_dist_avail_and_initialized():
                threshold = broadcast_scalar(threshold, src=0)

            self.previous_median = threshold

            # Mark all batches with loss >= threshold as hard examples
            hard_indices = {
                idx for idx, l in enumerate(self._current_epoch_losses)
                if l >= threshold
            }
            # Fallback in case of degenerate zero or single batch
            if not hard_indices and self.total_batches_in_full_epoch > 0:
                hard_indices = set(range(self.total_batches_in_full_epoch))

            # In distributed mode, ensure all ranks agree on the exact same number of hard batches
            if is_dist_avail_and_initialized():
                local_k = len(hard_indices)
                synced_k = sync_scalar_min(local_k)
                if local_k > synced_k and synced_k > 0:
                    sorted_h = sorted(hard_indices, key=lambda i: self._current_epoch_losses[i], reverse=True)
                    hard_indices = set(sorted_h[:synced_k])

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
        """Serialize hard mining state for checkpointing with version 2 enhanced topology metadata."""
        from sid_unet.utils.distributed import is_dist_avail_and_initialized, get_world_size
        world_size = get_world_size() if is_dist_avail_and_initialized() else 1
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        batch_size = None
        if hasattr(self.config, "data"):
            batch_size = getattr(self.config.data, "batch_size", None)
        elif isinstance(self.config, dict) and "data" in self.config:
            batch_size = self.config["data"].get("batch_size")

        return {
            "version": 2,  # Schema version 2 for forward-compatible multi-GPU checkpointing
            "enabled": self.enabled,
            "metric": self.metric,
            "reset_epochs": self.reset_epochs,
            "current_epoch": self.current_epoch,
            "cycle_epoch": self.cycle_epoch,
            "is_active_epoch": self.is_active_epoch,
            "previous_median": self.previous_median,
            "threshold": self.previous_median,
            "hard_batch_indices": sorted(list(self.hard_batch_indices)),
            "total_batches_in_full_epoch": self.total_batches_in_full_epoch,
            "cycle_anchor_epoch": self.get_anchor_epoch(self.current_epoch),
            "world_size": world_size,
            "num_gpus": num_gpus,
            "per_device_batch_size": batch_size,
            "iterations_per_epoch": self.total_batches_in_full_epoch,
        }

    def load_state_dict(self, state: Dict[str, Any], current_iterations: Optional[int] = None) -> None:
        """
        Restore hard mining state from checkpoint.

        Handles both legacy (v1, untagged) checkpoints and enhanced (v2+) checkpoints:
        - For legacy checkpoints: inspects if the saved iterations (total_batches_in_full_epoch)
          matches the current machine's dataloader iterations.
          * If matches: preserves the hard mining state.
          * If differs (e.g. run on machine with different number of GPUs): ignores the legacy
            hard mining state, resets active status, and runs a full epoch to recalibrate.
        - For v2+ checkpoints: stores and verifies topology metadata (world_size, num_gpus, batch_size,
          iterations_per_epoch). If topology differs, recalibrates with a full epoch cleanly.
        """
        if not state:
            return

        is_legacy = "version" not in state or int(state.get("version", 1)) < 2
        checkpoint_version = int(state.get("version", 1))

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
        if "cycle_anchor_epoch" in state and state["cycle_anchor_epoch"] is not None:
            self.cycle_anchor_epoch = int(state["cycle_anchor_epoch"])

        saved_median = state.get("previous_median", state.get("threshold"))
        self.previous_median = float(saved_median) if saved_median is not None else None

        saved_total_batches = int(state.get("total_batches_in_full_epoch", 0))
        raw_hard_indices = set(int(x) for x in state.get("hard_batch_indices", []))
        was_active = bool(state.get("is_active_epoch", False))

        from sid_unet.utils.distributed import is_dist_avail_and_initialized, get_world_size
        cur_world_size = get_world_size() if is_dist_avail_and_initialized() else 1

        if is_legacy:
            # Legacy checkpoint logic:
            # Check if the number of iterations matches the current machine.
            if current_iterations is not None and saved_total_batches > 0:
                if current_iterations == saved_total_batches:
                    # Matches machine! Keep restored hard mining state
                    self.total_batches_in_full_epoch = saved_total_batches
                    self.hard_batch_indices = raw_hard_indices
                    self.is_active_epoch = was_active
                    logger.info(
                        f"⛏️ [HARD MINING] Legacy checkpoint iteration count ({saved_total_batches}) "
                        f"matches current machine dataloader ({current_iterations}). "
                        f"Preserving hard mining state ({len(self.hard_batch_indices)} hard batches, active={self.is_active_epoch})."
                    )
                else:
                    # Mismatched iterations (e.g. different GPU count or batch size) -> Ignore existing hardmining
                    self.total_batches_in_full_epoch = current_iterations
                    self.hard_batch_indices.clear()
                    self.is_active_epoch = False
                    logger.warning(
                        f"⚠️ [HARD MINING] Legacy checkpoint detected with mismatched iterations: "
                        f"checkpoint recorded {saved_total_batches} iterations, but current machine yields "
                        f"{current_iterations} iterations per epoch (e.g. different number of GPUs). "
                        f"Ignoring legacy hard mining state to prevent GPU/iteration mismatch. "
                        f"Epoch {self.current_epoch} will execute a full training epoch to recalibrate hard batches for this machine."
                    )
            else:
                self.total_batches_in_full_epoch = saved_total_batches
                self.hard_batch_indices = raw_hard_indices
                self.is_active_epoch = was_active
        else:
            # Future checkpoint (version >= 2):
            saved_world_size = int(state.get("world_size", 1))
            saved_iters = int(state.get("iterations_per_epoch", saved_total_batches))

            topology_matches = (cur_world_size == saved_world_size)
            iters_match = (current_iterations is None or current_iterations == saved_iters)

            if topology_matches and iters_match:
                self.total_batches_in_full_epoch = saved_iters
                self.hard_batch_indices = raw_hard_indices
                self.is_active_epoch = was_active
                logger.info(
                    f"⛏️ [HARD MINING] Resumed v{checkpoint_version} checkpoint with matching environment "
                    f"(world_size={cur_world_size}, iters={saved_iters}). "
                    f"Restored {len(self.hard_batch_indices)} hard batches (active={self.is_active_epoch})."
                )
            else:
                self.total_batches_in_full_epoch = current_iterations if current_iterations is not None else saved_iters
                self.hard_batch_indices.clear()
                self.is_active_epoch = False
                logger.warning(
                    f"⚠️ [HARD MINING] Checkpoint topology changed since save (saved world_size={saved_world_size}, "
                    f"iters={saved_iters}; current world_size={cur_world_size}, iters={current_iterations}). "
                    f"Ignoring stale batch indices and running full epoch {self.current_epoch} to recalibrate "
                    f"hard mining specifically for the current GPU count."
                )
