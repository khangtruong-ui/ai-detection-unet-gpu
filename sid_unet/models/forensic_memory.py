"""
Forensic Vector Memory for Detective SAM.
Maintains a CPU-resident vector database of representations extracted from ground-truth
tampered image regions. Supports similarity retrieval, epoch-level observation aggregation,
bounded-memory deterministic consolidation, and surrogate stop-gradient query replacement.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("sid_unet.models.forensic_memory")


def stop_gradient_query_replacement(
    original_query: torch.Tensor,
    retrieved_query: torch.Tensor,
) -> torch.Tensor:
    """
    Apply stop-gradient query replacement:
        Rx = R + (Rt - R).detach()

    Mathematically:
        forward(Rx) = Rt
        backward(dRx/dR) = Identity
        backward(dRx/dRt) = Zero

    Args:
        original_query (torch.Tensor): Original query tensor R of shape [B, ...].
        retrieved_query (torch.Tensor): Retrieved query tensor Rt matching R's shape,
            device, and dtype.

    Returns:
        torch.Tensor: Surrogate query tensor Rx with forward value Rt and autograd path to R.
    """
    if original_query.shape != retrieved_query.shape:
        raise ValueError(
            f"Shape mismatch in stop_gradient_query_replacement: "
            f"original_query={original_query.shape} vs retrieved_query={retrieved_query.shape}"
        )
    if original_query.device != retrieved_query.device:
        raise ValueError(
            f"Device mismatch in stop_gradient_query_replacement: "
            f"original_query on {original_query.device} vs retrieved_query on {retrieved_query.device}"
        )

    # Convert retrieved_query to matching dtype if needed
    if retrieved_query.dtype != original_query.dtype:
        retrieved_query = retrieved_query.to(dtype=original_query.dtype)

    # Core stop-gradient replacement
    # Rx = R + (Rt - R).detach()
    # Forward: Rx = R + Rt - R = Rt
    # Backward: dRx/dR = I, dRx/dRt = 0
    return original_query + (retrieved_query - original_query).detach()


class ForensicVectorMemory:
    """
    CPU-resident Learnable Forensic Vector Memory for Detective SAM.

    Maintains a bounded collection of (key, value) pairs extracted from ground-truth
    tampered image regions during training. Provides similarity retrieval (and randomized
    retrieval for baseline comparison), deterministic consolidation, and warm-up gating.

    Args:
        memory_enabled (bool): Whether memory retrieval and collection are active.
        memory_max_entries (int): Upper bound on stored vectors (default: 5000).
        memory_key_dim (int): Dimensionality of retrieval keys (default: 64).
        memory_value_dim (int): Dimensionality of stored values (default: 64).
        memory_vectors_per_image (int): Max tampered vectors collected per image (default: 16).
        memory_retrieval_k (int): Number of nearest neighbors to retrieve (default: 8).
        memory_similarity_metric (str): Similarity metric: 'cosine', 'l2', or 'dot' (default: 'cosine').
        memory_dedup_threshold (float): Similarity threshold above which candidates are discarded as redundant (default: 0.95).
        memory_consolidation_method (str): Consolidation strategy: 'dedup_greedy', 'reservoir', 'fifo' (default: 'dedup_greedy').
        memory_seed (int): Deterministic random seed for consolidation and sampling (default: 42).
    """

    SUPPORTED_METRICS = ("cosine", "l2", "dot")
    SUPPORTED_CONSOLIDATION_METHODS = ("dedup_greedy", "reservoir", "fifo")

    def __init__(
        self,
        memory_enabled: bool = False,
        memory_max_entries: int = 5000,
        memory_key_dim: int = 64,
        memory_value_dim: Optional[int] = None,
        memory_vectors_per_image: int = 16,
        memory_retrieval_k: int = 8,
        memory_similarity_metric: str = "cosine",
        memory_dedup_threshold: float = 0.95,
        memory_consolidation_method: str = "dedup_greedy",
        memory_seed: int = 42,
    ):
        self.memory_enabled = bool(memory_enabled)
        self.memory_max_entries = int(memory_max_entries)
        self.memory_key_dim = int(memory_key_dim)
        self.memory_value_dim = int(memory_value_dim if memory_value_dim is not None else memory_key_dim)
        self.memory_vectors_per_image = int(memory_vectors_per_image)
        self.memory_retrieval_k = int(memory_retrieval_k)
        self.memory_similarity_metric = str(memory_similarity_metric).lower()
        self.memory_dedup_threshold = float(memory_dedup_threshold)
        self.memory_consolidation_method = str(memory_consolidation_method).lower()
        self.memory_seed = int(memory_seed)

        self._validate_config()

        # Authoritative database stored strictly in CPU RAM
        self._keys: torch.Tensor = torch.empty((0, self.memory_key_dim), dtype=torch.float32, device="cpu")
        self._values: torch.Tensor = torch.empty((0, self.memory_value_dim), dtype=torch.float32, device="cpu")

        # Unconsolidated observations accumulated during current epoch
        self._epoch_observed_keys: List[torch.Tensor] = []
        self._epoch_observed_values: List[torch.Tensor] = []

        logger.info(
            f"Initialized ForensicVectorMemory (enabled={self.memory_enabled}, "
            f"max_entries={self.memory_max_entries}, dim={self.memory_key_dim}, "
            f"k={self.memory_retrieval_k}, metric='{self.memory_similarity_metric}', "
            f"consolidation='{self.memory_consolidation_method}')"
        )

    def _validate_config(self) -> None:
        """Validate configuration settings with informative error messages."""
        if self.memory_max_entries <= 0:
            raise ValueError(f"memory_max_entries must be positive, got {self.memory_max_entries}")
        if self.memory_key_dim <= 0:
            raise ValueError(f"memory_key_dim must be positive, got {self.memory_key_dim}")
        if self.memory_value_dim <= 0:
            raise ValueError(f"memory_value_dim must be positive, got {self.memory_value_dim}")
        if self.memory_vectors_per_image <= 0:
            raise ValueError(f"memory_vectors_per_image must be positive, got {self.memory_vectors_per_image}")
        if self.memory_retrieval_k <= 0:
            raise ValueError(f"memory_retrieval_k must be positive, got {self.memory_retrieval_k}")
        if self.memory_similarity_metric not in self.SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported memory_similarity_metric '{self.memory_similarity_metric}'. "
                f"Supported: {self.SUPPORTED_METRICS}"
            )
        if self.memory_similarity_metric == "cosine" and not (-1.0 <= self.memory_dedup_threshold <= 1.0):
            raise ValueError(
                f"memory_dedup_threshold for cosine must be in [-1, 1], got {self.memory_dedup_threshold}"
            )
        if self.memory_consolidation_method not in self.SUPPORTED_CONSOLIDATION_METHODS:
            raise ValueError(
                f"Unsupported memory_consolidation_method '{self.memory_consolidation_method}'. "
                f"Supported: {self.SUPPORTED_CONSOLIDATION_METHODS}"
            )

    @property
    def entry_count(self) -> int:
        """Current number of authoritative stored entries."""
        return self._keys.shape[0]

    @property
    def is_warmed_up(self) -> bool:
        """
        Warm-up gating policy:
        Memory retrieval is active only when database contains at least memory_retrieval_k entries.
        """
        return self.entry_count >= self.memory_retrieval_k

    def get_warmup_policy_summary(self) -> Dict[str, Any]:
        """Return human-readable warm-up policy report."""
        return {
            "is_warmed_up": self.is_warmed_up,
            "current_entries": self.entry_count,
            "required_k": self.memory_retrieval_k,
            "policy": (
                "Bypass retrieval until at least memory_retrieval_k valid entries have been consolidated."
                if not self.is_warmed_up
                else "Active: memory has sufficient entries for nearest-neighbor search."
            ),
        }

    def get_memory_bytes(self) -> int:
        """Calculate total CPU RAM bytes consumed by stored keys and values."""
        return (self._keys.numel() * self._keys.element_size()) + (
            self._values.numel() * self._values.element_size()
        )

    # =========================================================================
    # Memory Collection (Training Epoch)
    # =========================================================================

    def add_observations(
        self,
        keys: torch.Tensor,
        values: Optional[torch.Tensor] = None,
    ) -> int:
        """
        Add detached observation batch to epoch collection buffer.
        Guarantees tensors are detached from autograd graph and reside in CPU RAM.

        Args:
            keys (torch.Tensor): Forensic feature keys [M, key_dim].
            values (Optional[torch.Tensor]): Forensic feature values [M, value_dim].
                If None, keys are used as values.

        Returns:
            int: Number of observations added.
        """
        if not self.memory_enabled:
            return 0

        if keys.numel() == 0:
            return 0

        if keys.ndim != 2 or keys.shape[1] != self.memory_key_dim:
            raise ValueError(
                f"Keys tensor must be [M, {self.memory_key_dim}], got {keys.shape}"
            )

        if values is None:
            values = keys

        if values.ndim != 2 or values.shape[1] != self.memory_value_dim:
            raise ValueError(
                f"Values tensor must be [M, {self.memory_value_dim}], got {values.shape}"
            )

        if keys.shape[0] != values.shape[0]:
            raise ValueError(
                f"Keys count ({keys.shape[0]}) does not match values count ({values.shape[0]})"
            )

        # Strictly detach from autograd graph and transfer to CPU float32
        keys_cpu = keys.detach().to(device="cpu", dtype=torch.float32)
        values_cpu = values.detach().to(device="cpu", dtype=torch.float32)

        self._epoch_observed_keys.append(keys_cpu)
        self._epoch_observed_values.append(values_cpu)

        return keys_cpu.shape[0]

    def clear_epoch_observations(self) -> None:
        """Clear unconsolidated observations without affecting authoritative memory."""
        self._epoch_observed_keys.clear()
        self._epoch_observed_values.clear()

    # =========================================================================
    # Memory Retrieval
    # =========================================================================

    def retrieve(
        self,
        query: torch.Tensor,
        k: Optional[int] = None,
        random_retrieval: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], float]:
        """
        Retrieve top-k value vectors from CPU memory for given query batch.

        Args:
            query (torch.Tensor): Query representations of shape [B, key_dim].
            k (Optional[int]): Number of neighbors (defaults to memory_retrieval_k).
            random_retrieval (bool): If True, randomly samples k memory entries per query
                (used for Baseline C ablation to isolate semantic retrieval).

        Returns:
            Tuple[retrieved_values, retrieved_keys, latency_seconds]:
                - retrieved_values: [B, k, value_dim] on the query tensor's device & dtype,
                  or None if memory is not warmed up.
                - retrieved_keys: [B, k, key_dim] on the query tensor's device & dtype,
                  or None if memory is not warmed up.
                - latency_seconds: Time taken for CPU search and transfer.
        """
        start_time = time.perf_counter()

        if not self.memory_enabled or not self.is_warmed_up or self.entry_count == 0:
            return None, None, 0.0

        if query.ndim != 2 or query.shape[1] != self.memory_key_dim:
            raise ValueError(
                f"Query tensor must have shape [B, {self.memory_key_dim}], got {query.shape}"
            )

        b_sz = query.shape[0]
        k_req = self.memory_retrieval_k if k is None else int(k)
        k_actual = min(k_req, self.entry_count)

        if k_actual <= 0:
            return None, None, 0.0

        target_device = query.device
        target_dtype = query.dtype

        # Randomized retrieval path (Baseline C ablation)
        if random_retrieval:
            generator = torch.Generator(device="cpu").manual_seed(self.memory_seed)
            # Sample k_actual indices uniformly without replacement per query
            sampled_indices = []
            for _ in range(b_sz):
                perm = torch.randperm(self.entry_count, generator=generator)[:k_actual]
                sampled_indices.append(perm)
            indices = torch.stack(sampled_indices, dim=0)  # [B, k_actual]

            retrieved_vals = self._values[indices].to(device=target_device, dtype=target_dtype)
            retrieved_ks = self._keys[indices].to(device=target_device, dtype=target_dtype)
            elapsed = time.perf_counter() - start_time
            return retrieved_vals, retrieved_ks, elapsed

        # Authoritative Nearest-Neighbor Retrieval on CPU
        query_cpu = query.detach().to(device="cpu", dtype=torch.float32)

        sim_matrix = self._compute_similarity(query_cpu, self._keys)  # [B, N_entries]

        # Top-k selection
        topk_sims, topk_indices = torch.topk(sim_matrix, k=k_actual, dim=1, largest=True, sorted=True)

        retrieved_vals = self._values[topk_indices].to(device=target_device, dtype=target_dtype)
        retrieved_ks = self._keys[topk_indices].to(device=target_device, dtype=target_dtype)

        elapsed = time.perf_counter() - start_time
        return retrieved_vals, retrieved_ks, elapsed

    def _compute_similarity(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity matrix [B, N] between queries [B, D] and keys [N, D] on CPU.
        Safely handles zero-norm vectors without producing NaNs.
        """
        metric = self.memory_similarity_metric

        if metric == "cosine":
            # Normalize with safe handling for zero-norm vectors
            q_norm = torch.linalg.norm(queries, ord=2, dim=1, keepdim=True)
            k_norm = torch.linalg.norm(keys, ord=2, dim=1, keepdim=True)

            q_safe = torch.where(q_norm > 1e-8, queries / q_norm, torch.zeros_like(queries))
            k_safe = torch.where(k_norm > 1e-8, keys / k_norm, torch.zeros_like(keys))

            # Cosine similarity is dot product of L2-normalized vectors
            sim = torch.matmul(q_safe, k_safe.t())  # [B, N]
            return sim

        elif metric == "l2":
            # Negative Euclidean distance: -||q - k||_2
            # ||q - k||^2 = ||q||^2 + ||k||^2 - 2 q.k
            q_sq = torch.sum(queries ** 2, dim=1, keepdim=True)  # [B, 1]
            k_sq = torch.sum(keys ** 2, dim=1, keepdim=True).t()  # [1, N]
            dist_sq = q_sq + k_sq - 2.0 * torch.matmul(queries, keys.t())
            dist_sq = torch.clamp(dist_sq, min=0.0)
            return -torch.sqrt(dist_sq + 1e-8)

        elif metric == "dot":
            return torch.matmul(queries, keys.t())

        else:
            raise ValueError(f"Unknown similarity metric '{metric}'")

    # =========================================================================
    # Epoch-End Memory Consolidation
    # =========================================================================

    def consolidate(self) -> Dict[str, Any]:
        """
        Consolidate new observations from the epoch into the authoritative database.

        Strategy:
        1. Concatenate all epoch observations.
        2. Deduplicate candidates against existing memory using chunked similarity:
           Candidate is dropped if similarity >= memory_dedup_threshold.
        3. Enforce memory_max_entries with deterministic bounded pruning.
        4. Measure and return consolidation statistics.

        Never builds an unchunked N x N pairwise matrix, keeping memory bounded O(N).
        """
        start_time = time.perf_counter()

        if not self._epoch_observed_keys:
            duration = time.perf_counter() - start_time
            return {
                "new_observations": 0,
                "duplicates_removed": 0,
                "capacity_pruned": 0,
                "final_entry_count": self.entry_count,
                "consolidation_duration_sec": duration,
            }

        new_keys = torch.cat(self._epoch_observed_keys, dim=0)
        new_values = torch.cat(self._epoch_observed_values, dim=0)
        num_new = new_keys.shape[0]

        # Reset epoch buffer
        self.clear_epoch_observations()

        # If existing database is empty, initialize directly from new observations
        if self.entry_count == 0:
            kept_keys = []
            kept_values = []
            chunk_size = 512

            for i in range(0, num_new, chunk_size):
                k_chunk = new_keys[i : i + chunk_size]
                v_chunk = new_values[i : i + chunk_size]
                kept_keys.append(k_chunk)
                kept_values.append(v_chunk)

            self._keys = torch.cat(kept_keys, dim=0)
            self._values = torch.cat(kept_values, dim=0)
            duplicates_removed = 0
        else:
            # Chunked deduplication of new candidates against authoritative database
            chunk_size = 512
            accepted_keys = []
            accepted_values = []
            duplicates_removed = 0

            for i in range(0, num_new, chunk_size):
                cand_k = new_keys[i : i + chunk_size]
                cand_v = new_values[i : i + chunk_size]

                # Compute similarity of candidate chunk against existing keys
                sim = self._compute_similarity(cand_k, self._keys)  # [chunk, N_existing]
                max_sim, _ = torch.max(sim, dim=1)

                is_unique = max_sim < self.memory_dedup_threshold
                duplicates_removed += int((~is_unique).sum().item())

                if is_unique.any():
                    accepted_keys.append(cand_k[is_unique])
                    accepted_values.append(cand_v[is_unique])

            if accepted_keys:
                all_new_k = torch.cat(accepted_keys, dim=0)
                all_new_v = torch.cat(accepted_values, dim=0)
                self._keys = torch.cat([self._keys, all_new_k], dim=0)
                self._values = torch.cat([self._values, all_new_v], dim=0)

        # Enforce memory_max_entries capacity limit
        capacity_pruned = 0
        if self.entry_count > self.memory_max_entries:
            capacity_pruned = self.entry_count - self.memory_max_entries

            if self.memory_consolidation_method == "fifo":
                # Keep the newest max_entries
                self._keys = self._keys[-self.memory_max_entries :]
                self._values = self._values[-self.memory_max_entries :]

            elif self.memory_consolidation_method == "reservoir":
                # Deterministic random subsample
                generator = torch.Generator(device="cpu").manual_seed(self.memory_seed)
                keep_indices = torch.randperm(self.entry_count, generator=generator)[: self.memory_max_entries]
                keep_indices = torch.sort(keep_indices)[0]
                self._keys = self._keys[keep_indices]
                self._values = self._values[keep_indices]

            else:
                # Default: 'dedup_greedy' with uniform strided diversity selection
                step = self.entry_count / self.memory_max_entries
                indices = torch.tensor(
                    [int(i * step) for i in range(self.memory_max_entries)],
                    dtype=torch.long,
                    device="cpu",
                )
                self._keys = self._keys[indices]
                self._values = self._values[indices]

        duration = time.perf_counter() - start_time
        stats = {
            "new_observations": num_new,
            "duplicates_removed": duplicates_removed,
            "capacity_pruned": capacity_pruned,
            "final_entry_count": self.entry_count,
            "consolidation_duration_sec": duration,
        }

        logger.info(
            f"Consolidated memory in {duration:.3f}s: +{num_new} observations, "
            f"-{duplicates_removed} duplicates, -{capacity_pruned} pruned -> "
            f"Total entries: {self.entry_count}/{self.memory_max_entries}"
        )
        return stats

    # =========================================================================
    # Serialization & State Management
    # =========================================================================

    def state_dict(self) -> Dict[str, Any]:
        """Export authoritative memory and configuration for checkpointing."""
        return {
            "version": "1.0",
            "keys": self._keys.clone(),
            "values": self._values.clone(),
            "entry_count": self.entry_count,
            "config": {
                "memory_enabled": self.memory_enabled,
                "memory_max_entries": self.memory_max_entries,
                "memory_key_dim": self.memory_key_dim,
                "memory_value_dim": self.memory_value_dim,
                "memory_vectors_per_image": self.memory_vectors_per_image,
                "memory_retrieval_k": self.memory_retrieval_k,
                "memory_similarity_metric": self.memory_similarity_metric,
                "memory_dedup_threshold": self.memory_dedup_threshold,
                "memory_consolidation_method": self.memory_consolidation_method,
                "memory_seed": self.memory_seed,
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any], strict_dims: bool = True) -> None:
        """
        Load stored memory database from checkpoint state dictionary.

        Args:
            state_dict (Dict[str, Any]): Saved state dict.
            strict_dims (bool): If True, validates dimension matching.
        """
        if not isinstance(state_dict, dict) or "keys" not in state_dict:
            raise ValueError("Invalid memory state_dict: missing 'keys' entry")

        loaded_keys = state_dict["keys"].cpu().to(dtype=torch.float32)
        loaded_values = state_dict.get("values", loaded_keys).cpu().to(dtype=torch.float32)

        if strict_dims:
            saved_cfg = state_dict.get("config", {})
            saved_k_dim = saved_cfg.get("memory_key_dim")
            saved_v_dim = saved_cfg.get("memory_value_dim")
            if saved_k_dim is not None and saved_k_dim != self.memory_key_dim:
                raise ValueError(
                    f"Dimension mismatch loading memory keys: expected {self.memory_key_dim}, "
                    f"found {saved_k_dim}"
                )
            if saved_v_dim is not None and saved_v_dim != self.memory_value_dim:
                raise ValueError(
                    f"Dimension mismatch loading memory values: expected {self.memory_value_dim}, "
                    f"found {saved_v_dim}"
                )
            if loaded_keys.shape[0] > 0 and loaded_keys.shape[1] != self.memory_key_dim:
                raise ValueError(
                    f"Dimension mismatch loading memory keys: expected {self.memory_key_dim}, "
                    f"found {loaded_keys.shape[1]}"
                )
            if loaded_values.shape[0] > 0 and loaded_values.shape[1] != self.memory_value_dim:
                raise ValueError(
                    f"Dimension mismatch loading memory values: expected {self.memory_value_dim}, "
                    f"found {loaded_values.shape[1]}"
                )

        self._keys = loaded_keys
        self._values = loaded_values
        self.clear_epoch_observations()

        logger.info(
            f"Loaded {self.entry_count} entries into ForensicVectorMemory "
            f"(dim={self.memory_key_dim}, k={self.memory_retrieval_k})"
        )

    def save(self, filepath: str) -> None:
        """Save memory database to a standalone PyTorch file."""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save(self.state_dict(), filepath)
        logger.info(f"Saved ForensicVectorMemory database to '{filepath}' ({self.entry_count} entries)")

    def load(self, filepath: str) -> None:
        """Load memory database from a standalone PyTorch file."""
        import os
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Memory database file not found: '{filepath}'")
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=True)
        self.load_state_dict(checkpoint)
