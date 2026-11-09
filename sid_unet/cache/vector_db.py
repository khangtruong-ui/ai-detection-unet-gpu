"""
MaskCLIP Pre-TVCA Vector Database Extractor for `sid-cache`.

Extracts dense pre-TVCA representations (F_s) from MaskCLIP and continuous patch
ground-truth tampered fraction labels from KhangTruong/DEAL300K-30K (or similar datasets),
sampling with a strict 90% forged / 10% authentic ratio up to a bounded capacity
(e.g., 100,000 entries, ~51.6 MB in float16).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger("sid_unet.cache.vector_db")


class MaskCLIPVectorDBBuilder:
    """
    Builds a consolidated MaskCLIP pre-TVCA vector database with float patch labels.

    Args:
        model: MaskCLIP model instance.
        target_entries: Target total entries in vector DB (default: 100,000).
        forged_ratio: Fraction of vectors that must be forged (l > 0, default: 0.9).
        authentic_ratio: Fraction of vectors that must be authentic (l == 0, default: 0.1).
        dtype: Storage data type ("float16", "bfloat16", or "float32").
        max_patches_per_image_forged: Maximum forged patch vectors taken per image (default: 32).
        max_patches_per_image_authentic: Maximum authentic patch vectors taken per image (default: 4).
        seed: Deterministic random seed for patch sampling (default: 42).
    """

    def __init__(
        self,
        model: nn.Module,
        target_entries: int = 100000,
        forged_ratio: float = 0.9,
        authentic_ratio: float = 0.1,
        dtype: str = "float16",
        max_patches_per_image_forged: int = 32,
        max_patches_per_image_authentic: int = 4,
        seed: int = 42,
    ):
        self.model = model
        self.target_entries = int(target_entries)
        self.forged_ratio = float(forged_ratio)
        self.authentic_ratio = float(authentic_ratio)
        self.target_forged = int(round(self.target_entries * self.forged_ratio))
        self.target_authentic = self.target_entries - self.target_forged
        self.dtype_str = str(dtype).lower().strip()
        self.target_dtype = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(self.dtype_str, torch.float16)

        self.max_patches_per_image_forged = int(max_patches_per_image_forged)
        self.max_patches_per_image_authentic = int(max_patches_per_image_authentic)
        self.seed = int(seed)
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)

        # Storage buffers residing on CPU
        self.forged_keys: List[torch.Tensor] = []
        self.forged_labels: List[torch.Tensor] = []
        self.authentic_keys: List[torch.Tensor] = []
        self.authentic_labels: List[torch.Tensor] = []

        self.num_forged = 0
        self.num_authentic = 0

    @property
    def is_complete(self) -> bool:
        """Whether target quotas for both forged and authentic entries have been satisfied."""
        return (self.num_forged >= self.target_forged) and (self.num_authentic >= self.target_authentic)

    @torch.no_grad()
    def process_batch(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[int, int]:
        """
        Process a batch of RGB images and ground-truth masks.

        Args:
            images: Tensor of shape `[B, 3, H, W]` in `[0, 1]`.
            masks: Tensor of shape `[B, 1, H, W]` or `[B, H, W]` with binary or continuous values.

        Returns:
            Tuple[num_forged_added, num_authentic_added]: Count of added vectors from this batch.
        """
        if self.is_complete:
            return 0, 0

        self.model.eval()
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        images = images.to(device)

        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        masks = masks.to(device=device, dtype=torch.float32)

        # 1. Extract pre-TVCA F_s of shape [B, 256, H/4, W/4]
        if hasattr(self.model, "extract_pre_tvca_features"):
            f_s = self.model.extract_pre_tvca_features(images)
        elif hasattr(self.model, "module") and hasattr(self.model.module, "extract_pre_tvca_features"):
            f_s = self.model.module.extract_pre_tvca_features(images)
        else:
            raise NotImplementedError(
                f"Model {type(self.model).__name__} does not implement 'extract_pre_tvca_features'."
            )

        b_sz, c_dim, h_f, w_f = f_s.shape
        # 2. Downsample masks via 4x4 average pooling to match patch coordinates (H/4, W/4)
        # Result is exact continuous fraction of tampered pixels in that patch: [B, 1, H/4, W/4]
        patch_labels = F.avg_pool2d(masks, kernel_size=4, stride=4)
        if patch_labels.shape[2:] != (h_f, w_f):
            patch_labels = F.interpolate(patch_labels, size=(h_f, w_f), mode="area")

        f_s_cpu = f_s.detach().to(device="cpu", dtype=torch.float32)
        patch_labels_cpu = patch_labels.detach().to(device="cpu", dtype=torch.float32)

        added_forged = 0
        added_auth = 0

        for b in range(b_sz):
            if self.is_complete:
                break

            vecs = f_s_cpu[b].permute(1, 2, 0).reshape(-1, c_dim)  # [H_f * W_f, 256]
            lbls = patch_labels_cpu[b, 0].reshape(-1, 1)            # [H_f * W_f, 1]

            forged_idx = (lbls[:, 0] > 0.0).nonzero(as_tuple=True)[0]
            auth_idx = (lbls[:, 0] == 0.0).nonzero(as_tuple=True)[0]

            # Sample forged patches
            if self.num_forged < self.target_forged and len(forged_idx) > 0:
                k_f = min(len(forged_idx), self.max_patches_per_image_forged, self.target_forged - self.num_forged)
                perm_f = torch.randperm(len(forged_idx), generator=self.generator)[:k_f]
                sel_f_idx = forged_idx[perm_f]
                self.forged_keys.append(vecs[sel_f_idx])
                self.forged_labels.append(lbls[sel_f_idx])
                self.num_forged += k_f
                added_forged += k_f

            # Sample authentic patches
            if self.num_authentic < self.target_authentic and len(auth_idx) > 0:
                k_a = min(len(auth_idx), self.max_patches_per_image_authentic, self.target_authentic - self.num_authentic)
                perm_a = torch.randperm(len(auth_idx), generator=self.generator)[:k_a]
                sel_a_idx = auth_idx[perm_a]
                self.authentic_keys.append(vecs[sel_a_idx])
                self.authentic_labels.append(lbls[sel_a_idx])
                self.num_authentic += k_a
                added_auth += k_a

        return added_forged, added_auth

    def consolidate(self) -> Dict[str, Any]:
        """
        Consolidate collected entries into final quantized tensors on CPU.

        Returns:
            Dictionary containing authoritative 'keys', 'values', 'labels', and metadata.
        """
        if self.num_forged == 0 and self.num_authentic == 0:
            raise RuntimeError("Cannot consolidate empty vector database: no entries collected.")

        all_keys = []
        all_labels = []

        if self.forged_keys:
            f_k = torch.cat(self.forged_keys, dim=0)[: self.target_forged]
            f_l = torch.cat(self.forged_labels, dim=0)[: self.target_forged]
            all_keys.append(f_k)
            all_labels.append(f_l)

        if self.authentic_keys:
            a_k = torch.cat(self.authentic_keys, dim=0)[: self.target_authentic]
            a_l = torch.cat(self.authentic_labels, dim=0)[: self.target_authentic]
            all_keys.append(a_k)
            all_labels.append(a_l)

        keys_tensor = torch.cat(all_keys, dim=0).to(dtype=self.target_dtype)
        labels_tensor = torch.cat(all_labels, dim=0).to(dtype=self.target_dtype)

        # Deterministic shuffle so forged and authentic entries are uniformly mixed
        total_n = keys_tensor.shape[0]
        perm = torch.randperm(total_n, generator=self.generator)
        keys_tensor = keys_tensor[perm]
        labels_tensor = labels_tensor[perm]

        actual_forged = int((labels_tensor[:, 0] > 0.0).sum().item())
        actual_auth = int((labels_tensor[:, 0] == 0.0).sum().item())
        forged_pct = (actual_forged / total_n * 100.0) if total_n > 0 else 0.0
        auth_pct = (actual_auth / total_n * 100.0) if total_n > 0 else 0.0

        size_bytes = (keys_tensor.numel() * keys_tensor.element_size()) + (
            labels_tensor.numel() * labels_tensor.element_size()
        )
        size_mb = size_bytes / (1024 * 1024)

        logger.info(
            f"✅ Consolidated MaskCLIP Vector DB: {total_n} entries ({size_mb:.2f} MB, {self.dtype_str}). "
            f"Forged: {actual_forged} ({forged_pct:.1f}%), Authentic: {actual_auth} ({auth_pct:.1f}%)"
        )

        return {
            "version": "1.0",
            "keys": keys_tensor,
            "values": keys_tensor,
            "labels": labels_tensor,
            "entry_count": total_n,
            "config": {
                "memory_enabled": True,
                "memory_max_entries": self.target_entries,
                "memory_key_dim": keys_tensor.shape[1],
                "memory_value_dim": keys_tensor.shape[1],
                "memory_dtype": self.dtype_str,
                "memory_similarity_metric": "cosine",
                "tap_point": "linear_fuse_pre_tvca",
                "target_entries": self.target_entries,
                "forged_ratio": self.forged_ratio,
                "authentic_ratio": self.authentic_ratio,
                "actual_forged": actual_forged,
                "actual_authentic": actual_auth,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

    def save(self, output_path: str) -> str:
        """
        Consolidate and save vector database to a PyTorch file (.pt).

        Args:
            output_path: Path to the target file.

        Returns:
            Absolute path to saved file.
        """
        payload = self.consolidate()
        abs_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        torch.save(payload, abs_path)
        logger.info(f"💾 Saved MaskCLIP Vector Database to: {abs_path}")
        return abs_path


def extract_mask_clip_vector_db_from_loader(
    model: nn.Module,
    loader: Any,
    output_path: str,
    target_entries: int = 100000,
    forged_ratio: float = 0.9,
    authentic_ratio: float = 0.1,
    dtype: str = "float16",
    max_batches: Optional[int] = None,
) -> str:
    """
    Convenience function to extract pre-TVCA vector DB from a DataLoader or PrefetchedBatchLoader.
    """
    builder = MaskCLIPVectorDBBuilder(
        model=model,
        target_entries=target_entries,
        forged_ratio=forged_ratio,
        authentic_ratio=authentic_ratio,
        dtype=dtype,
    )

    batch_count = 0
    pbar = tqdm(total=target_entries, desc="Extracting MaskCLIP Pre-TVCA Vectors")

    for batch in loader:
        if isinstance(batch, dict):
            images = batch["images"]
            masks = batch.get("masks")
        elif isinstance(batch, (tuple, list)):
            images = batch[0]
            masks = batch[1] if len(batch) > 1 else torch.zeros_like(images[:, :1])
        else:
            continue

        if isinstance(masks, list):
            # If masks are list of bytes or arrays from loader, decode to tensors
            from sid_unet.dataset.mask_utils import decode_mask_bytes
            decoded_masks = []
            for m in masks:
                if isinstance(m, bytes):
                    decoded_masks.append(torch.from_numpy(decode_mask_bytes(m)))
                elif isinstance(m, torch.Tensor):
                    decoded_masks.append(m)
                else:
                    decoded_masks.append(torch.zeros(images.shape[2:], dtype=torch.float32))
            masks = torch.stack(decoded_masks, dim=0)

        added_f, added_a = builder.process_batch(images, masks)
        total_curr = builder.num_forged + builder.num_authentic
        pbar.n = min(total_curr, target_entries)
        pbar.set_postfix({
            "forged": f"{builder.num_forged}/{builder.target_forged}",
            "authentic": f"{builder.num_authentic}/{builder.target_authentic}",
        })
        pbar.refresh()

        batch_count += 1
        if builder.is_complete:
            break
        if max_batches is not None and batch_count >= max_batches:
            break

    pbar.close()
    return builder.save(output_path)
