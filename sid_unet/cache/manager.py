"""
Dataset Cache Manager for Extracting, Sharding, and Syncing Latent Representations.
Handles batch processing, sharded Parquet serialization, metadata indexing, and Hugging Face Hub uploads.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

from sid_unet.cache.extractor import BaseCacheExtractor, get_extractor_for_model
from sid_unet.dataset.mask_utils import ensure_rgb_image, process_sample_mask
from sid_unet.utils.distributed import is_main_process

logger = logging.getLogger(__name__)


class DatasetCacheManager:
    """
    Manages end-to-end caching of high-dimensional model representations:
    1. Loads source dataset (streaming or map-style).
    2. Runs feature extractor in batched GPU mode.
    3. Writes sharded Parquet files with zstd compression.
    4. Pushes cached shards and dataset card to Hugging Face Hub.
    """

    def __init__(
        self,
        model: nn.Module,
        source_dataset_name: str = "KhangTruong/COCO-inpainted",
        output_dir: str = "outputs/dataset_cache",
        hf_repo_id: Optional[str] = None,
        batch_size: int = 8,
        samples_per_shard: int = 5000,
        image_size: Tuple[int, int] = (256, 256),
        device: Optional[Union[str, torch.device]] = None,
        fp16: bool = True,
        hf_token: Optional[str] = None,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.extractor = get_extractor_for_model(model, fp16=fp16)
        self.source_dataset_name = source_dataset_name
        self.output_dir = output_dir
        self.hf_repo_id = hf_repo_id
        self.batch_size = max(1, int(batch_size))
        self.samples_per_shard = max(1, int(samples_per_shard))
        self.image_size = tuple(image_size)
        self.fp16 = bool(fp16)
        self.hf_token = hf_token

        os.makedirs(self.output_dir, exist_ok=True)

    def _process_sample_to_tensors(self, sample: Dict[str, Any]) -> Tuple[torch.Tensor, bytes, int, str]:
        """Convert a raw dataset sample dict into an RGB tensor and mask PNG bytes."""
        raw_img = sample.get("image")
        raw_mask = sample.get("mask")
        raw_label = sample.get("label")
        img_id = str(sample.get("img_id", sample.get("id", "")))

        pil_img = ensure_rgb_image(raw_img)
        if pil_img.size != (self.image_size[1], self.image_size[0]):
            pil_img = pil_img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)

        img_arr = np.array(pil_img, dtype=np.float32) / 255.0
        # (3, H, W) in range [0, 1]
        img_tensor = torch.from_numpy(img_arr.transpose(2, 0, 1))

        mask_arr = process_sample_mask(
            mask_input=raw_mask,
            label=int(raw_label) if raw_label is not None else None,
            image_size=(self.image_size[1], self.image_size[0]),
        )
        if mask_arr.shape != (self.image_size[0], self.image_size[1]):
            mask_pil = Image.fromarray((mask_arr * 255).astype(np.uint8)).resize(
                (self.image_size[1], self.image_size[0]), Image.NEAREST
            )
            mask_arr = np.array(mask_pil, dtype=np.float32) / 255.0

        # Encode mask as PNG bytes for compact storage
        mask_pil = Image.fromarray((mask_arr * 255).astype(np.uint8), mode="L")
        buf = io.BytesIO()
        mask_pil.save(buf, format="PNG", optimize=True)
        mask_bytes = buf.getvalue()

        if raw_label is not None:
            label = int(raw_label)
        else:
            m_sum = float(mask_arr.sum())
            total_px = float(mask_arr.size)
            if m_sum == 0.0:
                label = 0
            elif m_sum >= total_px:
                label = 1
            else:
                label = 2

        return img_tensor, mask_bytes, label, img_id

    def _write_shard(
        self,
        shard_data: List[Dict[str, Any]],
        split: str,
        shard_idx: int,
        latent_h: int,
        latent_w: int,
    ) -> str:
        """Write collected shard records to a compressed Parquet file."""
        split_dir = os.path.join(self.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        shard_filename = f"{split}-{shard_idx:05d}.parquet"
        shard_path = os.path.join(split_dir, shard_filename)

        table = pa.Table.from_pydict({
            "img_id": [d["img_id"] for d in shard_data],
            "label": [d["label"] for d in shard_data],
            "z_high_dim": [d["z_bytes"] for d in shard_data],
            "mask": [d["mask_bytes"] for d in shard_data],
            "channels": [self.extractor.total_channels] * len(shard_data),
            "latent_h": [latent_h] * len(shard_data),
            "latent_w": [latent_w] * len(shard_data),
        })

        pq.write_table(table, shard_path, compression="zstd")
        logger.info(f"💾 Saved Parquet shard: {shard_path} ({len(shard_data)} samples, {os.path.getsize(shard_path) / (1024 * 1024):.2f} MB)")
        return shard_path

    def cache_split(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> List[str]:
        """
        Extract and save cached representations for a single split.

        Args:
            split: Dataset split name (e.g. 'train' or 'validation').
            max_samples: Optional limit on the number of samples to process.

        Returns:
            List of generated Parquet file paths.
        """
        from datasets import load_dataset

        logger.info(f"🔄 Starting cache extraction for split '{split}' from '{self.source_dataset_name}'...")
        try:
            ds = load_dataset(self.source_dataset_name, split=split, streaming=True)
        except Exception as e:
            logger.warning(f"Could not load split '{split}' in streaming mode ({e}); attempting non-streaming...")
            ds = load_dataset(self.source_dataset_name, split=split, streaming=False)

        shard_files: List[str] = []
        shard_buffer: List[Dict[str, Any]] = []
        batch_images: List[torch.Tensor] = []
        batch_masks: List[bytes] = []
        batch_labels: List[int] = []
        batch_ids: List[str] = []

        total_processed = 0
        shard_idx = 0
        latent_h, latent_w = self.image_size[0] // 8, self.image_size[1] // 8

        pbar = tqdm(desc=f"Caching [{split}]", total=max_samples, unit="samples")

        def flush_batch():
            nonlocal batch_images, batch_masks, batch_labels, batch_ids, shard_buffer, shard_idx
            if not batch_images:
                return

            imgs = torch.stack(batch_images, dim=0).to(self.device)
            with torch.no_grad():
                z_batch = self.extractor.extract_batch(imgs)

            # Ensure CPU numpy bytes
            z_batch_cpu = z_batch.cpu()
            if self.fp16 and z_batch_cpu.dtype != torch.float16:
                z_batch_cpu = z_batch_cpu.half()

            for i in range(len(batch_images)):
                z_i = z_batch_cpu[i].numpy()
                shard_buffer.append({
                    "img_id": batch_ids[i],
                    "label": batch_labels[i],
                    "z_bytes": z_i.tobytes(),
                    "mask_bytes": batch_masks[i],
                })

            batch_images.clear()
            batch_masks.clear()
            batch_labels.clear()
            batch_ids.clear()

            if len(shard_buffer) >= self.samples_per_shard:
                path = self._write_shard(shard_buffer, split, shard_idx, latent_h, latent_w)
                shard_files.append(path)
                shard_buffer.clear()
                shard_idx += 1

        for raw_sample in ds:
            try:
                img_t, mask_b, lbl, i_id = self._process_sample_to_tensors(raw_sample)
                batch_images.append(img_t)
                batch_masks.append(mask_b)
                batch_labels.append(lbl)
                batch_ids.append(i_id)
                total_processed += 1
                pbar.update(1)

                if len(batch_images) >= self.batch_size:
                    flush_batch()

                if max_samples is not None and total_processed >= max_samples:
                    break
            except Exception as sample_err:
                logger.warning(f"Skipping corrupt sample: {sample_err}")
                continue

        flush_batch()

        # Flush any remaining samples in shard buffer
        if shard_buffer:
            path = self._write_shard(shard_buffer, split, shard_idx, latent_h, latent_w)
            shard_files.append(path)
            shard_buffer.clear()

        pbar.close()
        logger.info(f"✅ Completed caching split '{split}': {total_processed} samples across {len(shard_files)} shards.")
        return shard_files

    def cache_all_splits(
        self,
        splits: Sequence[str] = ("train", "validation"),
        max_samples_per_split: Optional[int] = None,
        push_to_hub: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Run cache extraction for all requested splits, write dataset info, and optionally push to HF Hub.
        """
        all_shards: Dict[str, List[str]] = {}
        for s in splits:
            shards = self.cache_split(split=s, max_samples=max_samples_per_split)
            all_shards[s] = shards

        # Save metadata and dataset card
        self._save_metadata(all_shards)

        if push_to_hub or (self.hf_repo_id and push_to_hub):
            self.push_to_hub()

        return all_shards

    def _save_metadata(self, all_shards: Dict[str, List[str]]) -> None:
        """Write cache_info.json and README.md dataset card."""
        info = {
            "source_dataset": self.source_dataset_name,
            "extractor": self.extractor.get_metadata(),
            "image_size": list(self.image_size),
            "latent_size": [self.image_size[0] // 8, self.image_size[1] // 8],
            "total_channels": self.extractor.total_channels,
            "splits": {s: len(files) for s, files in all_shards.items()},
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        meta_path = os.path.join(self.output_dir, "cache_info.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        # Dataset card
        card_content = f"""---
license: apache-2.0
task_categories:
- image-segmentation
tags:
- synthetic-image-detection
- diffusion-diff
- forensic-latents
---

# {os.path.basename(self.hf_repo_id) if self.hf_repo_id else 'COCO-Inpainted-Cache'}

Pre-computed High-Dimensional Forensic Latents for fast Diffusion-Diff forensics training.

## Dataset Details
- **Source Dataset**: `{self.source_dataset_name}`
- **Model Extractor**: `{self.extractor.model_name}`
- **Cached Channels**: `{self.extractor.total_channels}`
- **Latent Resolution**: `{self.image_size[0] // 8}x{self.image_size[1] // 8}` (input `{self.image_size[0]}x{self.image_size[1]}`)
- **Format**: Parquet with zstd compression

## Usage with SID-UNet
```bash
sid-train --config configs/experiments/diffusion_diff_minimized/default.yaml \\
          --cached-hf-repo {self.hf_repo_id or 'KhangTruong/COCO-inpainted-cache'}
```
"""
        readme_path = os.path.join(self.output_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(card_content)

        logger.info(f"📄 Saved cache metadata to {meta_path} and dataset card to {readme_path}")

    def push_to_hub(self, repo_id: Optional[str] = None) -> None:
        """Upload cached Parquet shards and metadata to Hugging Face Hub dataset repo."""
        target_repo = repo_id or self.hf_repo_id
        if not target_repo:
            raise ValueError("No Hugging Face repository ID provided for upload.")

        logger.info(f"🚀 Pushing cached dataset shards to Hugging Face Hub: '{target_repo}'...")
        from huggingface_hub import HfApi

        api = HfApi(token=self.hf_token)
        api.create_repo(repo_id=target_repo, repo_type="dataset", exist_ok=True)

        api.upload_folder(
            folder_path=self.output_dir,
            repo_id=target_repo,
            repo_type="dataset",
            commit_message=f"Upload cached forensics latents from {self.extractor.model_name}",
        )
        logger.info(f"🎉 Successfully uploaded cached dataset to https://huggingface.co/datasets/{target_repo}")
