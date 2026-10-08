"""
Dataset Cache Manager for Extracting, Sharding, and Progressively Syncing Latent Representations.
Handles batch processing, memory-bounded streaming, sharded Parquet serialization,
and non-blocking asynchronous uploads to Hugging Face Hub dataset repositories.
"""

from __future__ import annotations

import gc
import io
import itertools
import json
import logging
import os
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

from sid_unet.cache.extractor import BaseCacheExtractor, get_extractor_for_model
from sid_unet.dataset.mask_utils import ensure_rgb_image, process_sample_mask
from sid_unet.dataset.loader import load_parquet_or_hf_dataset

logger = logging.getLogger(__name__)

_SENTINEL = object()


class AsyncHubUploader:
    """
    Non-blocking background worker that uploads Parquet shards and metadata to Hugging Face Hub.
    Decouples GPU feature extraction from network uploads so the GPU never idles while uploading.
    """

    def __init__(
        self,
        repo_id: str,
        token: Optional[str] = None,
        max_pending: int = 3,
        delete_local_on_upload: bool = False,
    ):
        self.repo_id = repo_id
        self.token = token
        self.max_pending = max(1, int(max_pending))
        self.delete_local_on_upload = bool(delete_local_on_upload)
        self.queue: queue.Queue = queue.Queue(maxsize=self.max_pending)
        self.stop_event = threading.Event()
        self.uploaded_files: List[str] = []
        self.uploaded_bytes: int = 0
        self.failed_error: Optional[Exception] = None
        self._lock = threading.Lock()

        from huggingface_hub import HfApi
        self.api = HfApi(token=token)
        self._ensure_repo_exists()

        self.worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="hf-hub-async-uploader"
        )
        self.worker_thread.start()

    def _ensure_repo_exists(self) -> None:
        """Create or verify existence of the remote dataset repository."""
        try:
            self.api.create_repo(repo_id=self.repo_id, repo_type="dataset", exist_ok=True)
            logger.info(f"Verified Hugging Face dataset repository: '{self.repo_id}'")
        except Exception as e:
            logger.warning(f"Could not verify or create Hugging Face dataset repo '{self.repo_id}': {e}")

    def _worker_loop(self) -> None:
        """Background thread worker loop consuming upload jobs from the queue."""
        while not self.stop_event.is_set():
            try:
                task = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if task is _SENTINEL:
                self.queue.task_done()
                break

            local_path, rel_repo_path, commit_msg, callback = task
            success = False
            for attempt in range(1, 4):
                try:
                    fsize = os.path.getsize(local_path) if os.path.exists(local_path) else 0
                    logger.info(
                        f"🚀 [AsyncHubUploader] Uploading '{rel_repo_path}' ({fsize / (1024*1024):.2f} MB) "
                        f"to {self.repo_id} (attempt {attempt}/3)..."
                    )
                    self.api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=rel_repo_path,
                        repo_id=self.repo_id,
                        repo_type="dataset",
                        commit_message=commit_msg,
                    )
                    with self._lock:
                        self.uploaded_files.append(rel_repo_path)
                        self.uploaded_bytes += fsize
                    logger.info(
                        f"✅ [AsyncHubUploader] Uploaded '{rel_repo_path}' to "
                        f"https://huggingface.co/datasets/{self.repo_id}"
                    )

                    if self.delete_local_on_upload and os.path.exists(local_path):
                        try:
                            os.remove(local_path)
                            logger.debug(f"Removed local shard after upload: {local_path}")
                        except OSError as rm_err:
                            logger.warning(f"Could not remove local file {local_path}: {rm_err}")

                    if callback:
                        try:
                            callback()
                        except Exception as cb_err:
                            logger.warning(f"Callback error after uploading {rel_repo_path}: {cb_err}")

                    success = True
                    break
                except Exception as upload_err:
                    logger.warning(
                        f"⚠️ [AsyncHubUploader] Upload attempt {attempt}/3 failed for '{rel_repo_path}': {upload_err}"
                    )
                    if attempt < 3:
                        time.sleep(2 ** attempt)
                    else:
                        with self._lock:
                            self.failed_error = upload_err

            self.queue.task_done()

    def submit_upload(
        self,
        local_path: str,
        rel_repo_path: str,
        commit_message: Optional[str] = None,
        callback: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Submit a shard file for asynchronous non-blocking upload.
        Blocks only if the in-flight upload queue is full (exceeds max_pending).
        """
        if self.failed_error is not None:
            raise RuntimeError(f"AsyncHubUploader encountered fatal error: {self.failed_error}")
        msg = commit_message or f"Upload {rel_repo_path}"
        self.queue.put((local_path, rel_repo_path, msg, callback))

    def upload_file_sync(
        self,
        local_path: str,
        rel_repo_path: str,
        commit_message: Optional[str] = None,
    ) -> None:
        """Synchronously upload a file (e.g. metadata or README card)."""
        msg = commit_message or f"Upload {rel_repo_path}"
        try:
            self.api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=rel_repo_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=msg,
            )
            with self._lock:
                if rel_repo_path not in self.uploaded_files:
                    self.uploaded_files.append(rel_repo_path)
        except Exception as e:
            logger.warning(f"Sync upload for '{rel_repo_path}' failed: {e}")

    def wait_all(self) -> None:
        """Block until all queued shard uploads are finished."""
        self.queue.join()
        if self.failed_error is not None:
            raise RuntimeError(f"AsyncHubUploader encountered fatal error: {self.failed_error}")

    def close(self) -> None:
        """Signal background thread to exit and wait for shutdown."""
        self.queue.put(_SENTINEL)
        self.worker_thread.join(timeout=30)


class DatasetCacheManager:
    """
    Manages end-to-end caching of high-dimensional model representations:
    1. Loads source dataset using memory-bounded streaming (IndexedParquetDataset).
    2. Runs feature extractor in batched GPU mode with zero leakage.
    3. Writes sharded Parquet files with columnar arrays and zstd compression.
    4. Progressively and non-blockingly pushes shards to Hugging Face Hub dataset repo.
    5. Supports robust resumption from already-uploaded or local shards.
    """

    def __init__(
        self,
        model: nn.Module,
        source_dataset_name: str = "KhangTruong/COCO-inpainted",
        output_dir: str = "outputs/dataset_cache",
        hf_repo_id: Optional[str] = None,
        batch_size: int = 8,
        samples_per_shard: int = 2000,
        image_size: Tuple[int, int] = (256, 256),
        device: Optional[Union[str, torch.device]] = None,
        fp16: bool = True,
        hf_token: Optional[str] = None,
        resume: bool = True,
        delete_local_on_upload: bool = False,
        push_to_hub: bool = False,
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
        self.resume = bool(resume)
        self.delete_local_on_upload = bool(delete_local_on_upload)
        self.push_to_hub_flag = bool(push_to_hub)

        os.makedirs(self.output_dir, exist_ok=True)

        self.uploader: Optional[AsyncHubUploader] = None
        if self.push_to_hub_flag and self.hf_repo_id:
            logger.info(
                f"📡 Initializing progressive background uploader for Hugging Face Hub: '{self.hf_repo_id}'..."
            )
            self.uploader = AsyncHubUploader(
                repo_id=self.hf_repo_id,
                token=self.hf_token,
                max_pending=3,
                delete_local_on_upload=self.delete_local_on_upload,
            )
            # Upload initial metadata and dataset card upfront
            self._save_metadata(all_shards={}, status="in_progress")
            self._upload_metadata_sync()

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

        # Safely close PIL handles to prevent handle/buffer leaks
        if hasattr(raw_img, "close"):
            try:
                raw_img.close()
            except Exception:
                pass
        if hasattr(raw_mask, "close"):
            try:
                raw_mask.close()
            except Exception:
                pass

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

    def _write_shard_columnar(
        self,
        img_ids: List[str],
        labels: List[int],
        z_bytes: List[bytes],
        mask_bytes: List[bytes],
        split: str,
        shard_idx: int,
        latent_h: int,
        latent_w: int,
    ) -> str:
        """Write collected columnar shard records to a compressed Parquet file."""
        split_dir = os.path.join(self.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        shard_filename = f"{split}-{shard_idx:05d}.parquet"
        shard_path = os.path.join(split_dir, shard_filename)

        num_records = len(img_ids)
        table = pa.Table.from_arrays(
            [
                pa.array(img_ids, type=pa.string()),
                pa.array(labels, type=pa.int64()),
                pa.array(z_bytes, type=pa.binary()),
                pa.array(mask_bytes, type=pa.binary()),
                pa.array([self.extractor.total_channels] * num_records, type=pa.int64()),
                pa.array([latent_h] * num_records, type=pa.int64()),
                pa.array([latent_w] * num_records, type=pa.int64()),
            ],
            names=["img_id", "label", "z_high_dim", "mask", "channels", "latent_h", "latent_w"],
        )

        pq.write_table(table, shard_path, compression="zstd")
        del table
        shard_size_mb = os.path.getsize(shard_path) / (1024 * 1024)
        logger.info(f"💾 Saved Parquet shard: {shard_path} ({num_records} samples, {shard_size_mb:.2f} MB)")
        return shard_path

    def _detect_existing_shards(self, split: str) -> List[str]:
        """Detect already completed shards on Hugging Face Hub repo or in local directory."""
        existing_names = set()

        if self.hf_repo_id and self.uploader:
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=self.hf_token)
                files = api.list_repo_files(repo_id=self.hf_repo_id, repo_type="dataset")
                prefix = f"{split}/"
                for f in files:
                    if f.startswith(prefix) and f.endswith(".parquet"):
                        existing_names.add(os.path.basename(f))
            except Exception as e:
                logger.warning(f"Could not list remote files in {self.hf_repo_id}: {e}")

        # Also inspect local output directory
        split_dir = os.path.join(self.output_dir, split)
        if os.path.exists(split_dir):
            for f in os.listdir(split_dir):
                if f.startswith(f"{split}-") and f.endswith(".parquet"):
                    existing_names.add(f)

        return sorted(list(existing_names))

    def _count_existing_samples(self, split: str, existing_shards: List[str]) -> int:
        """
        Count the actual number of samples across existing shards accurately.
        Avoids false shard-multiplication assumptions when shards are partial or variable sized.
        """
        if not existing_shards:
            return 0

        # 1. First check local parquet shards if present
        local_total = 0
        all_local_found = True
        split_dir = os.path.join(self.output_dir, split)
        for s in existing_shards:
            local_path = os.path.join(split_dir, s)
            if os.path.exists(local_path):
                try:
                    local_total += pq.read_metadata(local_path).num_rows
                except Exception:
                    all_local_found = False
                    break
            else:
                all_local_found = False
                break

        if all_local_found and local_total > 0:
            return local_total

        # 2. If shards are remote on Hugging Face Hub, query remote count via parquet-dataset-loader
        if self.hf_repo_id:
            try:
                import parquet_dataset_loader as pdl
                remote_ds = pdl.load_dataset(
                    path=self.hf_repo_id,
                    split=split,
                    token=self.hf_token,
                )
                remote_len = len(remote_ds)
                if remote_len > 0:
                    return remote_len
            except Exception as e:
                logger.debug(f"Could not read remote dataset row count via pdl for '{self.hf_repo_id}/{split}': {e}")

        # 3. Fallback: inspect individual local shards that exist
        partial_total = 0
        has_at_least_one = False
        for s in existing_shards:
            local_path = os.path.join(split_dir, s)
            if os.path.exists(local_path):
                try:
                    partial_total += pq.read_metadata(local_path).num_rows
                    has_at_least_one = True
                except Exception:
                    pass
        if has_at_least_one:
            return partial_total

        # 4. Fallback estimation only if no metadata can be inspected
        return len(existing_shards) * self.samples_per_shard

    def cache_split(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        resume: Optional[bool] = None,
    ) -> List[str]:
        """
        Extract and save cached representations for a single split.
        Streams non-blockingly, avoids memory leaks, and pushes shards progressively.

        Args:
            split: Dataset split name (e.g. 'train' or 'validation').
            max_samples: Optional limit on the number of samples to process.
            resume: Whether to resume from existing remote/local shards.

        Returns:
            List of generated Parquet file paths.
        """
        should_resume = self.resume if resume is None else bool(resume)
        logger.info(f"🔄 Initializing cache extraction for split '{split}' from '{self.source_dataset_name}'...")

        ds, resolved_split = load_parquet_or_hf_dataset(
            self.source_dataset_name,
            requested_split=split,
            streaming=True,
        )

        total_dataset_len = len(ds) if hasattr(ds, "__len__") else None

        existing_shards = self._detect_existing_shards(split) if should_resume else []
        already_cached_shards = len(existing_shards)
        already_cached_samples = self._count_existing_samples(split, existing_shards) if should_resume else 0

        shard_files: List[str] = [
            os.path.join(self.output_dir, split, f) for f in existing_shards
        ]

        if should_resume and already_cached_shards > 0:
            logger.info(
                f"⏩ [RESUME] Found {already_cached_shards} existing shards for split '{split}' "
                f"({already_cached_samples} actual samples). Resuming from shard index {already_cached_shards:05d}."
            )
            # If dataset is already completely processed, return early
            if total_dataset_len is not None and already_cached_samples >= total_dataset_len:
                logger.info(f"✨ Split '{split}' is already fully cached ({already_cached_shards} shards, {already_cached_samples} samples). Skipping.")
                if hasattr(ds, "close"):
                    ds.close()
                return shard_files

            # Advance the dataset stream
            if hasattr(ds, "skip"):
                ds = ds.skip(already_cached_samples)
            elif hasattr(ds, "iter_from"):
                ds = ds.iter_from(already_cached_samples)
            else:
                ds = itertools.islice(ds, already_cached_samples, None)

        shard_idx = already_cached_shards
        total_processed = already_cached_samples
        target_total = max_samples if max_samples is not None else total_dataset_len

        latent_h = self.image_size[0] // 8
        latent_w = self.image_size[1] // 8

        # Columnar shard buffers to avoid Python dict overhead
        shard_img_ids: List[str] = []
        shard_labels: List[int] = []
        shard_z_bytes: List[bytes] = []
        shard_mask_bytes: List[bytes] = []

        batch_images: List[torch.Tensor] = []
        batch_masks: List[bytes] = []
        batch_labels: List[int] = []
        batch_ids: List[str] = []

        pbar = tqdm(desc=f"Caching [{split}]", total=target_total, initial=total_processed, unit="samples")

        def flush_batch() -> None:
            nonlocal batch_images, batch_masks, batch_labels, batch_ids
            if not batch_images:
                return

            imgs = torch.stack(batch_images, dim=0).to(self.device)
            with torch.no_grad():
                z_batch = self.extractor.extract_batch(imgs)

            z_batch_cpu = z_batch.cpu()
            if self.fp16 and z_batch_cpu.dtype != torch.float16:
                z_batch_cpu = z_batch_cpu.half()

            z_np = z_batch_cpu.numpy()

            for i in range(len(batch_images)):
                shard_img_ids.append(batch_ids[i])
                shard_labels.append(batch_labels[i])
                shard_z_bytes.append(z_np[i].tobytes())
                shard_mask_bytes.append(batch_masks[i])

            del imgs, z_batch, z_batch_cpu, z_np
            batch_images.clear()
            batch_masks.clear()
            batch_labels.clear()
            batch_ids.clear()

        def commit_shard() -> None:
            nonlocal shard_idx
            if not shard_img_ids:
                return

            path = self._write_shard_columnar(
                img_ids=shard_img_ids,
                labels=shard_labels,
                z_bytes=shard_z_bytes,
                mask_bytes=shard_mask_bytes,
                split=split,
                shard_idx=shard_idx,
                latent_h=latent_h,
                latent_w=latent_w,
            )
            shard_files.append(path)
            num_written = len(shard_img_ids)

            shard_img_ids.clear()
            shard_labels.clear()
            shard_z_bytes.clear()
            shard_mask_bytes.clear()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if self.uploader:
                rel_repo_path = f"{split}/{os.path.basename(path)}"
                commit_msg = f"Add {split} shard {shard_idx:05d} ({num_written} samples)"
                self.uploader.submit_upload(
                    local_path=path,
                    rel_repo_path=rel_repo_path,
                    commit_message=commit_msg,
                )

            shard_idx += 1

        try:
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

                    if len(shard_img_ids) >= self.samples_per_shard:
                        commit_shard()

                    if max_samples is not None and total_processed >= max_samples:
                        break
                except Exception as sample_err:
                    logger.warning(f"Skipping corrupt sample: {sample_err}")
                    continue

            flush_batch()
            commit_shard()

        finally:
            if hasattr(ds, "close"):
                try:
                    ds.close()
                except Exception:
                    pass
            pbar.close()

        logger.info(
            f"✅ Completed caching split '{split}': {total_processed} samples across {len(shard_files)} shards."
        )
        return shard_files

    def cache_all_splits(
        self,
        splits: Sequence[str] = ("train", "validation"),
        max_samples_per_split: Optional[int] = None,
        push_to_hub: bool = False,
        resume: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Run cache extraction for all requested splits, write dataset info, and progressively sync to HF Hub.
        """
        all_shards: Dict[str, List[str]] = {}
        for s in splits:
            shards = self.cache_split(
                split=s,
                max_samples=max_samples_per_split,
                resume=resume,
            )
            all_shards[s] = shards
            # Update progressive metadata card after each split
            self._save_metadata(all_shards, status="in_progress")
            if self.uploader:
                self._upload_metadata_sync()

        # Wait for all background in-flight uploads to finish cleanly
        if self.uploader:
            logger.info("⏳ Waiting for all background shard uploads to complete...")
            self.uploader.wait_all()
            logger.info("🎉 All shards successfully uploaded to Hugging Face Hub!")

        # Final metadata save and card commit
        self._save_metadata(all_shards, status="completed")
        if self.uploader:
            self._upload_metadata_sync()
            self.uploader.close()

        return all_shards

    def _save_metadata(self, all_shards: Dict[str, List[str]], status: str = "in_progress") -> None:
        """Write cache_info.json and README.md dataset card."""
        meta_path = os.path.join(self.output_dir, "cache_info.json")
        splits_dict: Dict[str, int] = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                    if isinstance(old_data, dict):
                        splits_dict.update(old_data.get("splits", {}))
            except Exception:
                pass

        # Update with newly provided shards
        for s, files in all_shards.items():
            if files:
                splits_dict[s] = len(files)

        # Retain counts for common splits if already present on remote repo or disk
        for known_s in ["train", "validation"]:
            if known_s not in splits_dict:
                existing = self._detect_existing_shards(known_s)
                if existing:
                    splits_dict[known_s] = len(existing)

        info = {
            "source_dataset": self.source_dataset_name,
            "extractor": self.extractor.get_metadata(),
            "image_size": list(self.image_size),
            "latent_size": [self.image_size[0] // 8, self.image_size[1] // 8],
            "total_channels": self.extractor.total_channels,
            "splits": splits_dict,
            "status": status,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        # Dataset card with formal schema configs for Hugging Face Viewer
        card_content = f"""---
configs:
- config_name: default
  data_files:
  - split: train
    path: "train/*.parquet"
  - split: validation
    path: "validation/*.parquet"
license: apache-2.0
task_categories:
- image-segmentation
tags:
- synthetic-image-detection
- diffusion-diff
- forensic-latents
---

# {os.path.basename(self.hf_repo_id) if self.hf_repo_id else 'COCO-Inpainted-Cache'}

Pre-computed High-Dimensional Forensic Latents for ultra-fast Diffusion-Diff forensics training.

## Dataset Details
- **Source Dataset**: `{self.source_dataset_name}`
- **Model Extractor**: `{self.extractor.model_name}`
- **Cached Channels**: `{self.extractor.total_channels}`
- **Latent Resolution**: `{self.image_size[0] // 8}x{self.image_size[1] // 8}` (input `{self.image_size[0]}x{self.image_size[1]}`)
- **Format**: Parquet with zstd compression
- **Status**: `{status}`

## Schema / Columns
- `img_id` (`string`): Image identifier.
- `label` (`int64`): Class label (`0`: authentic, `1`: fully synthetic, `2`: tampered/inpainted).
- `z_high_dim` (`binary`): Pre-computed {self.extractor.total_channels}-channel fp16 latent tensor bytes.
- `mask` (`binary`): Ground-truth binary manipulation mask encoded as PNG bytes.
- `channels` (`int64`): Total channel count ({self.extractor.total_channels}).
- `latent_h` (`int64`): Latent height ({self.image_size[0] // 8}).
- `latent_w` (`int64`): Latent width ({self.image_size[1] // 8}).

## High-Speed Training with SID-UNet
```bash
sid-train --config configs/experiments/diffusion_diff_minimized/default.yaml \\
          --cached-hf-repo {self.hf_repo_id or 'KhangTruong/COCO-inpainted-cache'}
```
"""
        readme_path = os.path.join(self.output_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(card_content)

    def _upload_metadata_sync(self) -> None:
        """Push README.md and cache_info.json directly to Hugging Face Hub."""
        if not self.uploader:
            return
        meta_path = os.path.join(self.output_dir, "cache_info.json")
        readme_path = os.path.join(self.output_dir, "README.md")

        if os.path.exists(meta_path):
            self.uploader.upload_file_sync(
                local_path=meta_path,
                rel_repo_path="cache_info.json",
                commit_message="Update cache_info metadata",
            )
        if os.path.exists(readme_path):
            self.uploader.upload_file_sync(
                local_path=readme_path,
                rel_repo_path="README.md",
                commit_message="Update dataset card README.md",
            )

    def push_to_hub(self, repo_id: Optional[str] = None) -> None:
        """Upload entire cached dataset folder to Hugging Face Hub dataset repo."""
        target_repo = repo_id or self.hf_repo_id
        if not target_repo:
            raise ValueError("No Hugging Face repository ID provided for upload.")

        logger.info(f"🚀 Pushing cached dataset to Hugging Face Hub: '{target_repo}'...")
        from huggingface_hub import HfApi

        api = HfApi(token=self.hf_token)
        api.create_repo(repo_id=target_repo, repo_type="dataset", exist_ok=True)

        api.upload_folder(
            folder_path=self.output_dir,
            repo_id=target_repo,
            repo_type="dataset",
            commit_message=f"Upload cached forensic latents from {self.extractor.model_name}",
        )
        logger.info(f"🎉 Successfully uploaded cached dataset to https://huggingface.co/datasets/{target_repo}")
