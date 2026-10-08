"""
Cached Dataset Loaders for High-Dimensional Forensics Tensors.
Loads pre-computed latent representations from local Parquet files or Hugging Face Hub repos,
with zero-copy deserialization and synchronous spatial augmentations.
"""

from __future__ import annotations

import io
import json
import logging
import os
import random
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, IterableDataset
import pyarrow.parquet as pq
import pyarrow as pa

logger = logging.getLogger(__name__)


class SpatialJointTransform:
    """
    Applies synchronous spatial 2D transformations (flips, 90-degree rotations)
    to both the high-dimensional latent representation Z and the ground-truth mask.
    Spatial coordinates are preserved since both tensors share congruent 2D dimensions.
    """

    def __init__(
        self,
        horizontal_flip: float = 0.5,
        vertical_flip: float = 0.0,
        random_rotate90: float = 0.25,
    ):
        self.hflip = float(horizontal_flip)
        self.vflip = float(vertical_flip)
        self.rot90 = float(random_rotate90)

    def __call__(
        self, z: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.hflip > 0 and random.random() < self.hflip:
            z = torch.flip(z, dims=[-1])
            mask = torch.flip(mask, dims=[-1])

        if self.vflip > 0 and random.random() < self.vflip:
            z = torch.flip(z, dims=[-2])
            mask = torch.flip(mask, dims=[-2])

        if self.rot90 > 0 and random.random() < self.rot90:
            k = random.choice([1, 2, 3])
            z = torch.rot90(z, k=k, dims=[-2, -1])
            mask = torch.rot90(mask, k=k, dims=[-2, -1])

        return z.contiguous(), mask.contiguous()


def decode_cached_tensor(
    raw_z: Any,
    channels: Optional[int] = None,
    h: Optional[int] = None,
    w: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Zero-copy decode of cached tensor from bytes or array into PyTorch tensor.
    """
    if isinstance(raw_z, torch.Tensor):
        return raw_z.to(dtype=dtype)

    if isinstance(raw_z, (bytes, bytearray)):
        # Stored as float16 byte buffer
        np_arr = np.frombuffer(raw_z, dtype=np.float16).copy()
        if channels is not None and h is not None and w is not None:
            np_arr = np_arr.reshape((channels, h, w))
        t = torch.from_numpy(np_arr).to(dtype=dtype)
        return t

    if isinstance(raw_z, np.ndarray):
        if raw_z.dtype == np.uint8:
            np_arr = np.frombuffer(raw_z.tobytes(), dtype=np.float16).copy()
            if channels is not None and h is not None and w is not None:
                np_arr = np_arr.reshape((channels, h, w))
            return torch.from_numpy(np_arr).to(dtype=dtype)
        return torch.from_numpy(raw_z).to(dtype=dtype)

    if isinstance(raw_z, (list, tuple)):
        t = torch.tensor(raw_z, dtype=torch.float16)
        if channels is not None and h is not None and w is not None:
            t = t.view(channels, h, w)
        return t.to(dtype=dtype)

    raise TypeError(f"Unsupported cached tensor type: {type(raw_z)}")


def decode_mask_tensor(
    raw_mask: Any,
    target_size: Optional[Tuple[int, int]] = (256, 256),
) -> torch.Tensor:
    """
    Decode mask into float tensor of shape (1, H, W) normalized to [0.0, 1.0].
    """
    if isinstance(raw_mask, torch.Tensor):
        if raw_mask.dim() == 2:
            raw_mask = raw_mask.unsqueeze(0)
        return (raw_mask > 0.5).float()

    if isinstance(raw_mask, (bytes, bytearray)):
        try:
            pil_img = Image.open(io.BytesIO(raw_mask)).convert("L")
            if target_size and pil_img.size != (target_size[1], target_size[0]):
                pil_img = pil_img.resize((target_size[1], target_size[0]), Image.NEAREST)
            arr = np.array(pil_img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0).float()
        except Exception:
            # Fallback if raw bytes
            pass

    if isinstance(raw_mask, Image.Image):
        pil_img = raw_mask.convert("L")
        if target_size and pil_img.size != (target_size[1], target_size[0]):
            pil_img = pil_img.resize((target_size[1], target_size[0]), Image.NEAREST)
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0).float()

    if isinstance(raw_mask, np.ndarray):
        arr = raw_mask.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        t = torch.from_numpy(arr)
        if t.dim() == 2:
            t = t.unsqueeze(0)
        return t.float()

    raise TypeError(f"Unsupported mask format: {type(raw_mask)}")


from collections import OrderedDict


class CachedTensorDataset(Dataset):
    """
    Map-style PyTorch Dataset for loading high-dimensional forensics cached tensors.
    Reads from local Parquet files with fast in-memory indexing and bounded LRU caching.
    """

    def __init__(
        self,
        parquet_files: Sequence[str],
        transform: Optional[SpatialJointTransform] = None,
        target_image_size: Tuple[int, int] = (256, 256),
        expected_channels: int = 84,
        max_samples: Optional[int] = None,
        max_cached_tables: int = 2,
    ):
        self.parquet_files = [str(f) for f in parquet_files if os.path.exists(str(f))]
        self.transform = transform
        self.target_image_size = target_image_size
        self.expected_channels = expected_channels
        self.max_samples = max_samples
        self.max_cached_tables = max(1, int(max_cached_tables))

        if not self.parquet_files:
            raise FileNotFoundError(f"No valid Parquet files found in: {parquet_files}")

        # Index row groups across files
        self._index: List[Tuple[str, int, int]] = []  # (file_path, row_group_idx, row_idx_in_rg)
        self._row_counts: List[int] = []

        total_rows = 0
        for f in self.parquet_files:
            meta = pq.read_metadata(f)
            file_rows = meta.num_rows
            self._row_counts.append(file_rows)
            for rg_idx in range(meta.num_row_groups):
                rg_rows = meta.row_group(rg_idx).num_rows
                for row_in_rg in range(rg_rows):
                    self._index.append((f, rg_idx, row_in_rg))
                    total_rows += 1
                    if self.max_samples is not None and total_rows >= self.max_samples:
                        break
                if self.max_samples is not None and total_rows >= self.max_samples:
                    break
            if self.max_samples is not None and total_rows >= self.max_samples:
                break

        # Bounded LRU table cache per worker process to prevent memory exhaustion
        self._cached_tables: OrderedDict[str, pa.Table] = OrderedDict()

    def __len__(self) -> int:
        return len(self._index)

    def _get_row(self, idx: int) -> Dict[str, Any]:
        file_path, _, row_in_file = self._index[idx]
        if file_path in self._cached_tables:
            table = self._cached_tables[file_path]
            self._cached_tables.move_to_end(file_path)
        else:
            table = pq.read_table(file_path)
            self._cached_tables[file_path] = table
            while len(self._cached_tables) > self.max_cached_tables:
                self._cached_tables.popitem(last=False)

        row_dict = {
            col: table[col][row_in_file].as_py()
            for col in table.column_names
        }
        return row_dict

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self._get_row(idx)

        z_raw = row["z_high_dim"]
        ch = int(row.get("channels", self.expected_channels))
        latent_h = int(row.get("latent_h", self.target_image_size[0] // 8))
        latent_w = int(row.get("latent_w", self.target_image_size[1] // 8))

        z_tensor = decode_cached_tensor(z_raw, channels=ch, h=latent_h, w=latent_w)
        mask_tensor = decode_mask_tensor(row["mask"], target_size=self.target_image_size)

        # Apply spatial augmentations synchronously
        if self.transform is not None:
            z_tensor, mask_tensor = self.transform(z_tensor, mask_tensor)

        raw_label = row.get("label", 2)
        label = int(raw_label) if raw_label is not None else 2
        img_id = str(row.get("img_id", f"cached_{idx}"))

        return {
            "image": z_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "img_id": img_id,
            "is_cached": True,
        }


class CachedStreamingDataset(IterableDataset):
    """
    Streaming IterableDataset for loading cached Parquet shards without loading
    the full dataset into memory.
    Supports multi-worker DataLoaders, DDP sharding, shuffle buffers, and length reporting.
    """

    def __init__(
        self,
        parquet_files: Sequence[str],
        transform: Optional[SpatialJointTransform] = None,
        target_image_size: Tuple[int, int] = (256, 256),
        expected_channels: int = 84,
        max_samples: Optional[int] = None,
        shuffle: bool = False,
        shuffle_buffer_size: int = 0,
        seed: int = 42,
    ):
        self.parquet_files = [str(f) for f in parquet_files if os.path.exists(str(f))]
        self.transform = transform
        self.target_image_size = target_image_size
        self.expected_channels = expected_channels
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.shuffle_buffer_size = max(0, int(shuffle_buffer_size))
        self.seed = int(seed)

        self._total_rows = 0
        for f in self.parquet_files:
            try:
                meta = pq.read_metadata(f)
                self._total_rows += meta.num_rows
            except Exception:
                pass

    def __len__(self) -> int:
        if self.max_samples is not None and self.max_samples > 0:
            return min(self._total_rows, self.max_samples)
        return self._total_rows

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        from sid_unet.utils.distributed import is_dist_avail_and_initialized, get_rank, get_world_size

        is_dist = is_dist_avail_and_initialized()
        rank = get_rank() if is_dist else 0
        world_size = get_world_size() if is_dist else 1

        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        global_worker_id = rank * num_workers + worker_id
        total_global_workers = world_size * num_workers

        files = list(self.parquet_files)
        if self.shuffle:
            rng = random.Random(self.seed + global_worker_id)
            rng.shuffle(files)
        else:
            rng = random.Random(self.seed)

        if total_global_workers > 1:
            assigned_files = [f for idx, f in enumerate(files) if idx % total_global_workers == global_worker_id]
        else:
            assigned_files = files

        count = 0
        buffer: List[Dict[str, Any]] = []

        for f in assigned_files:
            if not os.path.exists(f):
                continue
            table = pq.read_table(f)
            num_rows = table.num_rows
            row_indices = list(range(num_rows))
            if self.shuffle:
                rng.shuffle(row_indices)

            for row_idx in row_indices:
                z_raw = table["z_high_dim"][row_idx].as_py()
                mask_raw = table["mask"][row_idx].as_py()
                ch = int(table["channels"][row_idx].as_py() if "channels" in table.column_names else self.expected_channels)
                latent_h = int(table["latent_h"][row_idx].as_py() if "latent_h" in table.column_names else self.target_image_size[0] // 8)
                latent_w = int(table["latent_w"][row_idx].as_py() if "latent_w" in table.column_names else self.target_image_size[1] // 8)
                label_val = int(table["label"][row_idx].as_py() if "label" in table.column_names else 2)
                img_id_val = str(table["img_id"][row_idx].as_py() if "img_id" in table.column_names else f"stream_{count}")

                z_tensor = decode_cached_tensor(z_raw, channels=ch, h=latent_h, w=latent_w)
                mask_tensor = decode_mask_tensor(mask_raw, target_size=self.target_image_size)

                if self.transform is not None:
                    z_tensor, mask_tensor = self.transform(z_tensor, mask_tensor)

                sample = {
                    "image": z_tensor,
                    "mask": mask_tensor,
                    "label": torch.tensor(label_val, dtype=torch.long),
                    "img_id": img_id_val,
                    "is_cached": True,
                }

                if self.shuffle_buffer_size > 1:
                    buffer.append(sample)
                    if len(buffer) >= self.shuffle_buffer_size:
                        pop_idx = rng.randint(0, len(buffer) - 1)
                        yield buffer.pop(pop_idx)
                        count += 1
                        if self.max_samples is not None and count >= self.max_samples:
                            return
                else:
                    yield sample
                    count += 1
                    if self.max_samples is not None and count >= self.max_samples:
                        return

        if buffer:
            if self.shuffle:
                rng.shuffle(buffer)
            for sample in buffer:
                yield sample
                count += 1
                if self.max_samples is not None and count >= self.max_samples:
                    return
