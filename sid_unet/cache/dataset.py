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
        if z.dim() < 2:
            # 1D descriptor has no spatial dimensions; transform mask only
            if self.hflip > 0 and random.random() < self.hflip:
                mask = torch.flip(mask, dims=[-1])
            if self.vflip > 0 and random.random() < self.vflip:
                mask = torch.flip(mask, dims=[-2])
            if self.rot90 > 0 and random.random() < self.rot90:
                k = random.choice([1, 2, 3])
                mask = torch.rot90(mask, k=k, dims=[-2, -1])
            return z.contiguous(), mask.contiguous()

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
    latent_h: Optional[int] = None,
    latent_w: Optional[int] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Zero-copy decode of cached tensor from bytes or array into PyTorch tensor.
    Safely reconciles both 3D/4D spatial feature maps [C, H, W] and 1D descriptors [C],
    preventing shape mismatch crashes when metadata defines (C, H, W) for flat vectors.
    """
    if h is None and latent_h is not None:
        h = latent_h
    if w is None and latent_w is not None:
        w = latent_w

    if isinstance(raw_z, torch.Tensor):
        return raw_z.to(dtype=dtype)

    def _safe_reshape_np(arr: np.ndarray) -> np.ndarray:
        if channels is not None:
            if arr.size == channels:
                return arr.reshape((channels,))
            if h is not None and w is not None and h > 0 and w > 0 and arr.size == channels * h * w:
                if h == 1 and w == 1:
                    return arr.reshape((channels,))
                return arr.reshape((channels, h, w))
            if h is not None and w is not None and h > 0 and w > 0 and arr.size % (h * w) == 0:
                return arr.reshape((-1, h, w))
            if arr.size == 1:
                return arr.reshape((1,))
        return arr

    if isinstance(raw_z, (bytes, bytearray)):
        if len(raw_z) == 0:
            c = channels or 1
            sh = (c, h, w) if (h and w) else (c,)
            return torch.zeros(sh, dtype=dtype)
        # Stored as float16 byte buffer
        np_arr = np.frombuffer(raw_z, dtype=np.float16).copy()
        np_arr = _safe_reshape_np(np_arr)
        return torch.from_numpy(np_arr).to(dtype=dtype)

    if isinstance(raw_z, np.ndarray):
        if raw_z.dtype == np.uint8:
            np_arr = np.frombuffer(raw_z.tobytes(), dtype=np.float16).copy()
            np_arr = _safe_reshape_np(np_arr)
            return torch.from_numpy(np_arr).to(dtype=dtype)
        raw_z = _safe_reshape_np(raw_z)
        return torch.from_numpy(raw_z).to(dtype=dtype)

    if isinstance(raw_z, (list, tuple)):
        t = torch.tensor(raw_z, dtype=torch.float16)
        if channels is not None:
            if h is not None and w is not None and h > 0 and w > 0 and t.numel() == channels * h * w:
                t = t.view(channels, h, w)
            elif t.numel() == channels:
                t = t.view(channels)
            elif h is not None and w is not None and h > 0 and w > 0 and t.numel() % (h * w) == 0:
                t = t.view(-1, h, w)
        return t.to(dtype=dtype)

    raise TypeError(f"Unsupported cached tensor type: {type(raw_z)}")


def decode_cached_image(
    raw_img: Any,
    target_size: Optional[Tuple[int, int]] = (256, 256),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Decode image into float tensor of shape (3, H, W) normalized to [0.0, 1.0].
    """
    if isinstance(raw_img, torch.Tensor):
        if raw_img.dim() == 4 and raw_img.shape[0] == 1:
            raw_img = raw_img.squeeze(0)
        return raw_img.to(dtype=dtype)

    if isinstance(raw_img, (bytes, bytearray)):
        try:
            pil_img = Image.open(io.BytesIO(raw_img)).convert("RGB")
            if target_size and pil_img.size != (target_size[1], target_size[0]):
                pil_img = pil_img.resize((target_size[1], target_size[0]), Image.BILINEAR)
            arr = np.array(pil_img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr.transpose(2, 0, 1)).to(dtype=dtype)
        except Exception:
            pass

    if isinstance(raw_img, Image.Image):
        pil_img = raw_img.convert("RGB")
        if target_size and pil_img.size != (target_size[1], target_size[0]):
            pil_img = pil_img.resize((target_size[1], target_size[0]), Image.BILINEAR)
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr.transpose(2, 0, 1)).to(dtype=dtype)

    if isinstance(raw_img, np.ndarray):
        arr = raw_img.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr).to(dtype=dtype)

    raise TypeError(f"Unsupported image format: {type(raw_img)}")


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
            # Fallback if raw byte buffer (e.g. numpy tobytes)
            try:
                h, w = target_size if target_size else (256, 256)
                if len(raw_mask) == h * w:
                    arr = np.frombuffer(raw_mask, dtype=np.uint8).reshape((1, h, w)).astype(np.float32) / 255.0
                    return torch.from_numpy(arr).float()
                elif len(raw_mask) == h * w * 4:
                    arr = np.frombuffer(raw_mask, dtype=np.float32).reshape((1, h, w))
                    return torch.from_numpy(arr).float()
                else:
                    arr = np.frombuffer(raw_mask, dtype=np.uint8).astype(np.float32) / 255.0
                    if arr.size == h * w:
                        return torch.from_numpy(arr.reshape((1, h, w))).float()
                    return torch.from_numpy(arr).unsqueeze(0).float()
            except Exception:
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
        image_provider: Optional[Callable[[int, str], torch.Tensor]] = None,
    ):
        self.parquet_files = [str(f) for f in parquet_files if os.path.exists(str(f))]
        self.transform = transform
        self.target_image_size = target_image_size
        self.expected_channels = expected_channels
        self.max_samples = max_samples
        self.max_cached_tables = max(1, int(max_cached_tables))
        self.image_provider = image_provider

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

        raw_label = row.get("label", 2)
        label = int(raw_label) if raw_label is not None else 2
        img_id = str(row.get("img_id", f"cached_{idx}"))

        if z_tensor.dim() > 1 and (z_tensor.numel() == ch or z_tensor.shape[-2:] == (1, 1)):
            z_tensor = z_tensor.view(-1)

        if z_tensor.dim() == 1:
            # 1D descriptor (e.g. GAP-SAM artifact descriptor gap_r of shape [256])
            if "image" in row and row["image"] is not None:
                img_tensor = decode_cached_image(row["image"], target_size=self.target_image_size)
            elif self.image_provider is not None:
                try:
                    img_tensor = self.image_provider(idx, img_id)
                except TypeError:
                    try:
                        img_tensor = self.image_provider(img_id, self.target_image_size)
                    except TypeError:
                        img_tensor = self.image_provider(img_id)
            else:
                img_tensor = torch.zeros(3, self.target_image_size[0], self.target_image_size[1], dtype=torch.float32)

            if self.transform is not None:
                img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

            return {
                "image": img_tensor,
                "mask": mask_tensor,
                "f_r": z_tensor,
                "gap_r": z_tensor,
                "label": torch.tensor(label, dtype=torch.long),
                "img_id": img_id,
                "is_cached": True,
            }
        else:
            # Spatial latent tensor (e.g. Diffusion-Diff [C, H, W])
            if self.transform is not None:
                z_tensor, mask_tensor = self.transform(z_tensor, mask_tensor)

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
        image_provider: Optional[Callable[[int, str], torch.Tensor]] = None,
    ):
        self.parquet_files = [str(f) for f in parquet_files if os.path.exists(str(f))]
        self.transform = transform
        self.target_image_size = target_image_size
        self.expected_channels = expected_channels
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.shuffle_buffer_size = max(0, int(shuffle_buffer_size))
        self.seed = int(seed)
        self.image_provider = image_provider

        self._total_rows = 0
        for f in self.parquet_files:
            try:
                meta = pq.read_metadata(f)
                self._total_rows += meta.num_rows
            except Exception:
                pass

    def __len__(self) -> int:
        from sid_unet.utils.distributed import is_dist_avail_and_initialized, get_world_size
        world_size = get_world_size() if is_dist_avail_and_initialized() else 1
        total = self._total_rows
        if self.max_samples is not None and self.max_samples > 0:
            total = min(total, self.max_samples)
        return max(1, total // world_size)

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

                if z_tensor.dim() > 1 and (z_tensor.numel() == ch or z_tensor.shape[-2:] == (1, 1)):
                    z_tensor = z_tensor.view(-1)

                if z_tensor.dim() == 1:
                    raw_img_val = table["image"][row_idx].as_py() if "image" in table.column_names else None
                    if raw_img_val is not None:
                        img_tensor = decode_cached_image(raw_img_val, target_size=self.target_image_size)
                    elif self.image_provider is not None:
                        try:
                            img_tensor = self.image_provider(count, img_id_val)
                        except TypeError:
                            try:
                                img_tensor = self.image_provider(img_id_val, self.target_image_size)
                            except TypeError:
                                img_tensor = self.image_provider(img_id_val)
                    else:
                        img_tensor = torch.zeros(3, self.target_image_size[0], self.target_image_size[1], dtype=torch.float32)

                    if self.transform is not None:
                        img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

                    sample = {
                        "image": img_tensor,
                        "mask": mask_tensor,
                        "f_r": z_tensor,
                        "gap_r": z_tensor,
                        "label": torch.tensor(label_val, dtype=torch.long),
                        "img_id": img_id_val,
                        "is_cached": True,
                    }
                else:
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


class CachedIndexedDataset(Dataset):
    """
    High-performance PyTorch Dataset wrapping parquet-dataset-loader's IndexedParquetDataset
    for instant non-blocking random access to pre-computed latent representations.
    Supports DDP sharding via .shard(), multi-worker DataLoaders, and SpatialJointTransform.
    """

    def __init__(
        self,
        pdl_dataset: Any,
        transform: Optional[SpatialJointTransform] = None,
        target_image_size: Tuple[int, int] = (256, 256),
        expected_channels: int = 84,
        max_samples: Optional[int] = None,
        image_provider: Optional[Callable[[int, str], torch.Tensor]] = None,
    ):
        self.pdl_dataset = pdl_dataset
        self.transform = transform
        self.target_image_size = target_image_size
        self.expected_channels = expected_channels
        self.max_samples = max_samples
        self.image_provider = image_provider
        self._len = len(pdl_dataset)
        if max_samples is not None and max_samples > 0:
            self._len = min(self._len, max_samples)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self._len:
            raise IndexError(f"Index {idx} out of bounds for dataset of length {self._len}")
        row = self.pdl_dataset[idx]

        z_raw = row["z_high_dim"]
        ch = int(row.get("channels", self.expected_channels))
        latent_h = int(row.get("latent_h", self.target_image_size[0] // 8))
        latent_w = int(row.get("latent_w", self.target_image_size[1] // 8))

        z_tensor = decode_cached_tensor(z_raw, channels=ch, h=latent_h, w=latent_w)
        mask_tensor = decode_mask_tensor(row["mask"], target_size=self.target_image_size)

        raw_label = row.get("label", 2)
        label = int(raw_label) if raw_label is not None else 2
        img_id = str(row.get("img_id", f"cached_{idx}"))

        if z_tensor.dim() > 1 and (z_tensor.numel() == ch or z_tensor.shape[-2:] == (1, 1)):
            z_tensor = z_tensor.view(-1)

        if z_tensor.dim() == 1:
            if "image" in row and row["image"] is not None:
                img_tensor = decode_cached_image(row["image"], target_size=self.target_image_size)
            elif self.image_provider is not None:
                try:
                    img_tensor = self.image_provider(idx, img_id)
                except TypeError:
                    try:
                        img_tensor = self.image_provider(img_id, self.target_image_size)
                    except TypeError:
                        img_tensor = self.image_provider(img_id)
            else:
                img_tensor = torch.zeros(3, self.target_image_size[0], self.target_image_size[1], dtype=torch.float32)

            if self.transform is not None:
                img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

            return {
                "image": img_tensor,
                "mask": mask_tensor,
                "f_r": z_tensor,
                "gap_r": z_tensor,
                "label": torch.tensor(label, dtype=torch.long),
                "img_id": img_id,
                "is_cached": True,
            }
        else:
            if self.transform is not None:
                z_tensor, mask_tensor = self.transform(z_tensor, mask_tensor)

            return {
                "image": z_tensor,
                "mask": mask_tensor,
                "label": torch.tensor(label, dtype=torch.long),
                "img_id": img_id,
                "is_cached": True,
            }

    def shard(self, num_shards: int, index: int, contiguous: bool = True) -> "CachedIndexedDataset":
        sharded_pdl = self.pdl_dataset.shard(num_shards=num_shards, index=index, contiguous=contiguous)
        sharded_max = None
        if self.max_samples is not None and self.max_samples > 0:
            sharded_max = max(1, self.max_samples // num_shards)
        return CachedIndexedDataset(
            pdl_dataset=sharded_pdl,
            transform=self.transform,
            target_image_size=self.target_image_size,
            expected_channels=self.expected_channels,
            max_samples=sharded_max,
            image_provider=self.image_provider,
        )

    def get_row_group_indices(self) -> List[List[int]]:
        """Return dataset row indices grouped by underlying Parquet row group."""
        if hasattr(self.pdl_dataset, "get_row_group_indices"):
            raw_groups = self.pdl_dataset.get_row_group_indices()
            if self.max_samples is not None and self.max_samples > 0:
                filtered_groups = []
                for g in raw_groups:
                    valid_g = [i for i in g if i < self._len]
                    if valid_g:
                        filtered_groups.append(valid_g)
                return filtered_groups
            return raw_groups
        return [list(range(self._len))]

    def close(self) -> None:
        if hasattr(self.pdl_dataset, "close") and callable(self.pdl_dataset.close):
            try:
                self.pdl_dataset.close()
            except Exception:
                pass

