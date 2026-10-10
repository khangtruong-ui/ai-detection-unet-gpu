"""
Dataset loading pipeline for SID_Set and common image manipulation datasets (e.g. KhangTruong/IMD2020).
Supports both streaming (IterableDataset) and non-streaming (MapDataset) modes with
robust handling of label 0 (black mask), label 1 (white mask), label 2 (provided mask),
and 2-column image/mask datasets.
"""

from __future__ import annotations

import itertools
import io
import os
import queue
import sys
import threading
import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from datasets import load_dataset as hf_load_dataset

from sid_unet.utils.distributed import is_dist_avail_and_initialized, get_rank, get_world_size

logger = logging.getLogger(__name__)

_orig_hf_load_dataset = hf_load_dataset

try:
    import parquet_dataset_loader as pdl
    from parquet_dataset_loader import IndexedParquetDataset, ParquetDatasetError
    _HAS_PARQUET_LOADER = True
except ImportError:
    pdl = None  # type: ignore
    IndexedParquetDataset = None  # type: ignore
    ParquetDatasetError = Exception  # type: ignore
    _HAS_PARQUET_LOADER = False

# Ensure PIL loads truncated/partial images without raising OSError
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Optimize HuggingFace filesystem reads for large Parquet datasets (e.g. KhangTruong/COCO-inpainted).
# PyArrow's interleaved column reads can thrash fsspec's default readahead cache (5MB),
# resulting in 50k+ HTTP range requests and freezes per row group transition.
# Using blockcache with 16MB blocks and maxblocks=64 caches row groups effectively
# while avoiding network latency stalls.
try:
    from huggingface_hub.hf_file_system import HfFileSystemFile
    _orig_hf_file_init = HfFileSystemFile.__init__

    def _fast_hf_file_init(self, *args, **kwargs):
        kwargs.setdefault("cache_type", "blockcache")
        kwargs.setdefault("block_size", 16 * 1024 * 1024)
        if kwargs.get("cache_options") is None:
            kwargs["cache_options"] = {"maxblocks": 64}
        elif isinstance(kwargs["cache_options"], dict):
            kwargs["cache_options"].setdefault("maxblocks", 64)
            kwargs["cache_options"].pop("nblocks", None)
        return _orig_hf_file_init(self, *args, **kwargs)

    HfFileSystemFile.__init__ = _fast_hf_file_init
except Exception:
    pass

# Optimize PyArrow fragment scanning for streaming Parquet datasets (e.g. KhangTruong/COCO-inpainted).
# HuggingFace datasets hardcodes batch_readahead=0 and fragment_readahead=0, which forces synchronous
# blocking network I/O every row group transition (e.g. every 23 iterations with batch size 8).
# Setting batch_readahead=2 and fragment_readahead=1 with use_threads=True enables PyArrow's C++ worker
# threads to pre-read and decompress upcoming row groups concurrently while Python trains on the current batch.
# We use datasets' xopen to correctly resolve both local filesystem paths and remote hf:// URIs without FileNotFoundError.
try:
    from datasets.packaged_modules.parquet import parquet as _hf_parquet
    from datasets.utils.file_utils import xopen as _hf_xopen
    import pyarrow as _pa
    import pyarrow.dataset as _ds
    import pyarrow.parquet as _pq
    from packaging import version as _pkg_version
    import gc as _gc

    _orig_hf_generate_tables = _hf_parquet.Parquet._generate_tables

    def _fast_hf_generate_tables(self, files, row_groups_list):
        if self.config.features is not None and self.config.columns is not None:
            if sorted(field.name for field in self.info.features.arrow_schema) != sorted(self.config.columns):
                raise ValueError(
                    f"Tried to load parquet data with columns '{self.config.columns}' with mismatching features '{self.info.features}'"
                )
        filter_expr = (
            _pq.filters_to_expression(self.config.filters)
            if isinstance(self.config.filters, list)
            else self.config.filters
        )
        parquet_file_format = _ds.ParquetFileFormat(default_fragment_scan_options=self.config.fragment_scan_options)
        for file_idx, (file, row_groups) in enumerate(zip(files, row_groups_list)):
            try:
                with _hf_xopen(file, "rb") as f:
                    parquet_fragment = parquet_file_format.make_fragment(f)
                    fragment_is_closed = False
                    try:
                        if row_groups is not None:
                            parquet_fragment = parquet_fragment.subset(row_group_ids=row_groups)
                        if parquet_fragment.row_groups:
                            batch_size = self.config.batch_size or parquet_fragment.row_groups[0].num_rows
                            for batch_idx, record_batch in enumerate(
                                parquet_fragment.to_batches(
                                    batch_size=batch_size,
                                    columns=self.config.columns,
                                    filter=filter_expr,
                                    batch_readahead=2,
                                    fragment_readahead=1,
                                    use_threads=True,
                                )
                            ):
                                pa_table = _pa.Table.from_batches([record_batch])
                                yield _hf_parquet.Key(file_idx, batch_idx), self._cast_table(pa_table)
                            fragment_is_closed = True
                    finally:
                        if not fragment_is_closed and _hf_parquet.datasets.config.PYARROW_VERSION <= _pkg_version.parse("24.0.0"):
                            del parquet_fragment
                            _gc.collect()
            except Exception as e:
                _log = getattr(_hf_parquet, "logger", None)
                if _log is not None:
                    if self.config.on_bad_files == "error":
                        _log.error(f"Failed to read file '{file}' with error {type(e).__name__}: {e}")
                    elif self.config.on_bad_files == "warn":
                        _log.warning(f"Skipping bad file '{file}'. {type(e).__name__}: {e}")
                    else:
                        _log.debug(f"Skipping bad file '{file}'. {type(e).__name__}: {e}")
                if self.config.on_bad_files == "error":
                    raise

    _hf_parquet.Parquet._generate_tables = _fast_hf_generate_tables
except Exception:
    pass




def worker_init_fn(worker_id: int) -> None:
    """Worker initialization function to configure PIL for DataLoader workers."""
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

from sid_unet.dataset.mask_utils import ensure_rgb_image, process_sample_mask, check_image_mask_mismatch
from sid_unet.dataset.transforms import get_transforms, JointCompose


_SENTINEL = object()


class _ExceptionWrapper:
    """Wraps an exception occurring in the prefetch thread to re-raise in the main thread."""

    def __init__(self, exc: Exception):
        self.exc = exc
        self.exc_info = sys.exc_info()

    def reraise(self) -> None:
        if self.exc_info[1] is not None:
            raise self.exc.with_traceback(self.exc_info[2])
        raise self.exc


class _BackgroundPrefetchIterator:
    """
    Iterator running DataLoader consumption in a dedicated background thread.
    Allows GPU computation and Parquet HTTP/disk streaming to overlap concurrently,
    while polling with a short timeout to catch KeyboardInterrupt (Ctrl+C) instantly.
    """

    def __init__(self, loader: Any, maxsize: int = 32):
        self.loader = loader
        self.maxsize = maxsize
        self.queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._fetch_loop, daemon=True)
        self._worker.start()

    def _fetch_loop(self) -> None:
        try:
            for item in self.loader:
                while not self._stop_event.is_set():
                    try:
                        self.queue.put(item, timeout=0.1)
                        break
                    except Exception:
                        continue
                if self._stop_event.is_set():
                    break
        except Exception as e:
            if not self._stop_event.is_set():
                try:
                    self.queue.put(_ExceptionWrapper(e), timeout=0.1)
                except Exception:
                    pass
        finally:
            while not self._stop_event.is_set():
                try:
                    self.queue.put(_SENTINEL, timeout=0.1)
                    break
                except Exception:
                    continue

    def __iter__(self) -> _BackgroundPrefetchIterator:
        return self

    def __next__(self) -> Any:
        while not self._stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.2)
            except Exception:
                if not self._worker.is_alive() and (self.queue is None or self.queue.empty()):
                    self.close()
                    raise StopIteration
                continue

            if item is _SENTINEL:
                self.close()
                raise StopIteration
            if isinstance(item, _ExceptionWrapper):
                self.close()
                item.reraise()
            return item
        raise StopIteration

    def close(self) -> None:
        """Signal worker to stop, drain queue, and close underlying dataset stream."""
        if getattr(self, "_stop_event", None) is not None:
            self._stop_event.set()
        q = getattr(self, "queue", None)
        if q is not None:
            while not q.empty():
                try:
                    q.get_nowait()
                except Exception:
                    break
        loader = getattr(self, "loader", None)
        if loader is not None and hasattr(loader, "dataset") and hasattr(loader.dataset, "close"):
            try:
                loader.dataset.close()
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class BackgroundPrefetcher:
    """
    Lightweight asynchronous prefetcher wrapper for PyTorch DataLoaders.
    Pre-buffers batches on a background thread so training loops never wait on
    remote Parquet row-group downloads or decompression boundaries.
    """

    def __init__(self, loader: Any, maxsize: int = 32):
        self.loader = loader
        self.maxsize = maxsize
        self._active_iterator: Optional[_BackgroundPrefetchIterator] = None

    def __iter__(self) -> _BackgroundPrefetchIterator:
        self.close()
        self._active_iterator = _BackgroundPrefetchIterator(self.loader, maxsize=self.maxsize)
        return self._active_iterator

    def __len__(self) -> int:
        return len(self.loader)

    @property
    def dataset(self) -> Any:
        return self.loader.dataset

    @property
    def batch_size(self) -> Optional[int]:
        return getattr(self.loader, "batch_size", None)

    def close(self) -> None:
        """Close active iterator and worker thread."""
        if getattr(self, "_active_iterator", None) is not None:
            try:
                self._active_iterator.close()
            except Exception:
                pass
            self._active_iterator = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.loader, name)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class _RawSamplePrefetchIterator:
    """
    Asynchronously pre-buffers raw samples from the streaming HuggingFace/PyArrow dataset.
    Overlaps PyArrow row group extraction and HTTP fetching with sample decoding and transformation,
    preventing 5s stalls at row group boundaries (e.g. every 23 iterations with batch size 8).
    """

    def __init__(self, stream: Iterator[Dict[str, Any]], maxsize: int = 64):
        self.stream = stream
        self.maxsize = maxsize
        self.queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._fetch_loop, daemon=True)
        self._worker.start()

    def _fetch_loop(self) -> None:
        try:
            for item in self.stream:
                while not self._stop_event.is_set():
                    try:
                        self.queue.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if self._stop_event.is_set():
                    break
        except Exception as e:
            if not self._stop_event.is_set():
                try:
                    self.queue.put(_ExceptionWrapper(e), timeout=0.2)
                except Exception:
                    pass
        finally:
            while not self._stop_event.is_set():
                try:
                    self.queue.put(_SENTINEL, timeout=0.1)
                    break
                except Exception:
                    continue

    def __iter__(self) -> _RawSamplePrefetchIterator:
        return self

    def __next__(self) -> Dict[str, Any]:
        while not self._stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.2)
            except queue.Empty:
                if not self._worker.is_alive() and (self.queue is None or self.queue.empty()):
                    self.close()
                    raise StopIteration
                continue

            if item is _SENTINEL:
                self.close()
                raise StopIteration
            if isinstance(item, _ExceptionWrapper):
                self.close()
                item.reraise()
            return item
        raise StopIteration

    def close(self) -> None:
        """Signal worker to stop, drain queue, and close underlying stream."""
        if getattr(self, "_stop_event", None) is not None:
            self._stop_event.set()
        q = getattr(self, "queue", None)
        if q is not None:
            while not q.empty():
                try:
                    q.get_nowait()
                except Exception:
                    break
        stream = getattr(self, "stream", None)
        if stream is not None and hasattr(stream, "close") and callable(getattr(stream, "close", None)):
            try:
                stream.close()
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def process_raw_sample(
    sample: Dict[str, Any],
    transform: Optional[JointCompose],
    target_image_size: Tuple[int, int] = (256, 256),
) -> Dict[str, Any]:
    """
    Process a single raw sample from datasets such as KhangTruong/IMD2020 or saberzl/SID_Set into model tensors.
    Handles 2-column format (image, mask) as well as labeled format (image, label, mask, img_id).
    Validates and checks for any image/mask data or dimension mismatches.
    """
    if not isinstance(sample, dict):
        raise TypeError(f"Expected sample to be a dict, got {type(sample)}")

    if "image" not in sample and "mask" not in sample:
        raise KeyError(f"Sample is missing required dataset columns ('image', 'mask'). Keys present: {list(sample.keys())}")

    raw_img = sample.get("image")
    if raw_img is None:
        raise ValueError(f"Sample '{sample.get('img_id', sample.get('id', 'unknown'))}' has null or missing image data.")

    raw_label = sample.get("label", None)
    raw_mask = sample.get("mask", None)
    img_id = str(sample.get("img_id", sample.get("id", "")))

    # Check for image and mask mismatches
    check_image_mask_mismatch(raw_img, raw_mask, raise_on_mismatch=False)

    # 1. Convert to RGB PIL Image
    pil_image = ensure_rgb_image(raw_img)

    # 2. Synthesize / extract mask based on label & provided mask
    mask_arr = process_sample_mask(
        mask_input=raw_mask,
        label=int(raw_label) if raw_label is not None else None,
        image_size=pil_image.size,  # (width, height)
    )

    # 3. Determine scalar class label (0: Real, 1: Fully Synthetic, 2: Partially Synthetic / Tampered)
    if raw_label is not None:
        label = int(raw_label)
    else:
        # Infer class label from mask for standard 2-column image/mask datasets like KhangTruong/IMD2020
        mask_sum = float(mask_arr.sum())
        total_pixels = float(mask_arr.size)
        if mask_sum == 0.0:
            label = 0  # Real / Untampered
        elif mask_sum >= total_pixels:
            label = 1  # Fully Synthetic
        else:
            label = 2  # Partially Synthetic / Tampered

    # 4. Apply joint transforms
    if transform is not None:
        img_tensor, mask_tensor = transform(pil_image, mask_arr)
    else:
        # Fallback default transform
        default_tf = get_transforms(image_size=target_image_size, is_train=False)
        img_tensor, mask_tensor = default_tf(pil_image, mask_arr)

    return {
        "image": img_tensor,           # (3, H, W)
        "mask": mask_tensor,           # (1, H, W)
        "label": torch.tensor(label, dtype=torch.long), # Scalar label (0, 1, 2)
        "img_id": img_id,
    }


def resolve_sample_limit(
    samples_val: Optional[Any],
    steps_val: Optional[Any],
    batch_size: int,
    default_samples: Optional[int] = None,
) -> Optional[int]:
    """
    Resolve sample count limit.
    If steps or samples <= 0 (e.g. -1), returns None (meaning run until dataset is depleted).
    """
    if steps_val is not None:
        steps_int = int(steps_val)
        if steps_int <= 0:
            return None
        return steps_int * batch_size

    if samples_val is not None:
        samples_int = int(samples_val)
        if samples_int <= 0:
            return None
        return samples_int

    return default_samples


def safe_dataloader_len(loader: Optional[DataLoader]) -> Optional[int]:
    """
    Safely return DataLoader length, or None if dataset has no length (e.g. IterableDataset).
    Avoids TypeError when DataLoader wraps an IterableDataset without __len__.
    """
    if loader is None:
        return None
    try:
        return len(loader)
    except (TypeError, NotImplementedError):
        return None


def get_split_candidates(requested_split: str) -> List[str]:
    """
    Return priority list of split name candidates for a requested split.
    Handles synonyms and sensible fallbacks (e.g. cross_test, test -> validation -> val -> train).
    """
    req_lower = requested_split.strip().lower()
    if req_lower in ("cross_test", "crosstest", "cross-test"):
        candidates = [requested_split, "cross_test", "cross-test", "test", "testing", "validation", "val", "eval", "train"]
    elif req_lower in ("test", "testing", "eval", "evaluation"):
        candidates = [requested_split, "cross_test", "test", "testing", "validation", "val", "eval", "evaluation", "train"]
    elif req_lower in ("val", "validation", "valid", "dev"):
        candidates = [requested_split, "validation", "val", "valid", "dev", "cross_test", "test", "testing", "eval", "train"]
    elif req_lower in ("train", "training"):
        candidates = [requested_split, "train", "training", "train_data", "train_set"]
    else:
        candidates = [requested_split, "cross_test", "test", "validation", "val", "train"]

    # De-duplicate while preserving order
    seen = set()
    return [c for c in candidates if not (c in seen or seen.add(c))]


def load_hf_dataset_robust(
    dataset_name: str,
    requested_split: str,
    streaming: bool = False,
) -> Tuple[Any, str]:
    """
    Load a HuggingFace dataset with robust split fallback.
    Tries the requested split first, then equivalent aliases, then fallback splits.
    Returns a tuple of (dataset, resolved_split_name).
    """
    candidates = get_split_candidates(requested_split)
    last_error: Optional[Exception] = None

    for cand in candidates:
        try:
            ds = hf_load_dataset(dataset_name, split=cand, streaming=streaming)
            return ds, cand
        except Exception as e:
            last_error = e
            continue

    # If all candidates failed with specific split, try loading raw dataset object
    try:
        raw = hf_load_dataset(dataset_name, streaming=streaming)
        if isinstance(raw, dict):
            for cand in candidates:
                if cand in raw:
                    return raw[cand], cand
            first_key = next(iter(raw.keys()))
            return raw[first_key], first_key
        return raw, requested_split
    except Exception as e:
        if last_error is not None:
            raise last_error
        raise e


def load_parquet_dataset(
    dataset_name: str,
    requested_split: str,
    streaming: bool = True,
    save_to_disk: Union[bool, str] = False,
    background_download: bool = False,
    columns: Optional[Sequence[str]] = None,
    token: Optional[Union[bool, str]] = None,
    cache_dir: Optional[str] = None,
    max_cached_row_groups: int = 2,
    **kwargs: Any,
) -> Tuple[Any, str]:
    """
    Load a Parquet dataset via parquet-dataset-loader for instant, non-blocking access.
    Tries requested split and fallback candidates.
    Returns (IndexedParquetDataset, resolved_split_name).
    """
    if not _HAS_PARQUET_LOADER or pdl is None:
        raise ImportError(
            "parquet-dataset-loader is not installed or failed to import. "
            "Install via pip install parquet-dataset-loader."
        )

    candidates = get_split_candidates(requested_split)
    last_error: Optional[Exception] = None

    for cand in candidates:
        try:
            ds = pdl.load_dataset(
                path=dataset_name,
                split=cand,
                streaming=streaming,
                save_to_disk=save_to_disk,
                background_download=background_download,
                columns=columns,
                token=token,
                cache_dir=cache_dir,
                max_cached_row_groups=max_cached_row_groups,
                **kwargs,
            )
            return ds, cand
        except Exception as e:
            last_error = e
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError(
        f"Could not load parquet dataset '{dataset_name}' with split '{requested_split}'"
    )


def load_parquet_or_hf_dataset(
    dataset_name: str,
    requested_split: str,
    streaming: bool = True,
    save_to_disk: Union[bool, str] = False,
    background_download: bool = False,
    columns: Optional[Sequence[str]] = None,
    token: Optional[Union[bool, str]] = None,
    use_parquet_loader: bool = True,
    cache_dir: Optional[str] = None,
    max_cached_row_groups: int = 2,
    **kwargs: Any,
) -> Tuple[Any, str]:
    """
    Universal dataset loader that prefers parquet-dataset-loader for non-blocking
    O(log M) index streaming, automatically falling back to HuggingFace datasets if
    the dataset is not a Parquet repository, if parquet loader is disabled, or on error.
    """
    if hf_load_dataset is not _orig_hf_load_dataset:
        # Caller or test framework monkeypatched hf_load_dataset; respect the override
        return load_hf_dataset_robust(
            dataset_name, requested_split=requested_split, streaming=streaming
        )

    if use_parquet_loader and _HAS_PARQUET_LOADER and pdl is not None:
        try:
            return load_parquet_dataset(
                dataset_name=dataset_name,
                requested_split=requested_split,
                streaming=streaming,
                save_to_disk=save_to_disk,
                background_download=background_download,
                columns=columns,
                token=token,
                cache_dir=cache_dir,
                max_cached_row_groups=max_cached_row_groups,
                **kwargs,
            )
        except Exception as e:
            logger.info(
                f"parquet-dataset-loader could not load '{dataset_name}' ({type(e).__name__}: {e}); "
                "falling back to Hugging Face dataset loader."
            )

    return load_hf_dataset_robust(
        dataset_name, requested_split=requested_split, streaming=streaming
    )


def is_mock_dataset(dataset_name: Optional[str]) -> bool:
    """Check if the provided dataset name signifies a synthetic/mock dataset override."""
    if not dataset_name or not isinstance(dataset_name, str):
        return False
    norm = dataset_name.strip().lower()
    return norm in ("mock", "dummy", "synthetic", "mock:synthetic", "mock/synthetic")


def generate_mock_raw_sample(
    idx: int,
    target_image_size: Tuple[int, int] = (256, 256),
    split: str = "train",
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate a realistic synthetic image, mask, and label sample deterministically without network calls."""
    w, h = target_image_size
    rng = np.random.RandomState((seed + idx * 37) % (2**31 - 1))

    # Cycle across all 3 classes:
    # 0: Authentic / Real (all-zero mask)
    # 1: Fully Synthetic / Deepfake (all-one mask)
    # 2: Partially Tampered / Inpainted (localized masked region)
    label = idx % 3

    # Generate synthetic RGB background
    base_color = rng.randint(40, 200, size=(3,), dtype=np.uint8)
    noise = rng.randint(-25, 25, size=(h, w, 3), dtype=np.int16)
    img_arr = np.clip(base_color[None, None, :] + noise, 0, 255).astype(np.uint8)

    mask_arr = np.zeros((h, w), dtype=np.uint8)

    if label == 1:
        mask_arr.fill(255)
    elif label == 2:
        # Create a localized box of tampering
        box_w = max(16, int(w * rng.uniform(0.25, 0.6)))
        box_h = max(16, int(h * rng.uniform(0.25, 0.6)))
        x0 = int(rng.randint(0, max(1, w - box_w)))
        y0 = int(rng.randint(0, max(1, h - box_h)))
        mask_arr[y0 : y0 + box_h, x0 : x0 + box_w] = 255
        # Alter the image content inside the tampered region
        tamper_color = rng.randint(0, 255, size=(3,), dtype=np.uint8)
        img_arr[y0 : y0 + box_h, x0 : x0 + box_w] = (
            img_arr[y0 : y0 + box_h, x0 : x0 + box_w] // 2 + tamper_color[None, None, :] // 2
        )

    pil_img = Image.fromarray(img_arr, mode="RGB")

    return {
        "image": pil_img,
        "mask": mask_arr,
        "label": label,
        "img_id": f"mock_{split}_{idx}",
    }


class SIDMockDataset(Dataset):
    """Indexable map-style mock dataset for offline testing and fast pipeline iteration."""

    def __init__(
        self,
        split: str = "train",
        transform: Optional[JointCompose] = None,
        max_samples: Optional[int] = 100,
        target_image_size: Tuple[int, int] = (256, 256),
        seed: int = 42,
    ):
        super().__init__()
        self.split = split
        self.transform = transform
        self.max_samples = max_samples if (max_samples is not None and max_samples > 0) else 100
        self.target_image_size = target_image_size
        self.seed = seed

    def __len__(self) -> int:
        return self.max_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self.max_samples:
            raise IndexError(f"Index {idx} out of range for SIDMockDataset of size {self.max_samples}")
        raw = generate_mock_raw_sample(
            idx=idx,
            target_image_size=self.target_image_size,
            split=self.split,
            seed=self.seed,
        )
        return process_raw_sample(raw, transform=self.transform, target_image_size=self.target_image_size)


class SIDStreamingMockDataset(IterableDataset):
    """Streaming iterable mock dataset for offline testing and fast pipeline iteration."""

    def __init__(
        self,
        split: str = "train",
        transform: Optional[JointCompose] = None,
        max_samples: Optional[int] = None,
        target_image_size: Tuple[int, int] = (256, 256),
        seed: int = 42,
    ):
        super().__init__()
        self.split = split
        self.transform = transform
        self.max_samples = max_samples if (max_samples is not None and max_samples > 0) else None
        self.target_image_size = target_image_size
        self.seed = seed

    def __len__(self) -> int:
        if self.max_samples is not None:
            return self.max_samples
        return 100

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        limit = self.max_samples if self.max_samples is not None else 100
        idx = worker_id
        while idx < limit:
            raw = generate_mock_raw_sample(
                idx=idx,
                target_image_size=self.target_image_size,
                split=self.split,
                seed=self.seed,
            )
            yield process_raw_sample(raw, transform=self.transform, target_image_size=self.target_image_size)
            idx += num_workers


class SIDStreamingDataset(IterableDataset):
    """
    Streaming PyTorch Dataset wrapping HuggingFace IterableDataset.
    Ideal for massive datasets without downloading everything to disk.
    When max_samples is None or <= 0, streams all samples until dataset depletion.
    """

    def __init__(
        self,
        dataset_name: str = "saberzl/SID_Set",
        split: str = "train",
        transform: Optional[JointCompose] = None,
        shuffle_buffer_size: int = 1000,
        max_samples: Optional[int] = None,
        seed: int = 42,
        target_image_size: Tuple[int, int] = (256, 256),
        use_parquet_loader: bool = True,
        save_to_disk: Union[bool, str] = False,
        background_download: bool = False,
        columns: Optional[Sequence[str]] = None,
        token: Optional[Union[bool, str]] = None,
        cache_dir: Optional[str] = None,
        max_cached_row_groups: int = 2,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.requested_split = split
        self.split = split
        self.resolved_split = split
        self.transform = transform
        self.shuffle_buffer_size = shuffle_buffer_size
        self.max_samples = None if (max_samples is not None and max_samples <= 0) else max_samples
        self.seed = seed
        self.target_image_size = target_image_size
        self.use_parquet_loader = use_parquet_loader
        self.save_to_disk = save_to_disk
        self.background_download = background_download
        self.columns = columns
        self.token = token
        self.cache_dir = cache_dir
        self.max_cached_row_groups = max_cached_row_groups
        self._current_dataset = None

    def __len__(self) -> int:
        if self.max_samples is not None and self.max_samples > 0:
            return self.max_samples
        if is_mock_dataset(self.dataset_name):
            return 100
        cur = getattr(self, "_current_dataset", None)
        if cur is not None and hasattr(cur, "__len__"):
            try:
                return len(cur)
            except (TypeError, NotImplementedError):
                pass
        raise TypeError(f"'{type(self).__name__}' object has no len() when max_samples is None")

    def _get_mock_stream(self) -> Iterator[Dict[str, Any]]:
        limit = self.max_samples if (self.max_samples is not None and self.max_samples > 0) else 100
        is_dist = is_dist_avail_and_initialized()
        rank = get_rank() if is_dist else 0
        world_size = get_world_size() if is_dist else 1

        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        global_worker_id = rank * num_workers + worker_id
        total_global_workers = world_size * num_workers

        idx = global_worker_id
        while idx < limit:
            yield generate_mock_raw_sample(
                idx=idx,
                target_image_size=self.target_image_size,
                split=self.split,
                seed=self.seed,
            )
            idx += total_global_workers

    def _get_stream(self) -> Iterator[Dict[str, Any]]:
        if is_mock_dataset(self.dataset_name):
            return self._get_mock_stream()

        # Load streamed dataset with robust fallback
        raw_ds, resolved = load_parquet_or_hf_dataset(
            self.dataset_name,
            requested_split=self.split,
            streaming=True,
            save_to_disk=self.save_to_disk,
            background_download=self.background_download,
            columns=self.columns,
            token=self.token,
            use_parquet_loader=self.use_parquet_loader,
            cache_dir=self.cache_dir,
            max_cached_row_groups=self.max_cached_row_groups,
        )
        self.resolved_split = resolved
        self._current_dataset = raw_ds

        # Note: Do NOT call hf_ds.shuffle() on remote multi-shard IterableDataset streams,
        # as Hugging Face opens simultaneous HTTP readers across all shards (e.g. 62 parquet files),
        # causing PyArrow and host memory to spike by >6 GB and trigger cgroup OOM kills.
        # Shuffling is handled safely below via the local reservoir buffer on processed samples.

        is_dist = is_dist_avail_and_initialized()
        rank = get_rank() if is_dist else 0
        world_size = get_world_size() if is_dist else 1

        # Shard across distributed ranks if world_size > 1
        if world_size > 1:
            if hasattr(raw_ds, "shard") and callable(raw_ds.shard):
                try:
                    raw_ds = raw_ds.shard(num_shards=world_size, index=rank)
                except Exception:
                    raw_ds = itertools.islice(raw_ds, rank, None, world_size)
            elif hasattr(raw_ds, "n_shards") and raw_ds.n_shards >= world_size:
                raw_ds = raw_ds.shard(num_shards=world_size, index=rank)
            else:
                raw_ds = itertools.islice(raw_ds, rank, None, world_size)

        worker_info = get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            if hasattr(raw_ds, "shard") and callable(raw_ds.shard):
                try:
                    stream_iter = iter(raw_ds.shard(num_shards=num_workers, index=worker_id))
                except Exception:
                    stream_iter = itertools.islice(raw_ds, worker_id, None, num_workers)
            elif hasattr(raw_ds, "n_shards") and raw_ds.n_shards >= num_workers:
                stream_iter = iter(raw_ds.shard(num_shards=num_workers, index=worker_id))
            else:
                stream_iter = itertools.islice(raw_ds, worker_id, None, num_workers)

            if self.max_samples is not None and self.max_samples > 0:
                worker_max = (self.max_samples - 1 - worker_id) // num_workers + 1 if self.max_samples > worker_id else 0
                stream_iter = itertools.islice(stream_iter, worker_max)
        else:
            if hasattr(raw_ds, "take") and callable(raw_ds.take) and self.max_samples is not None and self.max_samples > 0:
                stream_iter = iter(raw_ds.take(self.max_samples))
            else:
                stream_iter = iter(raw_ds)
                if self.max_samples is not None and self.max_samples > 0:
                    stream_iter = itertools.islice(stream_iter, self.max_samples)

        return stream_iter

    def close(self) -> None:
        """Explicitly close the current active stream, reader, and raw prefetch iterator if supported."""
        raw_iter = getattr(self, "_raw_prefetch_iter", None)
        if raw_iter is not None and hasattr(raw_iter, "close") and callable(raw_iter.close):
            try:
                raw_iter.close()
            except Exception:
                pass
        self._raw_prefetch_iter = None

        stream = getattr(self, "_current_stream", None)
        if stream is not None and hasattr(stream, "close") and callable(stream.close):
            try:
                stream.close()
            except Exception:
                pass
        self._current_stream = None

        ds = getattr(self, "_current_dataset", None)
        if ds is not None and hasattr(ds, "close") and callable(ds.close):
            try:
                ds.close()
            except Exception:
                pass
        self._current_dataset = None

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        import random
        self.close()
        stream = self._get_stream()
        self._current_stream = stream
        # Asynchronously prefetch raw samples from remote stream to overlap Parquet row group
        # extraction with image transformation and decode on the CPU
        prefetch_stream = _RawSamplePrefetchIterator(stream, maxsize=128)
        self._raw_prefetch_iter = prefetch_stream

        # Maintain a lightweight reservoir/shuffle buffer on processed (resized) samples
        # Capped to 32 samples to prevent memory ballooning while providing local randomness
        target_buf_size = min(max(0, self.shuffle_buffer_size), 32) if self.resolved_split.lower() in ("train", "training") else 0
        buf: List[Dict[str, Any]] = []
        rng = random.Random(self.seed)

        try:
            while True:
                try:
                    raw_sample = next(prefetch_stream)
                except StopIteration:
                    break

                try:
                    processed = process_raw_sample(
                        raw_sample,
                        transform=self.transform,
                        target_image_size=self.target_image_size,
                    )
                except Exception as proc_err:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Skipping corrupted sample during processing: {proc_err}"
                    )
                    continue
                finally:
                    del raw_sample

                if target_buf_size > 1:
                    buf.append(processed)
                    if len(buf) >= target_buf_size:
                        idx = rng.randint(0, len(buf) - 1)
                        yield buf.pop(idx)
                else:
                    yield processed

            if buf:
                rng.shuffle(buf)
                for item in buf:
                    yield item
        finally:
            buf.clear()
            if prefetch_stream is not None:
                prefetch_stream.close()
            self._raw_prefetch_iter = None
            if hasattr(stream, "close") and callable(getattr(stream, "close", None)):
                try:
                    stream.close()
                except Exception:
                    pass
            self._current_stream = None
            del stream


class SIDMapDataset(Dataset):
    """
    Standard Indexable PyTorch Dataset when streaming = False.
    When use_parquet_loader is True, leverages IndexedParquetDataset to provide
    instant non-blocking index streaming with O(log M) random access, avoiding multi-gigabyte
    download blocking at initialization.
    Supports background downloading and progressive disk persistence.
    When max_samples is None or <= 0, uses the full dataset until depletion.
    """

    def __init__(
        self,
        dataset_name: str = "saberzl/SID_Set",
        split: str = "train",
        transform: Optional[JointCompose] = None,
        max_samples: Optional[int] = None,
        target_image_size: Tuple[int, int] = (256, 256),
        use_parquet_loader: bool = True,
        save_to_disk: Union[bool, str] = False,
        background_download: bool = False,
        columns: Optional[Sequence[str]] = None,
        token: Optional[Union[bool, str]] = None,
        cache_dir: Optional[str] = None,
        max_cached_row_groups: int = 2,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.requested_split = split
        self.transform = transform
        self.target_image_size = target_image_size
        self.max_samples = None if (max_samples is not None and max_samples <= 0) else max_samples
        self.use_parquet_loader = use_parquet_loader
        self.save_to_disk = save_to_disk
        self.background_download = background_download
        self.columns = columns
        self.token = token
        self.cache_dir = cache_dir
        self.max_cached_row_groups = max_cached_row_groups

        if is_mock_dataset(dataset_name):
            self.resolved_split = split
            self.split = split
            self._is_mock = True
            self._mock_len = self.max_samples if (self.max_samples is not None and self.max_samples > 0) else 100
            self.data = None
            return

        self._is_mock = False
        ds, resolved = load_parquet_or_hf_dataset(
            dataset_name=dataset_name,
            requested_split=split,
            streaming=True,  # For parquet-dataset-loader, streaming=True returns IndexedParquetDataset with instant O(log M) random access!
            save_to_disk=save_to_disk,
            background_download=background_download,
            columns=columns,
            token=token,
            use_parquet_loader=use_parquet_loader,
            cache_dir=cache_dir,
            max_cached_row_groups=max_cached_row_groups,
        )
        self.resolved_split = resolved
        self.split = resolved

        # If fallback loaded an IterableDataset from HuggingFace (not indexable),
        # reload using HF non-streaming so it becomes an indexable Dataset:
        if not hasattr(ds, "__getitem__"):
            ds, resolved = load_hf_dataset_robust(dataset_name, requested_split=split, streaming=False)
            self.resolved_split = resolved
            self.split = resolved

        if self.max_samples is not None and 0 < self.max_samples < len(ds):
            if hasattr(ds, "take") and callable(ds.take):
                ds = ds.take(self.max_samples)
            elif hasattr(ds, "select") and callable(ds.select):
                ds = ds.select(range(self.max_samples))
            else:
                ds = ds[: self.max_samples]

        self.data = ds

    def __len__(self) -> int:
        if getattr(self, "_is_mock", False):
            return self._mock_len
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if getattr(self, "_is_mock", False):
            if idx < 0 or idx >= self._mock_len:
                raise IndexError(f"Index {idx} out of range for mock dataset of size {self._mock_len}")
            raw_sample = generate_mock_raw_sample(
                idx=idx,
                target_image_size=self.target_image_size,
                split=self.split,
            )
            sample_dict = process_raw_sample(
                raw_sample,
                transform=self.transform,
                target_image_size=self.target_image_size,
            )
            sample_dict["sample_idx"] = idx
            return sample_dict
        raw_sample = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                raw_sample = self.data[idx]
                break
            except Exception as exc:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.5 * (2 ** attempt))
                else:
                    raise
        sample_dict = process_raw_sample(
            raw_sample,
            transform=self.transform,
            target_image_size=self.target_image_size,
        )
        sample_dict["sample_idx"] = idx
        return sample_dict

    def close(self) -> None:
        """Close underlying dataset reader and stop any background download thread."""
        data = getattr(self, "data", None)
        if data is not None and hasattr(data, "close") and callable(data.close):
            try:
                data.close()
            except Exception:
                pass


def resolve_num_workers(
    num_workers_cfg: Optional[Union[int, str]] = None,
    max_workers: Optional[int] = None,
) -> int:
    """Resolve number of dataloader workers. -1 or negative defaults to number of CPU cores."""
    if num_workers_cfg is None:
        val = -1
    else:
        try:
            val = int(num_workers_cfg)
        except (ValueError, TypeError):
            val = -1

    if val < 0:
        try:
            cores = max(1, len(os.sched_getaffinity(0)))
        except (AttributeError, NotImplementedError, OSError):
            cores = max(1, os.cpu_count() or 1)
        if max_workers is not None and max_workers > 0:
            return min(cores, max_workers)
        return cores
    return val


def resolve_batch_size(config: Any) -> int:
    """
    Resolve and scale batch size for multi-GPU data parallelism.
    In DistributedDataParallel (DDP) multi-process mode, each process trains on base_batch_size,
    achieving an effective global batch size of base_batch_size * world_size across all ranks.
    In single-process DataParallel mode, automatically multiplies data.batch_size with actual GPU numbers
    so each device trains with at least batch size 1.
    Preserves data.base_batch_size and sets data._batch_size_scaled to prevent double scaling.
    """
    if not hasattr(config, "data"):
        return 16

    base_bs = int(config.data.get("base_batch_size", config.data.get("batch_size", 16)))
    if base_bs < 1:
        base_bs = 1
    config.data.base_batch_size = base_bs

    if getattr(config.data, "_batch_size_scaled", False):
        return int(config.data.batch_size)

    dev_cfg = str(config.project.get("device", "auto")).lower() if hasattr(config, "project") else "auto"
    data_parallel = False
    if hasattr(config, "training"):
        data_parallel = bool(config.training.get("data_parallel", False))

    # DistributedDataParallel (DDP) multi-process mode
    if is_dist_avail_and_initialized():
        world_size = get_world_size()
        config.data.batch_size = base_bs
        config.data.num_gpus = world_size
        if hasattr(config, "training"):
            config.training.num_gpus = world_size
            config.training.data_parallel = (world_size > 1)
        config.data._batch_size_scaled = True
        return base_bs

    if torch.cuda.is_available() and dev_cfg in ("auto", "cuda") and data_parallel:
        num_gpus = max(1, torch.cuda.device_count())
    else:
        num_gpus = 1

    actual_bs = base_bs * num_gpus
    config.data.batch_size = actual_bs
    config.data.num_gpus = num_gpus
    if hasattr(config, "training"):
        config.training.num_gpus = num_gpus
        config.training.data_parallel = (num_gpus > 1)
    config.data._batch_size_scaled = True
    return actual_bs


def create_eval_dataloader(
    config: Any,
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
    samples_override: Optional[int] = None,
) -> DataLoader:
    """
    Create a DataLoader specifically for evaluation or testing on a given split.
    Automatically resolves test vs validation split with robust fallback.

    Args:
        config: ConfigDict or configuration object.
        split: Split name (e.g. 'test', 'validation', 'val').
               If None, defaults to config.data.get('eval_split', config.data.get('test_split', 'test')).
        max_samples: Optional sample count limit (overrides config).
        samples_override: Alias for max_samples.
    """
    if max_samples is None and samples_override is not None:
        max_samples = samples_override

    dataset_name = config.data.get("dataset_name", "KhangTruong/IMD2020")
    if is_mock_dataset(dataset_name) or bool(config.data.get("mock", False)):
        dataset_name = "mock"
    streaming = bool(config.data.get("streaming", False))
    batch_size = int(config.data.get("batch_size", 16))
    max_eval_workers = int(config.data.get("max_eval_workers", config.data.get("max_workers", 4)))
    num_workers = resolve_num_workers(config.data.get("num_workers", -1), max_workers=max_eval_workers)
    pin_memory = bool(config.data.get("pin_memory", True)) and torch.cuda.is_available()
    image_size = tuple(config.data.get("image_size", [256, 256]))
    seed = int(config.project.get("seed", 42))

    eval_split = split or config.data.get("eval_split", config.data.get("test_split", "test"))

    if max_samples is not None:
        eval_max_samples = None if max_samples <= 0 else int(max_samples)
    else:
        if "test" in str(eval_split).lower():
            samples_cfg = config.data.get("test_samples", -1)
        else:
            samples_cfg = config.data.get("val_samples_per_epoch", config.data.get("val_samples", -1))
        steps_cfg = config.data.get("eval_steps", None)
        eval_max_samples = resolve_sample_limit(
            samples_val=samples_cfg,
            steps_val=steps_cfg,
            batch_size=batch_size,
            default_samples=None,
        )

    transform = get_transforms(image_size=image_size, is_train=False)
    mp_context = torch.multiprocessing.get_context("spawn") if (num_workers > 0 and os.name != "nt") else None

    use_parquet_loader = bool(config.data.get("use_parquet_loader", True))
    save_to_disk = config.data.get("save_to_disk", False)
    background_download = bool(config.data.get("background_download", False))
    columns = config.data.get("columns", None)
    token = config.data.get("token", None)
    pdl_cache_dir = config.data.get("cache_dir", config.data.get("pdl_cache_dir", None))
    max_cached_row_groups = int(config.data.get("max_cached_row_groups", 2))

    if streaming:
        eval_dataset = SIDStreamingDataset(
            dataset_name=dataset_name,
            split=eval_split,
            transform=transform,
            shuffle_buffer_size=0,
            max_samples=eval_max_samples,
            seed=seed,
            target_image_size=image_size,
            use_parquet_loader=use_parquet_loader,
            save_to_disk=save_to_disk,
            background_download=background_download,
            columns=columns,
            token=token,
            cache_dir=pdl_cache_dir,
            max_cached_row_groups=max_cached_row_groups,
        )
        prefetch_batches = int(config.data.get("prefetch_batches", 24))
        raw_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
        )
        return BackgroundPrefetcher(raw_loader, maxsize=prefetch_batches)
    else:
        eval_dataset = SIDMapDataset(
            dataset_name=dataset_name,
            split=eval_split,
            transform=transform,
            max_samples=eval_max_samples,
            target_image_size=image_size,
            use_parquet_loader=use_parquet_loader,
            save_to_disk=save_to_disk,
            background_download=background_download,
            columns=columns,
            token=token,
            cache_dir=pdl_cache_dir,
            max_cached_row_groups=max_cached_row_groups,
        )
        eval_sampler = None
        if is_dist_avail_and_initialized():
            eval_sampler = torch.utils.data.distributed.DistributedSampler(
                eval_dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=False,
                seed=seed,
            )
        return DataLoader(
            eval_dataset,
            batch_size=batch_size,
            sampler=eval_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            multiprocessing_context=mp_context,
            worker_init_fn=worker_init_fn,
        )


def resolve_cached_parquet_files(
    repo_or_dir: str,
    split: str = "train",
    token: Optional[str] = None,
) -> List[str]:
    """
    Resolve local or Hugging Face Hub Parquet file paths for a cached tensor dataset split.
    """
    import glob
    if not repo_or_dir:
        return []

    # 1. Local path
    if os.path.exists(repo_or_dir):
        if os.path.isfile(repo_or_dir) and repo_or_dir.endswith(".parquet"):
            return [repo_or_dir]
        patterns = [
            os.path.join(repo_or_dir, split, "*.parquet"),
            os.path.join(repo_or_dir, f"{split}-*.parquet"),
            os.path.join(repo_or_dir, f"*{split}*.parquet"),
            os.path.join(repo_or_dir, "*.parquet"),
        ]
        for p in patterns:
            matched = sorted(glob.glob(p))
            if matched:
                return matched
        return []

    # 2. Remote Hugging Face Hub repo
    try:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(
            repo_id=repo_or_dir,
            repo_type="dataset",
            allow_patterns="*.parquet",
            token=token,
        )
        patterns = [
            os.path.join(local_dir, split, "*.parquet"),
            os.path.join(local_dir, f"{split}-*.parquet"),
            os.path.join(local_dir, f"*{split}*.parquet"),
            os.path.join(local_dir, "*.parquet"),
        ]
        for p in patterns:
            matched = sorted(glob.glob(p))
            if matched:
                return matched
        return []
    except Exception as e:
        logger.warning(f"Could not download cached parquet files from HF repo '{repo_or_dir}': {e}")
        return []


def _generate_mock_cached_parquet_shard(
    split: str = "train",
    image_size: Tuple[int, int] = (256, 256),
    channels: int = 84,
    num_samples: int = 16,
) -> List[str]:
    """Generate a temporary synthetic cached Parquet shard file for testing/fallback."""
    import tempfile
    import pyarrow as pa
    import pyarrow.parquet as pq

    temp_dir = os.path.join(tempfile.gettempdir(), "sid_mock_cache", split)
    os.makedirs(temp_dir, exist_ok=True)
    shard_path = os.path.join(temp_dir, f"mock_{split}_00000.parquet")

    latent_h, latent_w = image_size[0] // 8, image_size[1] // 8
    img_ids = [f"mock_{split}_{i}" for i in range(num_samples)]
    labels = [i % 3 for i in range(num_samples)]

    z_bytes_list = []
    mask_bytes_list = []
    img_bytes_list = []
    for i in range(num_samples):
        if channels == 256:
            z_t = np.random.randn(channels).astype(np.float16)
        else:
            z_t = np.random.randn(channels, latent_h, latent_w).astype(np.float16)
        z_bytes_list.append(z_t.tobytes())

        mask_img = Image.new("L", (image_size[1], image_size[0]), color=(255 if (i % 2 == 1) else 0))
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        mask_bytes_list.append(buf.getvalue())

        rgb_img = Image.new("RGB", (image_size[1], image_size[0]), color=((i * 20) % 255, (i * 40) % 255, (i * 60) % 255))
        ibuf = io.BytesIO()
        rgb_img.save(ibuf, format="JPEG")
        img_bytes_list.append(ibuf.getvalue())

    pydict = {
        "img_id": img_ids,
        "label": labels,
        "z_high_dim": z_bytes_list,
        "mask": mask_bytes_list,
        "channels": [channels] * num_samples,
        "latent_h": [(1 if channels == 256 else latent_h)] * num_samples,
        "latent_w": [(1 if channels == 256 else latent_w)] * num_samples,
    }
    if channels == 256:
        pydict["image"] = img_bytes_list

    table = pa.Table.from_pydict(pydict)
    pq.write_table(table, shard_path, compression="zstd")
    return [shard_path]


def create_cached_dataloaders(
    config: Any,
    cached_repo: str,
    include_test: bool = False,
) -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
    """
    Build DataLoaders from cached high-dimensional forensics representations.
    Prefers parquet-dataset-loader for instant, non-blocking HTTP streaming with
    DDP sharding and fork-safe multi-worker DataLoaders.
    Falls back to local shard resolution (CachedStreamingDataset / CachedTensorDataset).
    """
    from sid_unet.cache.dataset import (
        CachedTensorDataset,
        CachedStreamingDataset,
        CachedIndexedDataset,
        SpatialJointTransform,
    )

    batch_size = resolve_batch_size(config)
    streaming = bool(config.data.get("streaming", False))
    pin_memory = bool(config.data.get("pin_memory", True)) and torch.cuda.is_available()
    image_size = tuple(config.data.get("image_size", [256, 256]))
    token = config.data.get("token", None)
    seed = int(config.project.get("seed", 42))

    train_split = config.data.get("train_split", "train")
    val_split = config.data.get("val_split", "validation")
    test_split = config.data.get("test_split", val_split)

    train_samples_cfg = config.data.get("train_samples_per_epoch", -1)
    train_max_samples = resolve_sample_limit(
        samples_val=train_samples_cfg, steps_val=None, batch_size=batch_size, default_samples=None
    )

    val_samples_cfg = config.data.get("val_samples_per_epoch", config.data.get("val_samples", -1))
    val_max_samples = resolve_sample_limit(
        samples_val=val_samples_cfg, steps_val=None, batch_size=batch_size, default_samples=None
    )

    aug_cfg = config.data.get("augmentations", {})
    train_transform = SpatialJointTransform(
        horizontal_flip=aug_cfg.get("horizontal_flip", 0.5),
        vertical_flip=aug_cfg.get("vertical_flip", 0.0),
        random_rotate90=aug_cfg.get("random_rotate90", 0.25),
    )

    model_name = str(config.model.get("name", "")).lower() if hasattr(config, "model") and hasattr(config.model, "get") else ""
    default_channels = 256 if model_name in ("gap_sam", "gap-sam", "gapsam") else 84
    expected_channels = int(config.model.get("total_z_channels", default_channels))
    use_parquet_loader = bool(config.data.get("use_parquet_loader", True))
    save_to_disk = config.data.get("save_to_disk", False)
    background_download = bool(config.data.get("background_download", bool(save_to_disk)))
    pdl_cache_dir = config.data.get("cache_dir", config.data.get("pdl_cache_dir", None))
    max_cached_row_groups = int(config.data.get("max_cached_row_groups", 4))

    # 1. High-performance non-blocking path via parquet-dataset-loader
    if use_parquet_loader and _HAS_PARQUET_LOADER and pdl is not None:
        try:
            logger.info(
                f"⚡ Loading cached dataset '{cached_repo}' via parquet-dataset-loader (streaming={streaming}, save_to_disk={save_to_disk})..."
            )
            train_pdl, resolved_train_split = load_parquet_dataset(
                dataset_name=cached_repo,
                requested_split=train_split,
                streaming=streaming,
                save_to_disk=save_to_disk,
                background_download=background_download,
                token=token,
                cache_dir=pdl_cache_dir,
                max_cached_row_groups=max_cached_row_groups,
            )
            val_pdl, resolved_val_split = load_parquet_dataset(
                dataset_name=cached_repo,
                requested_split=val_split,
                streaming=streaming,
                save_to_disk=save_to_disk,
                background_download=background_download,
                token=token,
                cache_dir=pdl_cache_dir,
                max_cached_row_groups=max_cached_row_groups,
            )

            train_dataset = CachedIndexedDataset(
                pdl_dataset=train_pdl,
                transform=train_transform,
                target_image_size=image_size,
                expected_channels=expected_channels,
                max_samples=train_max_samples,
            )
            val_dataset = CachedIndexedDataset(
                pdl_dataset=val_pdl,
                transform=None,
                target_image_size=image_size,
                expected_channels=expected_channels,
                max_samples=val_max_samples,
            )

            if is_dist_avail_and_initialized():
                world_size = get_world_size()
                rank = get_rank()
                train_dataset = train_dataset.shard(num_shards=world_size, index=rank, contiguous=True)
                val_dataset = val_dataset.shard(num_shards=world_size, index=rank, contiguous=True)
                logger.info(
                    f"⚡ Sharded cached dataset for DDP rank {rank}/{world_size}: "
                    f"Train samples = {len(train_dataset)}, Val samples = {len(val_dataset)}"
                )

            raw_workers = resolve_num_workers(config.data.get("num_workers", -1), max_workers=4)
            max_cached_workers = int(config.data.get("max_cached_workers", 4))
            num_workers = min(max(0, raw_workers), max_cached_workers)
            mp_context = torch.multiprocessing.get_context("spawn") if (num_workers > 0 and os.name != "nt") else None

            # Zero-thrashing sampler preserving row-group cache locality
            try:
                from parquet_dataset_loader import BlockShuffledSampler
            except ImportError:
                BlockShuffledSampler = None

            extra_loader_kwargs: Dict[str, Any] = {}
            if num_workers > 0:
                extra_loader_kwargs["persistent_workers"] = bool(config.data.get("persistent_workers", True))
                extra_loader_kwargs["prefetch_factor"] = max(2, int(config.data.get("prefetch_factor", 2)))

            if BlockShuffledSampler is not None:
                window_blocks = int(config.data.get("window_blocks", 1))
                rank_seed = seed + (get_rank() if is_dist_avail_and_initialized() else 0)
                train_sampler = BlockShuffledSampler(
                    train_dataset,
                    window_blocks=window_blocks,
                    seed=rank_seed,
                    shuffle=True,
                )
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    sampler=train_sampler,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=False,
                    multiprocessing_context=mp_context,
                    worker_init_fn=worker_init_fn,
                    **extra_loader_kwargs,
                )
            else:
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=False,
                    multiprocessing_context=mp_context,
                    worker_init_fn=worker_init_fn,
                    **extra_loader_kwargs,
                )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
                multiprocessing_context=mp_context,
                worker_init_fn=worker_init_fn,
                **extra_loader_kwargs,
            )

            if include_test:
                test_pdl, _ = load_parquet_dataset(
                    dataset_name=cached_repo,
                    requested_split=test_split,
                    streaming=streaming,
                    save_to_disk=save_to_disk,
                    background_download=background_download,
                    token=token,
                    cache_dir=pdl_cache_dir,
                    max_cached_row_groups=max_cached_row_groups,
                )
                test_dataset = CachedIndexedDataset(
                    pdl_dataset=test_pdl,
                    transform=None,
                    target_image_size=image_size,
                    expected_channels=expected_channels,
                )
                if is_dist_avail_and_initialized():
                    test_dataset = test_dataset.shard(num_shards=get_world_size(), index=get_rank(), contiguous=True)
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=False,
                    multiprocessing_context=mp_context,
                    worker_init_fn=worker_init_fn,
                )
                return train_loader, val_loader, test_loader

            return train_loader, val_loader

        except Exception as pdl_err:
            logger.warning(
                f"⚠️ parquet-dataset-loader direct cached access encountered warning ({type(pdl_err).__name__}: {pdl_err}). "
                "Falling back to resolve_cached_parquet_files..."
            )

    # 2. Fallback path via resolve_cached_parquet_files
    train_files = resolve_cached_parquet_files(cached_repo, split=train_split, token=token)
    val_files = resolve_cached_parquet_files(cached_repo, split=val_split, token=token)

    # Fallback to mock shard if repo has no parquet files
    if not train_files:
        logger.warning(
            f"No cached parquet files found in '{cached_repo}' for split '{train_split}'. "
            "Generating temporary mock cached shard for training execution..."
        )
        train_files = _generate_mock_cached_parquet_shard(
            split=train_split, image_size=image_size, channels=expected_channels, num_samples=16
        )
    if not val_files:
        val_files = _generate_mock_cached_parquet_shard(
            split=val_split, image_size=image_size, channels=expected_channels, num_samples=8
        )

    if streaming:
        shuffle_buffer = int(config.data.get("shuffle_buffer_size", 32))
        train_dataset = CachedStreamingDataset(
            parquet_files=train_files,
            transform=train_transform,
            target_image_size=image_size,
            expected_channels=expected_channels,
            max_samples=train_max_samples,
            shuffle=True,
            shuffle_buffer_size=shuffle_buffer,
            seed=seed,
        )
        val_dataset = CachedStreamingDataset(
            parquet_files=val_files,
            transform=None,
            target_image_size=image_size,
            expected_channels=expected_channels,
            max_samples=val_max_samples,
            shuffle=False,
            seed=seed,
        )

        train_prefetch = max(48, int(config.data.get("prefetch_batches", 48)))
        val_prefetch = max(16, train_prefetch // 2)

        raw_train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
        )
        raw_val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
        )
        train_loader = BackgroundPrefetcher(raw_train_loader, maxsize=train_prefetch)
        val_loader = BackgroundPrefetcher(raw_val_loader, maxsize=val_prefetch)

        if include_test:
            test_files = resolve_cached_parquet_files(cached_repo, split=test_split, token=token) or val_files
            test_dataset = CachedStreamingDataset(
                parquet_files=test_files,
                transform=None,
                target_image_size=image_size,
                expected_channels=expected_channels,
                shuffle=False,
                seed=seed,
            )
            raw_test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                num_workers=0,
                pin_memory=pin_memory,
                worker_init_fn=worker_init_fn,
            )
            test_loader = BackgroundPrefetcher(raw_test_loader, maxsize=val_prefetch)
            return train_loader, val_loader, test_loader
        return train_loader, val_loader

    # Map-style (non-streaming) loader with memory-bounded LRU cache and safe worker count
    raw_workers = resolve_num_workers(config.data.get("num_workers", -1), max_workers=4)
    max_cached_workers = int(config.data.get("max_cached_workers", 4))
    num_workers = min(max(0, raw_workers), max_cached_workers)
    mp_context = torch.multiprocessing.get_context("spawn") if (num_workers > 0 and os.name != "nt") else None

    train_dataset = CachedTensorDataset(
        parquet_files=train_files,
        transform=train_transform,
        target_image_size=image_size,
        expected_channels=expected_channels,
        max_samples=train_max_samples,
        max_cached_tables=int(config.data.get("max_cached_tables", 2)),
    )
    val_dataset = CachedTensorDataset(
        parquet_files=val_files,
        transform=None,
        target_image_size=image_size,
        expected_channels=expected_channels,
        max_samples=val_max_samples,
        max_cached_tables=int(config.data.get("max_cached_tables", 2)),
    )

    train_sampler = None
    val_sampler = None
    if is_dist_avail_and_initialized():
        world_size = get_world_size()
        rank = get_rank()
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=seed,
        )

    extra_cached_kwargs: Dict[str, Any] = {}
    if num_workers > 0:
        extra_cached_kwargs["persistent_workers"] = bool(config.data.get("persistent_workers", True))
        extra_cached_kwargs["prefetch_factor"] = max(2, int(config.data.get("prefetch_factor", 2)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        multiprocessing_context=mp_context,
        worker_init_fn=worker_init_fn,
        **extra_cached_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        multiprocessing_context=mp_context,
        worker_init_fn=worker_init_fn,
        **extra_cached_kwargs,
    )

    if include_test:
        test_files = resolve_cached_parquet_files(cached_repo, split=test_split, token=token) or val_files
        test_dataset = CachedTensorDataset(
            parquet_files=test_files,
            transform=None,
            target_image_size=image_size,
            expected_channels=expected_channels,
            max_cached_tables=int(config.data.get("max_cached_tables", 2)),
        )
        test_sampler = None
        if is_dist_avail_and_initialized():
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=False,
                seed=seed,
            )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            multiprocessing_context=mp_context,
            worker_init_fn=worker_init_fn,
        )
        return train_loader, val_loader, test_loader

    return train_loader, val_loader


def create_test_dataloader(config: Any, max_samples: Optional[int] = None) -> DataLoader:
    """Create test DataLoader using config.data.test_split (default: 'test')."""
    test_split = config.data.get("test_split", "test")
    return create_eval_dataloader(config, split=test_split, max_samples=max_samples)


def create_dataloaders(
    config: Any,
    include_test: bool = False,
) -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
    """
    Create training and validation (and optionally test) DataLoaders based on configuration.
    Supports both streaming = True and streaming = False.
    Handles -1 or negative train_samples_per_epoch / steps_per_epoch / val_samples to run
    until dataset depletion.
    """
    cached_hf_repo = config.data.get("cached_hf_repo", None)
    if cached_hf_repo:
        logger.info(f"⚡ Creating DataLoaders from cached high-dimensional dataset: '{cached_hf_repo}'")
        return create_cached_dataloaders(config, cached_repo=str(cached_hf_repo), include_test=include_test)

    dataset_name = config.data.get("dataset_name", "saberzl/SID_Set")
    if is_mock_dataset(dataset_name) or bool(config.data.get("mock", False)):
        dataset_name = "mock"
    streaming = bool(config.data.get("streaming", True))
    batch_size = resolve_batch_size(config)
    max_data_workers = int(config.data.get("max_workers", 4))
    num_workers = resolve_num_workers(config.data.get("num_workers", -1), max_workers=max_data_workers)
    pin_memory = bool(config.data.get("pin_memory", True)) and torch.cuda.is_available()
    image_size = tuple(config.data.get("image_size", [256, 256]))
    shuffle_buffer = int(config.data.get("shuffle_buffer_size", 1000))
    seed = int(config.project.get("seed", 42))

    train_split = config.data.get("train_split", "train")
    val_split = config.data.get("val_split", "validation")

    train_samples_cfg = config.data.get("train_samples_per_epoch", 2000)
    steps_per_epoch_cfg = config.data.get(
        "steps_per_epoch",
        config.training.get("steps_per_epoch", None) if hasattr(config, "training") else None,
    )
    train_max_samples = resolve_sample_limit(
        samples_val=train_samples_cfg,
        steps_val=steps_per_epoch_cfg,
        batch_size=batch_size,
        default_samples=2000,
    )

    val_samples_cfg = config.data.get("val_samples_per_epoch", config.data.get("val_samples", 400))
    val_steps_cfg = config.data.get(
        "val_steps",
        config.training.get("val_steps", None) if hasattr(config, "training") else None,
    )
    val_max_samples = resolve_sample_limit(
        samples_val=val_samples_cfg,
        steps_val=val_steps_cfg,
        batch_size=batch_size,
        default_samples=400,
    )

    augment_config = config.data.get("augmentations", {})

    train_transform = get_transforms(image_size=image_size, is_train=True, augment_config=augment_config)
    val_transform = get_transforms(image_size=image_size, is_train=False)

    mp_context = torch.multiprocessing.get_context("spawn") if (num_workers > 0 and os.name != "nt") else None

    use_parquet_loader = bool(config.data.get("use_parquet_loader", True))
    save_to_disk = config.data.get("save_to_disk", False)
    background_download = bool(config.data.get("background_download", False))
    columns = config.data.get("columns", None)
    token = config.data.get("token", None)
    pdl_cache_dir = config.data.get("cache_dir", config.data.get("pdl_cache_dir", None))
    max_cached_row_groups = int(config.data.get("max_cached_row_groups", 2))

    if streaming:
        train_dataset = SIDStreamingDataset(
            dataset_name=dataset_name,
            split=train_split,
            transform=train_transform,
            shuffle_buffer_size=shuffle_buffer,
            max_samples=train_max_samples,
            seed=seed,
            target_image_size=image_size,
            use_parquet_loader=use_parquet_loader,
            save_to_disk=save_to_disk,
            background_download=background_download,
            columns=columns,
            token=token,
            cache_dir=pdl_cache_dir,
            max_cached_row_groups=max_cached_row_groups,
        )
        val_dataset = SIDStreamingDataset(
            dataset_name=dataset_name,
            split=val_split,
            transform=val_transform,
            shuffle_buffer_size=0,
            max_samples=val_max_samples,
            seed=seed,
            target_image_size=image_size,
            use_parquet_loader=use_parquet_loader,
            save_to_disk=save_to_disk,
            background_download=background_download,
            columns=columns,
            token=token,
            cache_dir=pdl_cache_dir,
            max_cached_row_groups=max_cached_row_groups,
        )

        train_prefetch = max(48, int(config.data.get("prefetch_batches", 48)))
        val_prefetch = max(16, train_prefetch // 2)

        raw_train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
        )
        raw_val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
        )
        train_loader = BackgroundPrefetcher(raw_train_loader, maxsize=train_prefetch)
        val_loader = BackgroundPrefetcher(raw_val_loader, maxsize=val_prefetch)
    else:
        train_dataset = SIDMapDataset(
            dataset_name=dataset_name,
            split=train_split,
            transform=train_transform,
            max_samples=train_max_samples,
            target_image_size=image_size,
            use_parquet_loader=use_parquet_loader,
            save_to_disk=save_to_disk,
            background_download=background_download,
            columns=columns,
            token=token,
            cache_dir=pdl_cache_dir,
            max_cached_row_groups=max_cached_row_groups,
        )
        val_dataset = SIDMapDataset(
            dataset_name=dataset_name,
            split=val_split,
            transform=val_transform,
            max_samples=val_max_samples,
            target_image_size=image_size,
            use_parquet_loader=use_parquet_loader,
            save_to_disk=save_to_disk,
            background_download=background_download,
            columns=columns,
            token=token,
            cache_dir=pdl_cache_dir,
            max_cached_row_groups=max_cached_row_groups,
        )

        train_sampler = None
        val_sampler = None
        if is_dist_avail_and_initialized():
            world_size = get_world_size()
            rank = get_rank()
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed,
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=seed,
            )

        extra_map_kwargs: Dict[str, Any] = {}
        if num_workers > 0:
            extra_map_kwargs["persistent_workers"] = bool(config.data.get("persistent_workers", True))
            extra_map_kwargs["prefetch_factor"] = max(2, int(config.data.get("prefetch_factor", 2)))

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=pin_memory,
            multiprocessing_context=mp_context,
            worker_init_fn=worker_init_fn,
            **extra_map_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            multiprocessing_context=mp_context,
            worker_init_fn=worker_init_fn,
            **extra_map_kwargs,
        )

    if include_test:
        test_loader = create_test_dataloader(config)
        return train_loader, val_loader, test_loader

    return train_loader, val_loader
