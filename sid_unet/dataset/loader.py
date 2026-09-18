"""
Dataset loading pipeline for SID_Set and common image manipulation datasets (e.g. KhangTruong/IMD2020).
Supports both streaming (IterableDataset) and non-streaming (MapDataset) modes with
robust handling of label 0 (black mask), label 1 (white mask), label 2 (provided mask),
and 2-column image/mask datasets.
"""

from __future__ import annotations

import itertools
import os
import queue
import sys
import threading
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from datasets import load_dataset as hf_load_dataset

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

    def __len__(self) -> int:
        if self.max_samples is not None and self.max_samples > 0:
            return self.max_samples
        if is_mock_dataset(self.dataset_name):
            return 100
        raise TypeError(f"'{type(self).__name__}' object has no len() when max_samples is None")

    def _get_mock_stream(self) -> Iterator[Dict[str, Any]]:
        limit = self.max_samples if (self.max_samples is not None and self.max_samples > 0) else 100
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        idx = worker_id
        while idx < limit:
            yield generate_mock_raw_sample(
                idx=idx,
                target_image_size=self.target_image_size,
                split=self.split,
                seed=self.seed,
            )
            idx += num_workers

    def _get_stream(self) -> Iterator[Dict[str, Any]]:
        if is_mock_dataset(self.dataset_name):
            return self._get_mock_stream()

        # Load streamed dataset with robust fallback
        hf_ds, resolved = load_hf_dataset_robust(self.dataset_name, requested_split=self.split, streaming=True)
        self.resolved_split = resolved

        # Note: Do NOT call hf_ds.shuffle() on remote multi-shard IterableDataset streams,
        # as Hugging Face opens simultaneous HTTP readers across all shards (e.g. 62 parquet files),
        # causing PyArrow and host memory to spike by >6 GB and trigger cgroup OOM kills.
        # Shuffling is handled safely below via the local reservoir buffer on processed samples.

        worker_info = get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            if hasattr(hf_ds, "n_shards") and hf_ds.n_shards >= num_workers:
                stream_iter = iter(hf_ds.shard(num_shards=num_workers, index=worker_id))
            else:
                stream_iter = itertools.islice(hf_ds, worker_id, None, num_workers)

            if self.max_samples is not None and self.max_samples > 0:
                worker_max = (self.max_samples - 1 - worker_id) // num_workers + 1 if self.max_samples > worker_id else 0
                stream_iter = itertools.islice(stream_iter, worker_max)
        else:
            stream_iter = iter(hf_ds)
            if self.max_samples is not None and self.max_samples > 0:
                stream_iter = itertools.islice(stream_iter, self.max_samples)

        return stream_iter

    def close(self) -> None:
        """Explicitly close the current active stream and raw prefetch iterator if supported."""
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

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        import random
        self.close()
        stream = self._get_stream()
        self._current_stream = stream
        # Asynchronously prefetch raw samples from remote stream to overlap Parquet row group
        # extraction with image transformation and decode on the CPU
        prefetch_stream = _RawSamplePrefetchIterator(stream, maxsize=64)
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
    When max_samples is None or <= 0, uses the full dataset until depletion.
    """

    def __init__(
        self,
        dataset_name: str = "saberzl/SID_Set",
        split: str = "train",
        transform: Optional[JointCompose] = None,
        max_samples: Optional[int] = None,
        target_image_size: Tuple[int, int] = (256, 256),
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.requested_split = split
        self.transform = transform
        self.target_image_size = target_image_size
        self.max_samples = None if (max_samples is not None and max_samples <= 0) else max_samples

        if is_mock_dataset(dataset_name):
            self.resolved_split = split
            self.split = split
            self._is_mock = True
            self._mock_len = self.max_samples if (self.max_samples is not None and self.max_samples > 0) else 100
            self.data = None
            return

        self._is_mock = False
        hf_ds, resolved = load_hf_dataset_robust(dataset_name, requested_split=split, streaming=False)
        self.resolved_split = resolved
        self.split = resolved

        if self.max_samples is not None and 0 < self.max_samples < len(hf_ds):
            hf_ds = hf_ds.select(range(self.max_samples))
        self.data = hf_ds

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
            return process_raw_sample(
                raw_sample,
                transform=self.transform,
                target_image_size=self.target_image_size,
            )
        raw_sample = self.data[idx]
        return process_raw_sample(
            raw_sample,
            transform=self.transform,
            target_image_size=self.target_image_size,
        )


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
    num_workers = int(config.data.get("num_workers", 2))
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

    if streaming:
        eval_dataset = SIDStreamingDataset(
            dataset_name=dataset_name,
            split=eval_split,
            transform=transform,
            shuffle_buffer_size=0,
            max_samples=eval_max_samples,
            seed=seed,
            target_image_size=image_size,
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
        )
        return DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            multiprocessing_context=mp_context,
            worker_init_fn=worker_init_fn,
        )


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
    dataset_name = config.data.get("dataset_name", "saberzl/SID_Set")
    if is_mock_dataset(dataset_name) or bool(config.data.get("mock", False)):
        dataset_name = "mock"
    streaming = bool(config.data.get("streaming", True))
    batch_size = int(config.data.get("batch_size", 16))
    num_workers = int(config.data.get("num_workers", 2))
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

    if streaming:
        train_dataset = SIDStreamingDataset(
            dataset_name=dataset_name,
            split=train_split,
            transform=train_transform,
            shuffle_buffer_size=shuffle_buffer,
            max_samples=train_max_samples,
            seed=seed,
            target_image_size=image_size,
        )
        val_dataset = SIDStreamingDataset(
            dataset_name=dataset_name,
            split=val_split,
            transform=val_transform,
            shuffle_buffer_size=0,
            max_samples=val_max_samples,
            seed=seed,
            target_image_size=image_size,
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
        )
        val_dataset = SIDMapDataset(
            dataset_name=dataset_name,
            split=val_split,
            transform=val_transform,
            max_samples=val_max_samples,
            target_image_size=image_size,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            multiprocessing_context=mp_context,
            worker_init_fn=worker_init_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            multiprocessing_context=mp_context,
            worker_init_fn=worker_init_fn,
        )

    if include_test:
        test_loader = create_test_dataloader(config)
        return train_loader, val_loader, test_loader

    return train_loader, val_loader
