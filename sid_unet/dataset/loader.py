"""
Dataset loading pipeline for SID_Set and common image manipulation datasets (e.g. KhangTruong/IMD2020).
Supports both streaming (IterableDataset) and non-streaming (MapDataset) modes with
robust handling of label 0 (black mask), label 1 (white mask), label 2 (provided mask),
and 2-column image/mask datasets.
"""

from __future__ import annotations

import itertools
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from datasets import load_dataset as hf_load_dataset

from sid_unet.dataset.mask_utils import ensure_rgb_image, process_sample_mask
from sid_unet.dataset.transforms import get_transforms, JointCompose


def process_raw_sample(
    sample: Dict[str, Any],
    transform: Optional[JointCompose],
    target_image_size: Tuple[int, int] = (256, 256),
) -> Dict[str, Any]:
    """
    Process a single raw sample from datasets such as KhangTruong/IMD2020 or saberzl/SID_Set into model tensors.
    Handles 2-column format (image, mask) as well as labeled format (image, label, mask, img_id).
    """
    raw_img = sample.get("image")
    raw_label = sample.get("label", None)
    raw_mask = sample.get("mask", None)
    img_id = str(sample.get("img_id", sample.get("id", "")))

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


def get_split_candidates(requested_split: str) -> List[str]:
    """
    Return priority list of split name candidates for a requested split.
    Handles synonyms and sensible fallbacks (e.g. test -> validation -> val -> train).
    """
    req_lower = requested_split.strip().lower()
    if req_lower in ("test", "testing", "eval", "evaluation"):
        candidates = [requested_split, "test", "testing", "validation", "val", "eval", "evaluation", "train"]
    elif req_lower in ("val", "validation", "valid", "dev"):
        candidates = [requested_split, "validation", "val", "valid", "dev", "test", "testing", "eval", "train"]
    elif req_lower in ("train", "training"):
        candidates = [requested_split, "train", "training", "train_data", "train_set"]
    else:
        candidates = [requested_split, "test", "validation", "val", "train"]

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

    def _get_stream(self) -> Iterator[Dict[str, Any]]:
        # Load streamed dataset with robust fallback
        hf_ds, resolved = load_hf_dataset_robust(self.dataset_name, requested_split=self.split, streaming=True)
        self.resolved_split = resolved

        if self.shuffle_buffer_size > 0 and resolved.lower() in ("train", "training"):
            hf_ds = hf_ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)

        worker_info = get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            # Multi-worker splitting for iterable dataset
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            stream_iter = itertools.islice(hf_ds, worker_id, None, num_workers)
        else:
            stream_iter = iter(hf_ds)

        if self.max_samples is not None and self.max_samples > 0:
            stream_iter = itertools.islice(stream_iter, self.max_samples)

        return stream_iter

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        stream = self._get_stream()
        try:
            for raw_sample in stream:
                try:
                    yield process_raw_sample(
                        raw_sample,
                        transform=self.transform,
                        target_image_size=self.target_image_size,
                    )
                except Exception:
                    # Silently skip corrupted samples in stream
                    continue
        finally:
            if hasattr(stream, "close") and callable(getattr(stream, "close", None)):
                try:
                    stream.close()
                except Exception:
                    pass
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

        hf_ds, resolved = load_hf_dataset_robust(dataset_name, requested_split=split, streaming=False)
        self.resolved_split = resolved
        self.split = resolved

        if self.max_samples is not None and 0 < self.max_samples < len(hf_ds):
            hf_ds = hf_ds.select(range(self.max_samples))
        self.data = hf_ds

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
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
) -> DataLoader:
    """
    Create a DataLoader specifically for evaluation or testing on a given split.
    Automatically resolves test vs validation split with robust fallback.

    Args:
        config: ConfigDict or configuration object.
        split: Split name (e.g. 'test', 'validation', 'val').
               If None, defaults to config.data.get('eval_split', config.data.get('test_split', 'test')).
        max_samples: Optional sample count limit (overrides config).
    """
    dataset_name = config.data.get("dataset_name", "KhangTruong/IMD2020")
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
        sample_key = "test_samples" if "test" in str(eval_split).lower() else "val_samples"
        samples_cfg = config.data.get(sample_key, config.data.get("val_samples", -1))
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
        return DataLoader(
            eval_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            multiprocessing_context=mp_context,
        )
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

    val_samples_cfg = config.data.get("val_samples", 400)
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

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            multiprocessing_context=mp_context,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            multiprocessing_context=mp_context,
        )
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
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            multiprocessing_context=mp_context,
        )

    if include_test:
        test_loader = create_test_dataloader(config)
        return train_loader, val_loader, test_loader

    return train_loader, val_loader
