"""
Dataset loading pipeline for SID_Set.
Supports both streaming (IterableDataset) and non-streaming (MapDataset) modes with
robust handling of label 0 (black mask), label 1 (white mask), and label 2 (provided mask).
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, Iterator, Optional, Tuple, Union
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
    Process a single raw sample from saberzl/SID_Set into model tensors.
    """
    raw_img = sample.get("image")
    label = int(sample.get("label", 0))
    raw_mask = sample.get("mask", None)
    img_id = str(sample.get("img_id", ""))

    # 1. Convert to RGB PIL Image
    pil_image = ensure_rgb_image(raw_img)

    # 2. Synthesize / extract mask based on label & provided mask
    mask_arr = process_sample_mask(
        mask_input=raw_mask,
        label=label,
        image_size=pil_image.size,  # (width, height)
    )

    # 3. Apply joint transforms
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
        self.split = split
        self.transform = transform
        self.shuffle_buffer_size = shuffle_buffer_size
        self.max_samples = None if (max_samples is not None and max_samples <= 0) else max_samples
        self.seed = seed
        self.target_image_size = target_image_size

    def _get_stream(self) -> Iterator[Dict[str, Any]]:
        # Load streamed dataset from Hugging Face
        hf_ds = hf_load_dataset(self.dataset_name, split=self.split, streaming=True)

        if self.shuffle_buffer_size > 0 and self.split == "train":
            hf_ds = hf_ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)

        worker_info = get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            # Multi-worker splitting for iterable dataset
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            # Use islice or split
            stream_iter = itertools.islice(hf_ds, worker_id, None, num_workers)
        else:
            stream_iter = iter(hf_ds)

        if self.max_samples is not None and self.max_samples > 0:
            stream_iter = itertools.islice(stream_iter, self.max_samples)

        return stream_iter

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        stream = self._get_stream()
        for raw_sample in stream:
            try:
                yield process_raw_sample(
                    raw_sample,
                    transform=self.transform,
                    target_image_size=self.target_image_size,
                )
            except Exception as e:
                # Silently skip corrupted samples in stream
                continue


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
        self.transform = transform
        self.target_image_size = target_image_size
        self.max_samples = None if (max_samples is not None and max_samples <= 0) else max_samples
        hf_ds = hf_load_dataset(dataset_name, split=split, streaming=False)
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


def create_dataloaders(config: Any) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders based on configuration.
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
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
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
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader
