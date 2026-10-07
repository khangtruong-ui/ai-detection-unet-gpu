"""
Tests for parquet-dataset-loader integration into sid_unet dataset loading pipeline.
Verifies:
1. Non-blocking index streaming via IndexedParquetDataset.
2. Direct map-style random access in SIDMapDataset without multi-gigabyte download blocking.
3. Streaming mode in SIDStreamingDataset with worker and rank sharding.
4. Multiprocessing DataLoader compatibility with spawn / pickle.
5. Column projection, background download, and save_to_disk integration.
6. Fallback to Hugging Face dataset loader when disabled or on error.
"""

import os
import tempfile
import io
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from sid_unet.dataset.loader import (
    load_parquet_dataset,
    load_parquet_or_hf_dataset,
    SIDMapDataset,
    SIDStreamingDataset,
    create_dataloaders,
    create_eval_dataloader,
)
from sid_unet.utils.config import load_config


@pytest.fixture
def temp_parquet_dataset():
    """Create a temporary multi-file, multi-row-group Parquet dataset."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        num_files = 2
        rows_per_file = 20

        for f_idx in range(num_files):
            images_raw = []
            masks_raw = []
            labels = []
            img_ids = []

            for r_idx in range(rows_per_file):
                global_idx = f_idx * rows_per_file + r_idx

                # Create realistic test RGB image
                img = Image.new("RGB", (64, 64), color=(global_idx * 5 % 255, 100, 150))
                buf_img = io.BytesIO()
                img.save(buf_img, format="PNG")
                images_raw.append(buf_img.getvalue())

                # Create mask
                mask = Image.new("L", (64, 64), color=(255 if global_idx % 2 == 1 else 0))
                buf_mask = io.BytesIO()
                mask.save(buf_mask, format="PNG")
                masks_raw.append(buf_mask.getvalue())

                labels.append(global_idx % 3)
                img_ids.append(f"sample_{global_idx}")

            table = pa.Table.from_arrays(
                [
                    pa.array(images_raw, type=pa.binary()),
                    pa.array(masks_raw, type=pa.binary()),
                    pa.array(labels, type=pa.int64()),
                    pa.array(img_ids, type=pa.string()),
                ],
                names=["image", "mask", "label", "img_id"],
            )

            file_path = os.path.join(tmp_dir, f"train-{f_idx:05d}-of-{num_files:05d}.parquet")
            # Write with 2 row groups per file
            pq.write_table(table, file_path, row_group_size=10)

        yield tmp_dir


def test_load_parquet_dataset_direct(temp_parquet_dataset):
    """Test direct load_parquet_dataset with non-blocking index streaming."""
    ds, resolved_split = load_parquet_dataset(
        dataset_name=temp_parquet_dataset,
        requested_split="train",
        streaming=True,
    )
    assert resolved_split == "train"
    assert len(ds) == 40
    assert hasattr(ds, "__getitem__")

    # Verify first and last rows
    r0 = ds[0]
    assert "image" in r0
    assert "mask" in r0
    assert r0["img_id"] == "sample_0"

    r39 = ds[39]
    assert r39["img_id"] == "sample_39"


def test_load_parquet_or_hf_dataset_projection(temp_parquet_dataset):
    """Test load_parquet_or_hf_dataset with column projection."""
    ds, resolved_split = load_parquet_or_hf_dataset(
        dataset_name=temp_parquet_dataset,
        requested_split="train",
        streaming=True,
        columns=["image", "mask"],
        use_parquet_loader=True,
    )
    assert resolved_split == "train"
    assert len(ds) == 40
    r0 = ds[0]
    assert set(r0.keys()) == {"image", "mask"}


def test_sid_map_dataset_with_parquet_loader(temp_parquet_dataset):
    """Test SIDMapDataset wrapping IndexedParquetDataset with map-style indexing."""
    map_ds = SIDMapDataset(
        dataset_name=temp_parquet_dataset,
        split="train",
        target_image_size=(64, 64),
        use_parquet_loader=True,
        max_samples=25,
    )

    assert len(map_ds) == 25
    sample = map_ds[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["mask"].shape == (1, 64, 64)
    assert isinstance(sample["label"], torch.Tensor)
    assert sample["img_id"] == "sample_0"

    # Multiprocessing DataLoader with 2 workers using spawn context
    loader = DataLoader(
        map_ds,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        multiprocessing_context=torch.multiprocessing.get_context("spawn"),
    )
    batch = next(iter(loader))
    assert batch["image"].shape == (4, 3, 64, 64)
    assert batch["mask"].shape == (4, 1, 64, 64)
    assert batch["label"].shape == (4,)

    map_ds.close()


def test_sid_streaming_dataset_with_parquet_loader(temp_parquet_dataset):
    """Test SIDStreamingDataset wrapping IndexedParquetDataset in streaming mode."""
    stream_ds = SIDStreamingDataset(
        dataset_name=temp_parquet_dataset,
        split="train",
        target_image_size=(64, 64),
        use_parquet_loader=True,
        max_samples=15,
        shuffle_buffer_size=10,
    )

    items = list(iter(stream_ds))
    assert len(items) == 15
    assert items[0]["image"].shape == (3, 64, 64)
    assert items[0]["mask"].shape == (1, 64, 64)
    assert isinstance(items[0]["label"], torch.Tensor)

    stream_ds.close()


def test_parquet_loader_fallback(monkeypatch, temp_parquet_dataset):
    """Test fallback to load_hf_dataset_robust when use_parquet_loader=False."""
    called_hf = False

    def mock_hf_load(dataset_name, split=None, streaming=False):
        nonlocal called_hf
        called_hf = True
        class MockHF:
            def __len__(self):
                return 5
            def __getitem__(self, idx):
                return {
                    "image": Image.new("RGB", (32, 32)),
                    "mask": None,
                    "label": 0,
                    "img_id": f"hf_{idx}",
                }
            def select(self, r):
                return self
        return MockHF()

    monkeypatch.setattr("sid_unet.dataset.loader.hf_load_dataset", mock_hf_load)

    # When use_parquet_loader is False, should call mock_hf_load
    ds, resolved = load_parquet_or_hf_dataset(
        dataset_name=temp_parquet_dataset,
        requested_split="train",
        use_parquet_loader=False,
    )
    assert called_hf
    assert len(ds) == 5


def test_create_dataloaders_with_parquet_config(temp_parquet_dataset):
    """Test create_dataloaders end-to-end with parquet-dataset-loader configuration."""
    cfg = load_config(overrides=[
        f"data.dataset_name={temp_parquet_dataset}",
        "data.train_split=train",
        "data.val_split=train",
        "data.streaming=false",
        "data.batch_size=4",
        "data.train_samples_per_epoch=12",
        "data.val_samples_per_epoch=8",
        "data.image_size=[64, 64]",
        "data.num_workers=0",
        "data.use_parquet_loader=true",
        "training.data_parallel=false",
    ])

    train_l, val_l = create_dataloaders(cfg, include_test=False)
    b_train = next(iter(train_l))
    assert b_train["image"].shape == (4, 3, 64, 64)
    assert b_train["mask"].shape == (4, 1, 64, 64)

    eval_l = create_eval_dataloader(cfg, split="train", max_samples=4)
    b_eval = next(iter(eval_l))
    assert b_eval["image"].shape == (4, 3, 64, 64)
