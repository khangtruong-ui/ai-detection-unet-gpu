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
        "training.data_parallel=false",
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


def test_pdl_dataloader_exact_max_len_and_multi_epoch_resilience(temp_parquet_dataset):
    """
    Test that PDLDataLoader exposes exact batch length (eliminating 'no max len')
    and cleanly supports multi-epoch iteration even when close() is invoked intra-epoch.
    """
    from sid_unet.dataset.loader import PDLDataLoader

    cfg = load_config(overrides=[
        f"data.dataset_name={temp_parquet_dataset}",
        "data.train_split=train",
        "data.val_split=train",
        "data.streaming=true",
        "data.batch_size=4",
        "training.data_parallel=false",
        "data.train_samples_per_epoch=12",
        "data.val_samples_per_epoch=8",
        "data.image_size=[64, 64]",
        "data.num_workers=0",
        "data.use_parquet_loader=true",
    ])

    train_l, val_l = create_dataloaders(cfg, include_test=False)

    # 1. Verify exact length is immediately available
    assert isinstance(train_l, PDLDataLoader)
    assert isinstance(train_l, DataLoader)
    assert len(train_l) == 3   # 12 samples // batch_size 4
    assert len(val_l) == 2     # 8 samples // batch_size 4

    # 2. Iterate Epoch 1
    epoch_1_batches = list(train_l)
    assert len(epoch_1_batches) == 3

    # 3. Simulate intra-epoch close() call
    train_l.close()

    # 4. Iterate Epoch 2 without RuntimeError: Cannot iterate over a closed DataLoader.
    epoch_2_batches = list(train_l)
    assert len(epoch_2_batches) == 3


def test_sid_map_dataset_row_group_locality_and_block_shuffled_sampler(temp_parquet_dataset):
    """
    Verify SIDMapDataset delegates get_row_group_indices and get_row_group_file_indices,
    and PDLDataLoader automatically selects BlockShuffledSampler (avoiding RandomSampler cache thrashing).
    """
    from sid_unet.dataset.loader import PDLDataLoader, SIDMapDataset
    from parquet_dataset_loader.dataset import BlockShuffledSampler

    map_ds = SIDMapDataset(
        dataset_name=temp_parquet_dataset,
        split="train",
        target_image_size=(64, 64),
        use_parquet_loader=True,
        save_to_disk=True,
    )

    assert hasattr(map_ds, "get_row_group_indices")
    groups = map_ds.get_row_group_indices()
    assert isinstance(groups, list)
    assert len(groups) > 0

    assert hasattr(map_ds, "get_row_group_file_indices")
    f_indices = map_ds.get_row_group_file_indices()
    assert f_indices is not None

    loader = PDLDataLoader(
        dataset=map_ds,
        batch_size=4,
        shuffle=True,
        seed=42,
        num_workers=0,
    )

    assert isinstance(loader.sampler, BlockShuffledSampler)
    batches = list(loader)
    assert len(batches) == 10  # 40 samples // 4
    map_ds.close()


def test_create_dataloaders_with_max_cache_gb_forwarding(temp_parquet_dataset):
    """Verify max_cache_gb and cache_files config parameters are cleanly forwarded to parquet-dataset-loader."""
    cfg = load_config(overrides=[
        f"data.dataset_name={temp_parquet_dataset}",
        "data.train_split=train",
        "data.val_split=train",
        "data.streaming=true",
        "data.save_to_disk=true",
        "data.max_cache_gb=2.0",
        "data.cache_files=true",
        "data.batch_size=4",
        "data.train_samples_per_epoch=8",
        "data.val_samples_per_epoch=4",
        "data.image_size=[64, 64]",
        "data.num_workers=0",
        "data.use_parquet_loader=true",
        "training.data_parallel=false",
    ])

    train_l, val_l = create_dataloaders(cfg, include_test=False)
    b_train = next(iter(train_l))
    assert b_train["image"].shape == (4, 3, 64, 64)
    assert b_train["mask"].shape == (4, 1, 64, 64)


def test_pdl_dataloader_timeout_configuration(temp_parquet_dataset):
    """Verify that timeout configuration is properly propagated to PDLDataLoader."""
    from sid_unet.dataset.loader import PDLDataLoader

    # 1. Default timeout should be 300.0
    cfg_default = load_config(overrides=[
        f"data.dataset_name={temp_parquet_dataset}",
        "data.train_split=train",
        "data.val_split=train",
        "data.streaming=true",
        "data.batch_size=4",
        "training.data_parallel=false",
        "data.train_samples_per_epoch=8",
        "data.val_samples_per_epoch=4",
        "data.image_size=[64, 64]",
        "data.num_workers=0",
        "data.use_parquet_loader=true",
    ])
    train_l, val_l = create_dataloaders(cfg_default, include_test=False)
    assert isinstance(train_l, PDLDataLoader)
    assert train_l.timeout == 300.0
    assert val_l.timeout == 300.0

    eval_l = create_eval_dataloader(cfg_default, split="train", max_samples=4)
    assert isinstance(eval_l, PDLDataLoader)
    assert eval_l.timeout == 300.0

    # 2. Explicit custom timeout should be honored
    cfg_custom = load_config(overrides=[
        f"data.dataset_name={temp_parquet_dataset}",
        "data.train_split=train",
        "data.val_split=train",
        "data.streaming=true",
        "data.batch_size=4",
        "training.data_parallel=false",
        "data.train_samples_per_epoch=8",
        "data.val_samples_per_epoch=4",
        "data.image_size=[64, 64]",
        "data.num_workers=0",
        "data.use_parquet_loader=true",
        "data.timeout=600.0",
    ])
    train_custom, val_custom = create_dataloaders(cfg_custom, include_test=False)
    assert train_custom.timeout == 600.0
    assert val_custom.timeout == 600.0

    eval_custom = create_eval_dataloader(cfg_custom, split="train", max_samples=4)
    assert eval_custom.timeout == 600.0


def test_load_parquet_dataset_fail_fast_on_import_error(monkeypatch):
    """
    Verify that when pdl.load_dataset raises an ImportError (e.g. missing aiohttp),
    load_parquet_dataset immediately raises the ImportError rather than swallowing it
    and masking it with a downstream SplitNotFoundError on 'train'.
    """
    import parquet_dataset_loader as pdl

    calls = []
    def mock_load_dataset(*args, **kwargs):
        split = kwargs.get("split")
        calls.append(split)
        raise ImportError('HTTPFileSystem requires "requests" and "aiohttp" to be installed')

    monkeypatch.setattr(pdl, "load_dataset", mock_load_dataset)

    with pytest.raises(ImportError, match='HTTPFileSystem requires'):
        load_parquet_dataset("mock/dataset", requested_split="cross_test")

    # Must fail on the first split attempt and NOT iterate down to 'train'
    assert calls == ["cross_test"]


def test_load_parquet_dataset_preserves_requested_split_primary_error(monkeypatch):
    """
    Verify that when all candidate splits fail with SplitNotFoundError,
    the primary error for the requested split ('cross_test') is raised,
    rather than a confusing error referring to the fallback candidate ('train').
    """
    import parquet_dataset_loader as pdl
    from parquet_dataset_loader import SplitNotFoundError

    def mock_load_dataset(*args, **kwargs):
        split = kwargs.get("split")
        raise SplitNotFoundError(split, ["available_split_only"])

    monkeypatch.setattr(pdl, "load_dataset", mock_load_dataset)

    with pytest.raises(SplitNotFoundError) as exc_info:
        load_parquet_dataset("mock/dataset", requested_split="cross_test")

    # The exception must reference 'cross_test', NOT 'train'
    assert "cross_test" in str(exc_info.value)
    assert "train" not in str(exc_info.value)


def test_sid_map_dataset_hf_iterable_fallback_handling(monkeypatch):
    """
    Verify that SIDMapDataset correctly detects when Hugging Face fallback
    returns an IterableDataset (which defines __getitem__ in datasets v2+ but lacks __len__),
    and safely reloads it non-streaming so it becomes an indexable MapDataset with valid len().
    """
    import datasets
    from datasets import IterableDataset

    class MockHFIterableDataset:
        def __init__(self):
            self._samples = [
                {"image": Image.new("RGB", (32, 32), (10, 20, 30)), "mask": Image.new("L", (32, 32), 0), "label": 0, "img_id": f"s_{i}"}
                for i in range(5)
            ]

        def __iter__(self):
            return iter(self._samples)

        def __getitem__(self, idx):
            return self._samples[idx]

    class MockHFMapDataset:
        def __init__(self):
            self._samples = [
                {"image": Image.new("RGB", (32, 32), (10, 20, 30)), "mask": Image.new("L", (32, 32), 0), "label": 0, "img_id": f"s_{i}"}
                for i in range(5)
            ]

        def __len__(self):
            return len(self._samples)

        def __getitem__(self, idx):
            return self._samples[idx]

        def select(self, indices):
            return [self._samples[i] for i in indices]

    # When streaming=True, return MockHFIterableDataset; when streaming=False, return MockHFMapDataset
    def mock_hf_load(dataset_name, split=None, streaming=False, **kwargs):
        if streaming:
            return MockHFIterableDataset()
        return MockHFMapDataset()

    monkeypatch.setattr("sid_unet.dataset.loader._HAS_PARQUET_LOADER", False)
    monkeypatch.setattr("sid_unet.dataset.loader.hf_load_dataset", mock_hf_load)

    # Initialize SIDMapDataset with use_parquet_loader=False to force HF fallback
    ds = SIDMapDataset(
        dataset_name="dummy/test_repo",
        split="cross_test",
        use_parquet_loader=False,
    )

    # Verify length and indexing work cleanly without TypeError
    assert len(ds) == 5
    sample = ds[0]
    assert "image" in sample
    assert "mask" in sample




