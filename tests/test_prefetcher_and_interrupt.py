import os
import tempfile
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from sid_unet.dataset.loader import BackgroundPrefetcher
from sid_unet.training.trainer import Trainer
from sid_unet.utils.config import load_config


class SimpleDataset(Dataset):
    def __init__(self, size=12):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"data": torch.tensor([idx])}


class FailingDataset(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, idx):
        if idx == 2:
            raise RuntimeError("Corrupted sample error")
        return {"data": torch.tensor([idx])}


class InterruptDataset(Dataset):
    def __init__(self, size=8, img_size=(64, 64)):
        self.size = size
        self.img_size = img_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx == 2:
            raise KeyboardInterrupt("Simulated Ctrl+C")
        lbl = idx % 3
        img = torch.randn(3, *self.img_size)
        mask = torch.zeros(1, *self.img_size)
        return {
            "image": img,
            "mask": mask,
            "label": torch.tensor(lbl, dtype=torch.long),
            "img_id": f"syn_{idx}",
        }


def test_background_prefetcher_basic_and_multi_epoch():
    ds = SimpleDataset(size=12)
    raw_loader = DataLoader(ds, batch_size=3)
    prefetcher = BackgroundPrefetcher(raw_loader, maxsize=4)

    assert len(prefetcher) == 4
    assert prefetcher.batch_size == 3
    assert prefetcher.dataset is ds

    for _ in range(2):
        collected = []
        for batch in prefetcher:
            collected.extend(batch["data"].flatten().tolist())
        assert collected == list(range(12))

    prefetcher.close()


def test_background_prefetcher_exception_propagation():
    ds = FailingDataset()
    raw_loader = DataLoader(ds, batch_size=1)
    prefetcher = BackgroundPrefetcher(raw_loader, maxsize=2)

    with pytest.raises(RuntimeError, match="Corrupted sample error"):
        for _ in prefetcher:
            pass

    prefetcher.close()


def test_trainer_keyboard_interrupt_emergency_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = load_config(overrides=[
            f"project.output_dir={tmpdir}",
            "project.device=cpu",
            "training.epochs=2",
            "training.batch_size=2",
            "model.features=[16, 32]",
            "data.image_size=[64, 64]",
            "logging.log_interval=1",
            "logging.save_sample_images=false",
            "training.amp=false",
        ])

        train_ds = InterruptDataset(size=6, img_size=(64, 64))
        val_ds = InterruptDataset(size=2, img_size=(64, 64))
        train_loader = DataLoader(train_ds, batch_size=2)
        val_loader = DataLoader(val_ds, batch_size=2)

        trainer = Trainer(
            config=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        with pytest.raises(KeyboardInterrupt):
            trainer.train()

        # Verify emergency checkpoint was saved
        ckpt_latest = os.path.join(tmpdir, "checkpoints", "checkpoint_latest.pt")
        ckpt_periodic = os.path.join(tmpdir, "checkpoints", "checkpoint_periodic.pt")
        assert os.path.exists(ckpt_latest), "checkpoint_latest.pt must exist after KeyboardInterrupt"
        assert os.path.exists(ckpt_periodic), "checkpoint_periodic.pt must exist after KeyboardInterrupt"


def test_raw_sample_prefetch_iterator_basic_and_close():
    from sid_unet.dataset.loader import _RawSamplePrefetchIterator

    def sample_gen(n=20):
        for i in range(n):
            yield {"id": i, "data": f"sample_{i}"}

    iterator = _RawSamplePrefetchIterator(sample_gen(20), maxsize=8)
    collected = []
    for item in iterator:
        collected.append(item["id"])
    assert collected == list(range(20))
    iterator.close()


def test_raw_sample_prefetch_iterator_exception_propagation():
    from sid_unet.dataset.loader import _RawSamplePrefetchIterator

    def failing_gen():
        for i in range(5):
            if i == 3:
                raise ValueError("Remote stream read failure")
            yield {"id": i}

    iterator = _RawSamplePrefetchIterator(failing_gen(), maxsize=4)
    with pytest.raises(ValueError, match="Remote stream read failure"):
        for _ in iterator:
            pass
    iterator.close()


def test_raw_sample_prefetch_iterator_absorbs_row_group_stall():
    import time
    from sid_unet.dataset.loader import _RawSamplePrefetchIterator

    def stalling_gen(n=16, stall_idx=8, stall_duration=0.15):
        for i in range(n):
            if i == stall_idx:
                time.sleep(stall_duration)
            yield {"id": i}

    iterator = _RawSamplePrefetchIterator(stalling_gen(), maxsize=16)
    # Slow consumer that takes 0.03s per sample (simulating CPU image decode)
    delays = []
    for _ in iterator:
        t0 = time.time()
        time.sleep(0.03)
        delays.append(time.time() - t0)

    assert len(delays) == 16
    iterator.close()


def test_parquet_fast_generate_tables_patch_and_readahead_params(monkeypatch):
    """Verify that Parquet._generate_tables is patched to use _hf_xopen and PyArrow async readahead."""
    from datasets.packaged_modules.parquet.parquet import Parquet, ParquetConfig
    import pyarrow as pa
    import pyarrow.dataset as ds

    # Verify that Parquet._generate_tables is indeed the fast patched version
    from sid_unet.dataset.loader import _fast_hf_generate_tables
    assert Parquet._generate_tables == _fast_hf_generate_tables

    # Create dummy parquet table and fragment
    dummy_table = pa.Table.from_pydict({"image": [b"img1", b"img2"], "label": [0, 1]})
    record_batch = dummy_table.to_batches()[0]

    captured_to_batches_kwargs = {}

    class DummyFragment:
        def __init__(self):
            self.row_groups = [type("RG", (), {"num_rows": 2})()]

        def subset(self, row_group_ids):
            return self

        def to_batches(self, **kwargs):
            captured_to_batches_kwargs.update(kwargs)
            yield record_batch

    class DummyFileFormat:
        def make_fragment(self, f):
            return DummyFragment()

    monkeypatch.setattr(ds, "ParquetFileFormat", lambda *a, **kw: DummyFileFormat())

    # Create dummy builder instance
    builder = Parquet.__new__(Parquet)
    builder.config = ParquetConfig(name="test_config")
    builder.config.batch_size = None
    builder.config.columns = None
    builder.config.features = None
    builder.config.filters = None
    builder.config.fragment_scan_options = None
    builder.config.on_bad_files = "error"
    builder.info = type("Info", (), {"features": None})()

    # Call _fast_hf_generate_tables on a mock file
    with tempfile.NamedTemporaryFile("wb") as tmp_f:
        tmp_f.write(b"dummy")
        tmp_f.flush()

        results = list(builder._generate_tables([tmp_f.name], [None]))
        assert len(results) == 1
        key, yielded_table = results[0]
        assert len(yielded_table) == 2

    # Verify that multi-layer async prefetching arguments were passed to PyArrow
    assert captured_to_batches_kwargs.get("batch_readahead") == 2
    assert captured_to_batches_kwargs.get("fragment_readahead") == 1
    assert captured_to_batches_kwargs.get("use_threads") is True


def test_hf_file_system_file_init_blockcache_options(monkeypatch):
    """Verify that HfFileSystemFile is configured with blockcache, 16MB blocks, and cleans nblocks."""
    import sid_unet.dataset.loader as loader_module

    recorded_kwargs = {}

    def mock_orig_init(self, *args, **kwargs):
        recorded_kwargs.update(kwargs)
        return None

    monkeypatch.setattr(loader_module, "_orig_hf_file_init", mock_orig_init)

    dummy_self = type("DummyHfFile", (), {})()
    loader_module._fast_hf_file_init(dummy_self, fs=None, path="test.parquet", cache_options={"nblocks": 10})

    assert recorded_kwargs.get("cache_type") == "blockcache"
    assert recorded_kwargs.get("block_size") == 16 * 1024 * 1024
    assert recorded_kwargs.get("cache_options", {}).get("maxblocks") == 64
    assert "nblocks" not in recorded_kwargs.get("cache_options", {})


