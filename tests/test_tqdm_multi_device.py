import sys
import pytest
import torch
from unittest.mock import MagicMock
from torch.utils.data import DataLoader, Dataset
from sid_unet.utils.config import load_config
from sid_unet.training.trainer import Trainer


class DummyDataset(Dataset):
    def __init__(self, size: int = 4):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "image": torch.randn(3, 64, 64),
            "mask": torch.zeros(1, 64, 64),
            "label": torch.tensor(0, dtype=torch.long),
        }


def test_tqdm_train_and_validate_kwargs(monkeypatch, tmp_path):
    """Verify tqdm is called with file=sys.stdout, dynamic_ncols=True, leave=True in train_epoch, and leave=False in intra-epoch validate."""
    captured_kwargs = []
    import tqdm

    orig_tqdm = tqdm.tqdm

    class MockTqdm(orig_tqdm):
        def __init__(self, *args, **kwargs):
            captured_kwargs.append(dict(kwargs))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(tqdm, "tqdm", MockTqdm)
    import sid_unet.training.trainer as trainer_mod
    monkeypatch.setattr(trainer_mod, "tqdm", MockTqdm)

    cfg = load_config(overrides=[
        f"project.output_dir={tmp_path}",
        "project.device=cpu",
        "training.epochs=1",
        "training.batch_size=2",
        "model.features=[16, 32]",
        "data.image_size=[64, 64]",
        "logging.log_interval=1",
        "training.amp=false",
    ])

    loader = DataLoader(DummyDataset(4), batch_size=2)
    trainer = Trainer(config=cfg, train_loader=loader, val_loader=loader)

    # 1. Train epoch
    trainer.train_epoch(1)
    assert len(captured_kwargs) >= 1
    train_kwargs = captured_kwargs[0]
    assert train_kwargs.get("file") is sys.stdout
    assert train_kwargs.get("dynamic_ncols") is True
    assert train_kwargs.get("leave") is True
    assert train_kwargs.get("disable") is False

    # 2. Intra-epoch validate
    captured_kwargs.clear()
    trainer.validate(epoch=1)
    assert len(captured_kwargs) >= 1
    val_kwargs = captured_kwargs[0]
    assert val_kwargs.get("file") is sys.stdout
    assert val_kwargs.get("dynamic_ncols") is True
    assert val_kwargs.get("leave") is False  # intra-epoch validate should not leave bar
    assert val_kwargs.get("disable") is False

    # 3. Standalone evaluate
    captured_kwargs.clear()
    trainer.evaluate(loader=loader)
    assert len(captured_kwargs) >= 1
    eval_kwargs = captured_kwargs[0]
    assert eval_kwargs.get("file") is sys.stdout
    assert eval_kwargs.get("dynamic_ncols") is True
    assert eval_kwargs.get("leave") is True  # standalone evaluate leaves bar

    trainer.close()


def test_tqdm_disabled_on_non_main_process(monkeypatch, tmp_path):
    """Verify tqdm is disabled when is_main_process is False."""
    captured_kwargs = []
    import tqdm
    orig_tqdm = tqdm.tqdm

    class MockTqdm(orig_tqdm):
        def __init__(self, *args, **kwargs):
            captured_kwargs.append(dict(kwargs))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(tqdm, "tqdm", MockTqdm)
    import sid_unet.training.trainer as trainer_mod
    monkeypatch.setattr(trainer_mod, "tqdm", MockTqdm)

    cfg = load_config(overrides=[
        f"project.output_dir={tmp_path}",
        "project.device=cpu",
        "training.epochs=1",
        "training.batch_size=2",
        "model.features=[16, 32]",
        "data.image_size=[64, 64]",
        "logging.log_interval=1",
        "training.amp=false",
    ])

    loader = DataLoader(DummyDataset(4), batch_size=2)
    trainer = Trainer(config=cfg, train_loader=loader, val_loader=loader)
    trainer.is_main_process = False

    trainer.train_epoch(1)
    assert len(captured_kwargs) >= 1
    assert captured_kwargs[0].get("disable") is True

    captured_kwargs.clear()
    trainer.validate(epoch=1)
    assert len(captured_kwargs) >= 1
    assert captured_kwargs[0].get("disable") is True

    trainer.close()


def test_validate_sync_scalar_min_called_in_distributed(monkeypatch, tmp_path):
    """Verify safe_dataloader_len is synchronized across ranks in distributed validate."""
    sync_called = []

    def mock_sync(val):
        sync_called.append(val)
        return val

    import sid_unet.utils.distributed as dist_mod
    monkeypatch.setattr(dist_mod, "sync_scalar_min", mock_sync)

    cfg = load_config(overrides=[
        f"project.output_dir={tmp_path}",
        "project.device=cpu",
        "training.epochs=1",
        "training.batch_size=2",
        "model.features=[16, 32]",
        "data.image_size=[64, 64]",
        "logging.log_interval=1",
        "training.amp=false",
    ])

    loader = DataLoader(DummyDataset(4), batch_size=2)
    trainer = Trainer(config=cfg, train_loader=loader, val_loader=loader)
    trainer.is_distributed = True
    monkeypatch.setattr("sid_unet.training.trainer.is_dist_avail_and_initialized", lambda: True)

    trainer.validate(epoch=1)
    assert len(sync_called) >= 1
    assert sync_called[0] == 2

    trainer.close()
