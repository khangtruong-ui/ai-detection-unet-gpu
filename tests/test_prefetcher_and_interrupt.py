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
