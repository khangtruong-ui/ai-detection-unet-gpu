import os
import tempfile
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from sid_unet.models.unet import UNet
from sid_unet.losses.auxiliary import SIDTotalLoss
from sid_unet.training.trainer import Trainer
from sid_unet.utils.config import load_config


class SyntheticDataset(Dataset):
    def __init__(self, size=8, img_size=(64, 64)):
        self.size = size
        self.img_size = img_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        lbl = idx % 3
        img = torch.randn(3, *self.img_size)
        if lbl == 0:
            mask = torch.zeros(1, *self.img_size)
        elif lbl == 1:
            mask = torch.ones(1, *self.img_size)
        else:
            mask = torch.zeros(1, *self.img_size)
            mask[:, : self.img_size[0] // 2, :] = 1.0

        return {
            "image": img,
            "mask": mask,
            "label": torch.tensor(lbl, dtype=torch.long),
            "img_id": f"syn_{idx}",
        }


def test_trainer_mini_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = load_config(overrides=[
            f"project.output_dir={tmpdir}",
            "project.device=cpu",
            "training.epochs=1",
            "training.batch_size=2",
            "model.features=[16, 32]",
            "data.image_size=[64, 64]",
            "logging.log_interval=1",
            "logging.save_sample_images=false",
            "training.amp=false",
        ])

        train_ds = SyntheticDataset(size=4, img_size=(64, 64))
        val_ds = SyntheticDataset(size=2, img_size=(64, 64))

        train_loader = DataLoader(train_ds, batch_size=2)
        val_loader = DataLoader(val_ds, batch_size=2)

        trainer = Trainer(
            config=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        results = trainer.train()
        assert "best_score" in results
        assert not os.path.exists(os.path.join(tmpdir, "checkpoints", "checkpoint_latest.pt"))
        assert os.path.exists(os.path.join(tmpdir, "checkpoints", "checkpoint_best.pt"))
        assert os.path.exists(os.path.join(tmpdir, "checkpoints", "checkpoint_best_config.yaml"))
        assert os.path.exists(os.path.join(tmpdir, "reports", "training_final_report.json"))
        assert os.path.exists(os.path.join(tmpdir, "reports", "training_final_report.md"))
        assert os.path.exists(os.path.join(tmpdir, "reports", "training_curves.png"))
        assert os.path.exists(os.path.join(tmpdir, "reports", "training_curves.pdf"))
        assert os.path.exists(os.path.join(tmpdir, "reports", "training_curves.jpg"))
        assert os.path.exists(os.path.join(tmpdir, "reports", "training_history.json"))
        assert os.path.exists(os.path.join(tmpdir, "reports", "training_history.csv"))


def test_trainer_with_test_loader():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = load_config(overrides=[
            f"project.output_dir={tmpdir}",
            "project.device=cpu",
            "training.epochs=1",
            "training.batch_size=2",
            "model.features=[16, 32]",
            "data.image_size=[64, 64]",
            "logging.log_interval=1",
            "logging.save_sample_images=false",
            "training.amp=false",
        ])

        train_ds = SyntheticDataset(size=4, img_size=(64, 64))
        val_ds = SyntheticDataset(size=2, img_size=(64, 64))
        test_ds = SyntheticDataset(size=2, img_size=(64, 64))

        train_loader = DataLoader(train_ds, batch_size=2)
        val_loader = DataLoader(val_ds, batch_size=2)
        test_loader = DataLoader(test_ds, batch_size=2)

        trainer = Trainer(
            config=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )

        results = trainer.train()
        assert results["test_results"] is not None
        assert "test_report_path" in results
        assert os.path.exists(os.path.join(tmpdir, "reports", "test_evaluation_report.md"))
        assert os.path.exists(os.path.join(tmpdir, "reports", "test_evaluation_report.json"))

