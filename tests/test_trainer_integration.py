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


class SyntheticIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, size=5, img_size=(64, 64)):
        self.size = size
        self.img_size = img_size

    def __iter__(self):
        for idx in range(self.size):
            lbl = idx % 3
            img = torch.randn(3, *self.img_size)
            mask = torch.zeros(1, *self.img_size)
            yield {
                "image": img,
                "mask": mask,
                "label": torch.tensor(lbl, dtype=torch.long),
                "img_id": f"syn_{idx}",
            }


def test_trainer_with_iterable_dataset_no_len():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = load_config(overrides=[
            f"project.output_dir={tmpdir}",
            "project.device=cpu",
            "training.epochs=1",
            "training.batch_size=2",
            "training.gradient_accumulation_steps=2",
            "model.features=[16, 32]",
            "data.image_size=[64, 64]",
            "logging.log_interval=1",
            "logging.save_sample_images=false",
            "training.amp=false",
        ])

        train_ds = SyntheticIterableDataset(size=5, img_size=(64, 64))
        val_ds = SyntheticIterableDataset(size=2, img_size=(64, 64))

        train_loader = DataLoader(train_ds, batch_size=2)
        val_loader = DataLoader(val_ds, batch_size=2)

        trainer = Trainer(
            config=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        results = trainer.train()
        assert "best_score" in results
        assert "final_metrics" in results
        assert "val_total_loss" in results["final_metrics"]


def test_parse_checkpoint_period():
    from sid_unet.training.trainer import parse_checkpoint_period

    assert parse_checkpoint_period({}) == 3600.0
    assert parse_checkpoint_period(None) == 3600.0
    assert parse_checkpoint_period({"checkpoint_period": 3600}) == 3600.0
    assert parse_checkpoint_period({"checkpoint_period": 1}) == 3600.0  # <= 24 treated as hours
    assert parse_checkpoint_period({"checkpoint_period": 0.5}) == 1800.0
    assert parse_checkpoint_period({"checkpoint_period": "2h"}) == 7200.0
    assert parse_checkpoint_period({"checkpoint_period": "30m"}) == 1800.0
    assert parse_checkpoint_period({"checkpoint_period": "45s"}) == 45.0
    assert parse_checkpoint_period({"checkpoint_period_hours": 1.5}) == 5400.0


def test_checkpoint_manager_periodic_save():
    import time
    from sid_unet.training.callbacks import CheckpointManager

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(checkpoint_dir=tmpdir, checkpoint_period=0.01)
        time.sleep(0.02)
        assert mgr.should_save_periodic() is True

        model = torch.nn.Linear(2, 2)
        paths = mgr.save_periodic(epoch=1, model=model, config={"test": 1}, step=10)
        assert os.path.exists(paths["periodic"])
        assert os.path.exists(paths["latest"])
        assert mgr.should_save_periodic() is False


def test_checkpoint_manager_load_with_unexpected_keys():
    from sid_unet.training.callbacks import CheckpointManager

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(checkpoint_dir=tmpdir)
        model = torch.nn.Linear(2, 2)

        ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
        state_dict = model.state_dict()
        # Simulate quantization metadata / extra LoRA keys
        state_dict["extra_quant_map.weight.absmax"] = torch.tensor([1.0])
        torch.save({"model_state_dict": state_dict, "epoch": 3}, ckpt_path)

        # Loading should gracefully fall back to strict=False instead of raising RuntimeError
        loaded_epoch = mgr.load_checkpoint(ckpt_path, model)
        assert loaded_epoch == 3


def test_trainer_periodic_checkpointing_in_loop():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = load_config(overrides=[
            f"project.output_dir={tmpdir}",
            "project.device=cpu",
            "training.epochs=1",
            "training.batch_size=2",
            "training.checkpoint_period=0.0001",  # Trigger immediately
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

        trainer.train()
        periodic_ckpt = os.path.join(tmpdir, "checkpoints", "checkpoint_periodic.pt")
        assert os.path.exists(periodic_ckpt)


def test_val_samples_per_epoch_resolves():
    from sid_unet.dataset.loader import resolve_sample_limit

    # Specified val_samples_per_epoch
    res = resolve_sample_limit(samples_val=50, steps_val=None, batch_size=1)
    assert res == 50

    # -1 means full dataset (None)
    res_unlimited = resolve_sample_limit(samples_val=-1, steps_val=None, batch_size=1)
    assert res_unlimited is None


def test_training_loop_iou_metric_and_no_vram_report():
    """Verify that training loop computes IoU, includes train_iou in history, and does not report VRAM in loop."""
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

        train_metrics = trainer.train_epoch(1)
        assert "iou" in train_metrics, f"Expected 'iou' in train_metrics, got: {train_metrics.keys()}"
        assert 0.0 <= train_metrics["iou"] <= 1.0

        results = trainer.train()
        history_0 = results["history"][0]
        assert "train_iou" in history_0, f"Expected 'train_iou' in history, got: {history_0.keys()}"
        assert 0.0 <= history_0["train_iou"] <= 1.0


def test_tqdm_bar_postfix_format_and_network_reporting(monkeypatch):
    """Verify tqdm bar reports exactly 1 loss, includes network speed, and omits redundant mask_loss."""
    import tempfile
    from unittest.mock import MagicMock
    import tqdm

    captured_postfix = []
    orig_tqdm = tqdm.tqdm

    class MockTqdm(orig_tqdm):
        def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
            if ordered_dict:
                captured_postfix.append(dict(ordered_dict))
            super().set_postfix(ordered_dict=ordered_dict, refresh=refresh, **kwargs)

    monkeypatch.setattr(tqdm, "tqdm", MockTqdm)
    import sid_unet.training.trainer as trainer_mod
    monkeypatch.setattr(trainer_mod, "tqdm", MockTqdm)

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
        train_loader = DataLoader(train_ds, batch_size=2)

        trainer = Trainer(
            config=cfg,
            train_loader=train_loader,
            val_loader=train_loader,
        )

        trainer.train_epoch(1)
        trainer.close()

        assert len(captured_postfix) > 0
        last_pf = captured_postfix[-1]
        # Verify 1 loss only
        assert "loss" in last_pf
        assert "mask_loss" not in last_pf
        # Verify network speed reporting
        assert "net" in last_pf
        assert "iou" in last_pf
        assert "lr" in last_pf





