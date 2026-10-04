import os
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sid_unet.dataset.loader import resolve_num_workers, resolve_batch_size, create_eval_dataloader
from sid_unet.training.trainer import Trainer
from sid_unet.utils.config import load_config, ConfigDict


class DummyDataset(Dataset):
    def __init__(self, size=16, in_channels=3, img_size=(32, 32)):
        self.size = size
        self.in_channels = in_channels
        self.img_size = img_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "image": torch.randn(self.in_channels, *self.img_size),
            "mask": torch.zeros((1, *self.img_size), dtype=torch.float32),
            "label": torch.tensor(0, dtype=torch.long),
            "sample_idx": idx,
        }


def test_resolve_num_workers():
    # Negative values or -1 should resolve to os.cpu_count() or sched_getaffinity
    expected_cpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    assert resolve_num_workers(-1) == expected_cpu
    assert resolve_num_workers(-2) == expected_cpu

    # Non-negative explicit values should be respected
    assert resolve_num_workers(0) == 0
    assert resolve_num_workers(2) == 2
    assert resolve_num_workers(8) == 8


def test_resolve_batch_size_multiplication():
    num_gpus = torch.cuda.device_count()

    # Case 1: Data parallel enabled
    cfg = ConfigDict({
        "data": {"batch_size": 4},
        "training": {"data_parallel": True},
        "model": {"load_in_4bit": False, "load_in_8bit": False},
    })
    bs = resolve_batch_size(cfg)
    if torch.cuda.is_available() and num_gpus > 1:
        assert bs == 4 * num_gpus
        assert cfg.data.batch_size == 4 * num_gpus
        assert cfg.data.get("_batch_size_scaled") is True

        # Test idempotency - calling again should NOT multiply again
        bs2 = resolve_batch_size(cfg)
        assert bs2 == 4 * num_gpus
        assert cfg.data.batch_size == 4 * num_gpus
    else:
        assert bs == 4

    # Case 2: Data parallel disabled
    cfg_no_dp = ConfigDict({
        "data": {"batch_size": 4},
        "training": {"data_parallel": False},
        "model": {"load_in_4bit": False, "load_in_8bit": False},
    })
    bs_no_dp = resolve_batch_size(cfg_no_dp)
    assert bs_no_dp == 4


def test_data_parallel_trainer_execution():
    """Verify Trainer with DataParallel on multi-GPU (if available).
    Ensures that:
    1. model is wrapped in nn.DataParallel when multiple GPUs exist
    2. forward and backward pass work correctly
    3. checkpoint saving strips 'module.' prefix so state dict is clean
    """
    num_gpus = torch.cuda.device_count()

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = DummyDataset(size=16)
        train_loader = DataLoader(dataset, batch_size=4, shuffle=False)
        val_loader = DataLoader(dataset, batch_size=4, shuffle=False)

        cfg = load_config("configs/test_smoke.yaml")
        cfg.project.output_dir = tmpdir
        cfg.data.num_workers = 0
        cfg.data.batch_size = 4
        cfg.training.epochs = 1
        cfg.training.data_parallel = True
        cfg.training.save_best = True
        cfg.training.save_latest = True

        trainer = Trainer(
            config=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        if torch.cuda.is_available() and num_gpus > 1:
            assert trainer.is_data_parallel is True
            assert isinstance(trainer.model, nn.DataParallel)
            # raw_model should access underlying UNet without 'module.'
            assert hasattr(trainer.raw_model, "inc")
        else:
            assert trainer.is_data_parallel is False

        # Run 1 epoch of training
        metrics = trainer.train_epoch(1)
        assert "total_loss" in metrics
        assert metrics["total_loss"] > 0

        # Save checkpoint and verify no 'module.' prefix in saved weights
        trainer.ckpt_manager.save(
            epoch=1,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            metrics={"val_iou": 0.5},
            config=trainer.config,
            is_best=True,
            hard_mining=trainer.hard_miner.state_dict(),
        )

        best_path = os.path.join(tmpdir, "checkpoints", "checkpoint_best.pt")
        assert os.path.exists(best_path)
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        model_state = ckpt["model_state_dict"]
        for key in model_state.keys():
            assert not key.startswith("module."), f"Found module. prefix in key: {key}"


def _ddp_trainer_worker(rank, world_size, port, tmpdir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    from sid_unet.utils.distributed import (
        is_dist_avail_and_initialized,
        get_rank,
        get_world_size,
        is_main_process,
        reduce_dict,
        broadcast_scalar,
        cleanup_distributed,
    )
    from torch.utils.data.distributed import DistributedSampler

    assert is_dist_avail_and_initialized() is True
    assert get_rank() == rank
    assert get_world_size() == world_size
    assert is_main_process() == (rank == 0)

    # Test reduce_dict
    local_metrics = {"loss": 2.0 * (rank + 1), "steps": 1.0}
    reduced = reduce_dict(local_metrics, average=True)
    assert abs(reduced["loss"] - 3.0) < 1e-4

    # Test broadcast_scalar
    val = broadcast_scalar(42.0 if rank == 0 else 0.0, src=0)
    assert val == 42.0

    dataset = DummyDataset(size=16)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(dataset, batch_size=4, sampler=sampler)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    cfg = load_config("configs/test_smoke.yaml")
    cfg.project.output_dir = tmpdir
    cfg.data.num_workers = 0
    cfg.data.batch_size = 4
    cfg.training.epochs = 1
    cfg.training.data_parallel = True
    cfg.training.save_best = True
    cfg.training.save_latest = True

    trainer = Trainer(
        config=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    assert trainer.is_distributed is True
    assert isinstance(trainer.model, nn.parallel.DistributedDataParallel)
    assert hasattr(trainer.raw_model, "inc")

    # Run 1 epoch of training
    metrics = trainer.train_epoch(1)
    assert "total_loss" in metrics
    assert metrics["total_loss"] > 0

    # Save checkpoint
    trainer.ckpt_manager.save(
        epoch=1,
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        metrics={"val_iou": 0.5},
        config=trainer.config,
        is_best=True,
        hard_mining=trainer.hard_miner.state_dict(),
    )

    if rank == 0:
        best_path = os.path.join(tmpdir, "checkpoints", "checkpoint_best.pt")
        assert os.path.exists(best_path)
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        model_state = ckpt["model_state_dict"]
        for key in model_state.keys():
            assert not key.startswith("module."), f"Found module. prefix in key: {key}"

    cleanup_distributed()


def test_distributed_trainer_ddp_execution():
    """Verify Trainer with DistributedDataParallel (DDP) across multiple GPU processes."""
    num_gpus = torch.cuda.device_count()
    if not torch.cuda.is_available() or num_gpus < 2:
        pytest.skip("Test requires at least 2 CUDA GPUs")

    from sid_unet.utils.distributed import find_free_port

    with tempfile.TemporaryDirectory() as tmpdir:
        port = find_free_port()
        torch.multiprocessing.spawn(
            _ddp_trainer_worker,
            args=(num_gpus, port, tmpdir),
            nprocs=num_gpus,
            join=True,
        )


def _ddp_trainer_hard_mining_worker(rank, world_size, port, tmpdir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    from sid_unet.utils.distributed import cleanup_distributed
    from torch.utils.data.distributed import DistributedSampler

    dataset = DummyDataset(size=32)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(dataset, batch_size=4, sampler=sampler)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    cfg = load_config("configs/test_smoke.yaml")
    cfg.project.output_dir = tmpdir
    cfg.data.num_workers = 0
    cfg.data.batch_size = 4
    cfg.training.epochs = 2
    cfg.training.data_parallel = True
    cfg.training.use_hard_mining = True
    cfg.hard_mining = {"enabled": True, "metric": "median", "reset_epochs": 5}
    cfg.training.gradient_accumulation_steps = 2
    cfg.training.save_best = False
    cfg.training.save_latest = False

    trainer = Trainer(
        config=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    results = trainer.train()
    if rank == 0:
        assert len(results["history"]) == 2
        assert trainer.hard_miner.is_active_epoch is True

    cleanup_distributed()


def test_distributed_trainer_ddp_hard_mining_2epochs():
    """Verify Trainer with DDP and Hard Mining transitions across epochs 1 and 2 without desync or deadlock."""
    num_gpus = torch.cuda.device_count()
    if not torch.cuda.is_available() or num_gpus < 2:
        pytest.skip("Test requires at least 2 CUDA GPUs")

    from sid_unet.utils.distributed import find_free_port

    with tempfile.TemporaryDirectory() as tmpdir:
        port = find_free_port()
        torch.multiprocessing.spawn(
            _ddp_trainer_hard_mining_worker,
            args=(num_gpus, port, tmpdir),
            nprocs=num_gpus,
            join=True,
        )


