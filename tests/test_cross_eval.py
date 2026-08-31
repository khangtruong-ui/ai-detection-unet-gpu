import os
import sys
import tempfile
from PIL import Image
import pytest
import torch

from sid_unet.train import main as train_main
from sid_unet.cross_eval import (
    main as cross_eval_main,
    expand_config_patterns,
    expand_checkpoint_patterns,
    resolve_checkpoint_neighbor_dir,
    evaluate_checkpoint_on_config,
    run_cross_evaluation,
)


class MockHFDataset:
    def __init__(self, count=10):
        self.samples = [
            {
                "image": Image.new("RGB", (64, 64), color=(i * 20, 100, 100)),
                "label": i % 3,
                "mask": Image.new("L", (64, 64), color=255 if i % 3 == 2 else 0),
                "img_id": f"mock_{i}",
            }
            for i in range(count)
        ]

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def shuffle(self, seed=None, buffer_size=None):
        return self

    def select(self, indices):
        return [self.samples[i] for i in indices]


def test_expand_patterns():
    # Test valid config globs
    configs = expand_config_patterns(["configs/test_*.yaml"])
    assert len(configs) >= 2
    assert all(c.endswith(".yaml") for c in configs)

    # Test missing path raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        expand_config_patterns(["non_existent_config.yaml"])

    with pytest.raises(FileNotFoundError):
        expand_checkpoint_patterns(["non_existent_ckpt.pt"])


def test_resolve_checkpoint_neighbor_dir():
    # When checkpoint is inside a 'checkpoints' subdirectory
    path1 = "/workspace/outputs/RUN001/checkpoints/checkpoint_best.pt"
    res1 = resolve_checkpoint_neighbor_dir(path1)
    assert res1 == "/workspace/outputs/RUN001/cross_eval_reports"

    # When checkpoint is in a root/flat directory
    path2 = "/workspace/models/checkpoint_best.pt"
    res2 = resolve_checkpoint_neighbor_dir(path2)
    assert res2 == "/workspace/models/cross_eval_reports"


def test_cross_eval_pipeline(monkeypatch):
    monkeypatch.setattr("sid_unet.dataset.loader.hf_load_dataset", lambda *a, **kw: MockHFDataset(10))

    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = os.path.join(tmpdir, "train_runs")
        cross_out_dir = os.path.join(tmpdir, "cross_eval_master")

        # 1. Train 2 models with different configs
        train_args = [
            "sid-train",
            "--configs", "configs/test_smoke.yaml", "configs/test_quick.yaml",
            "--output_dir", train_dir,
            "--override",
            "project.device=cpu",
            "training.epochs=1",
            "data.num_workers=0",
            "data.train_samples_per_epoch=2",
            "data.val_samples=2",
            "model.features=[8, 16]",
            "data.image_size=[32, 32]",
            "training.amp=false",
        ]
        monkeypatch.setattr(sys, "argv", train_args)
        train_main()

        ckpt1 = os.path.join(train_dir, "RUN001", "checkpoints", "checkpoint_best.pt")
        ckpt2 = os.path.join(train_dir, "RUN002", "checkpoints", "checkpoint_best.pt")
        assert os.path.exists(ckpt1)
        assert os.path.exists(ckpt2)

        # 2. Run Cross Evaluation CLI across 2 checkpoints and 2 configs (2x2 = 4 evaluations)
        cross_args = [
            "sid-cross-eval",
            "--cross-configs", "configs/test_smoke.yaml", "configs/test_quick.yaml",
            "--checkpoints", ckpt1, ckpt2,
            "--split", "test",
            "--samples", "2",
            "--batch_size", "2",
            "--output_dir", cross_out_dir,
            "--override",
            "data.num_workers=0",
            "project.device=cpu",
        ]
        monkeypatch.setattr(sys, "argv", cross_args)
        results = cross_eval_main()

        # Verify 4 total cross evaluations executed
        assert len(results["cross_results"]) == 4

        # Verify neighbor reports for each checkpoint
        neighbor_1 = os.path.join(train_dir, "RUN001", "cross_eval_reports")
        neighbor_2 = os.path.join(train_dir, "RUN002", "cross_eval_reports")
        assert os.path.exists(neighbor_1)
        assert os.path.exists(neighbor_2)
        assert os.path.exists(os.path.join(neighbor_1, "cross_evaluation_report.md"))
        assert os.path.exists(os.path.join(neighbor_1, "cross_evaluation_report.json"))
        assert os.path.exists(os.path.join(neighbor_2, "cross_evaluation_report.md"))
        assert os.path.exists(os.path.join(neighbor_2, "cross_evaluation_report.json"))

        # Verify master report and matrix outputs
        assert os.path.exists(os.path.join(cross_out_dir, "master_cross_evaluation_report.md"))
        assert os.path.exists(os.path.join(cross_out_dir, "master_cross_evaluation_report.json"))
        assert os.path.exists(os.path.join(cross_out_dir, "cross_eval_matrix.json"))
        assert os.path.exists(os.path.join(cross_out_dir, "cross_evaluation.log"))

        # Verify metrics reported
        for cr in results["cross_results"]:
            m = cr["metrics"]
            assert "iou" in m
            assert "dice" in m
            assert "f1" in m
            assert "auroc" in m
            assert "pixel_acc" in m
            assert "eval_total_loss" in m
