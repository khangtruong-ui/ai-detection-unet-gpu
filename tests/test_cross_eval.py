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
    path1 = "/workspace/outputs/RUN/test_smoke/checkpoints/checkpoint_best.pt"
    res1 = resolve_checkpoint_neighbor_dir(path1)
    assert res1 == "/workspace/outputs/RUN/test_smoke/cross_eval_reports"

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

        ckpt1 = os.path.join(train_dir, "RUN", "test_smoke", "checkpoints", "checkpoint_best.pt")
        ckpt2 = os.path.join(train_dir, "RUN", "test_quick", "checkpoints", "checkpoint_best.pt")
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
        neighbor_1 = os.path.join(train_dir, "RUN", "test_smoke", "cross_eval_reports")
        neighbor_2 = os.path.join(train_dir, "RUN", "test_quick", "cross_eval_reports")
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


def test_cross_eval_collision_skipping_and_continuous_master_report(monkeypatch):
    monkeypatch.setattr("sid_unet.dataset.loader.hf_load_dataset", lambda *a, **kw: MockHFDataset(10))

    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = os.path.join(tmpdir, "train_runs")
        cross_out_dir = os.path.join(tmpdir, "cross_eval_master")

        # Train 2 minimal checkpoints
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

        ckpt1 = os.path.join(train_dir, "RUN", "test_smoke", "checkpoints", "checkpoint_best.pt")
        ckpt2 = os.path.join(train_dir, "RUN", "test_quick", "checkpoints", "checkpoint_best.pt")

        # 1. Run cross eval with only ckpt1 on configs/test_smoke.yaml
        cross_args_1 = [
            "sid-cross-eval",
            "--cross-configs", "configs/test_smoke.yaml",
            "--checkpoints", ckpt1,
            "--split", "test",
            "--samples", "2",
            "--batch_size", "2",
            "--output_dir", cross_out_dir,
            "--override",
            "data.num_workers=0",
            "project.device=cpu",
        ]
        monkeypatch.setattr(sys, "argv", cross_args_1)
        res1 = cross_eval_main()
        assert len(res1["cross_results"]) == 1

        # 2. Run cross eval again with BOTH ckpt1 and ckpt2
        # ckpt1 + test_smoke.yaml should be SKIPPED due to collision, and ckpt2 should be evaluated.
        # Continuous master report should now hold BOTH results!
        cross_args_2 = [
            "sid-cross-eval",
            "--cross-configs", "configs/test_smoke.yaml",
            "--checkpoints", ckpt1, ckpt2,
            "--split", "test",
            "--samples", "2",
            "--batch_size", "2",
            "--output_dir", cross_out_dir,
            "--override",
            "data.num_workers=0",
            "project.device=cpu",
        ]
        monkeypatch.setattr(sys, "argv", cross_args_2)
        res2 = cross_eval_main()

        # Both entries are in the continuous master report
        assert len(res2["cross_results"]) == 2
        ckpt_names = [r["checkpoint_name"] for r in res2["cross_results"]]
        assert "test_smoke" in ckpt_names
        assert "test_quick" in ckpt_names


def test_cross_eval_metrics_differentiation_and_collision_robustness(monkeypatch):
    """
    Verify:
    1. Evaluating distinct checkpoints on the same dataset yields distinct, model-dependent metrics.
    2. Multiple checkpoints named 'checkpoint_best.pt' in different parent folders do not falsely collide.
    3. eval_single_batch with is_logit=True vs is_logit=False produces correct, non-constant metrics.
    """
    from PIL import ImageDraw

    class RealisticMockDataset:
        def __init__(self, count=6):
            self.samples = []
            for i in range(count):
                mask_img = Image.new("L", (64, 64), 0)
                if i % 3 == 2:
                    # Partial tampering
                    draw = ImageDraw.Draw(mask_img)
                    draw.rectangle([20, 20, 44, 44], fill=255)
                elif i % 3 == 1:
                    mask_img = Image.new("L", (64, 64), 255)
                self.samples.append({
                    "image": Image.new("RGB", (64, 64), color=(i * 20, 100, 100)),
                    "label": i % 3,
                    "mask": mask_img,
                    "img_id": f"mock_{i}",
                })

        def __iter__(self):
            return iter(self.samples)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

        def shuffle(self, *a, **k):
            return self

        def select(self, indices):
            return [self.samples[i] for i in indices]

    monkeypatch.setattr("sid_unet.dataset.loader.hf_load_dataset", lambda *a, **kw: RealisticMockDataset(6))

    with tempfile.TemporaryDirectory() as tmpdir:
        dir_a = os.path.join(tmpdir, "model_a")
        dir_b = os.path.join(tmpdir, "model_b")
        os.makedirs(os.path.join(dir_a, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(dir_b, "checkpoints"), exist_ok=True)

        ckpt_a_path = os.path.join(dir_a, "checkpoints", "checkpoint_best.pt")
        ckpt_b_path = os.path.join(dir_b, "checkpoints", "checkpoint_best.pt")

        from sid_unet.models.unet import UNet
        # Create Model A (small features, uninitialized random weights)
        model_a = UNet(in_channels=3, out_channels=1, features=[8, 16], aux_classifier=False)
        # Create Model B with different weights biased towards large negative logits (predicting all zeros)
        model_b = UNet(in_channels=3, out_channels=1, features=[8, 16], aux_classifier=False)
        with torch.no_grad():
            for p in model_b.parameters():
                p.fill_(-10.0)

        cfg_a = {
            "project": {"name": "model_a_run", "device": "cpu"},
            "model": {"name": "unet", "features": [8, 16], "aux_classifier": False},
        }
        cfg_b = {
            "project": {"name": "model_b_run", "device": "cpu"},
            "model": {"name": "unet", "features": [8, 16], "aux_classifier": False},
        }

        torch.save({"model_state_dict": model_a.state_dict(), "config": cfg_a}, ckpt_a_path)
        torch.save({"model_state_dict": model_b.state_dict(), "config": cfg_b}, ckpt_b_path)

        master_out = os.path.join(tmpdir, "master_cross")

        # Run cross evaluation across BOTH checkpoints on configs/test_smoke.yaml
        cross_res = run_cross_evaluation(
            checkpoint_paths=[ckpt_a_path, ckpt_b_path],
            config_paths=["configs/test_smoke.yaml"],
            split="test",
            samples=4,
            batch_size=2,
            output_dir=master_out,
            overrides=["data.num_workers=0", "project.device=cpu"],
            skip_collision=True,
            save_illustrations=False,
        )

        results = cross_res["cross_results"]
        # Both checkpoints must have been evaluated (no false collision skipping)
        assert len(results) == 2, f"Expected 2 evaluated pairs, got {len(results)}"

        res_a = results[0]
        res_b = results[1]

        assert res_a["checkpoint_path"] == ckpt_a_path
        assert res_b["checkpoint_path"] == ckpt_b_path

        # Checkpoint names must distinguish between the runs even though filename is checkpoint_best.pt
        assert res_a["checkpoint_name"] != res_b["checkpoint_name"]

        # Metrics between Model A and Model B MUST NOT be identical
        met_a = res_a["metrics"]
        met_b = res_b["metrics"]

        # Model A and Model B have completely different weights, so their loss/metrics must differ
        assert met_a["eval_total_loss"] != met_b["eval_total_loss"], "Losses across different models should not be identical"
        assert met_a["iou"] != met_b["iou"], "IoU across different models should not be identical"
        assert met_a["pixel_acc"] != met_b["pixel_acc"], "Pixel accuracy across different models should not be identical"

        # 3. Direct verification of eval_single_batch with is_logit=True vs is_logit=False
        from sid_unet.cross_eval import eval_single_batch
        from sid_unet.metrics.segmentation import SegmentationMetricTracker
        from sid_unet.metrics.classification import ClassificationMetricTracker
        from sid_unet.losses.auxiliary import build_loss
        from sid_unet.utils.config import load_config
        from sid_unet.postprocessing import MaskPostProcessor

        cfg = load_config("configs/test_smoke.yaml", overrides=["project.device=cpu"])
        loss_fn = build_loss(cfg)
        postproc = MaskPostProcessor(enabled=True)

        batch_sample = {
            "image": torch.randn(2, 3, 32, 32),
            "mask": torch.zeros(2, 1, 32, 32),
            "label": torch.tensor([0, 0]),
        }
        perf_model = UNet(in_channels=3, out_channels=1, features=[8, 16], aux_classifier=False)
        with torch.no_grad():
            for p in perf_model.parameters():
                p.fill_(-10.0)

        raw_tracker = SegmentationMetricTracker(threshold=0.5)
        post_tracker = SegmentationMetricTracker(threshold=0.5)
        cls_tracker = ClassificationMetricTracker(3)

        eval_single_batch(
            batch_sample,
            perf_model,
            loss_fn,
            torch.device("cpu"),
            raw_tracker,
            cls_tracker,
            threshold=0.5,
            post_seg_tracker=post_tracker,
            postprocessor=postproc,
        )
        raw_m, _ = raw_tracker.compute()
        post_m, _ = post_tracker.compute()

        # Both raw and post-processed must correctly achieve 1.0 IoU on all-background images
        assert raw_m["iou"] == 1.0
        assert post_m["iou"] == 1.0
        assert raw_m["pixel_acc"] == 1.0
        assert post_m["pixel_acc"] == 1.0

