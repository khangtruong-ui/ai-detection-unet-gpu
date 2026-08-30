import os
import sys
import tempfile
from PIL import Image
import pytest
import torch

from sid_unet.train import main as train_main
from sid_unet.evaluate import main as eval_main
from sid_unet.predict import main as predict_main


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


def test_cli_train_eval_predict_pipeline(monkeypatch):
    monkeypatch.setattr("sid_unet.dataset.loader.hf_load_dataset", lambda *a, **kw: MockHFDataset(10))
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "outputs")
        pred_dir = os.path.join(tmpdir, "predictions")

        # 1. Test train CLI with synthetic / minimal override
        test_train_args = [
            "sid-train",
            "--config", "configs/default.yaml",
            "--override",
            f"project.output_dir={output_dir}",
            "project.device=cpu",
            "training.epochs=1",
            "data.num_workers=0",
            "data.train_samples_per_epoch=2",
            "data.val_samples=2",
            "data.batch_size=2",
            "model.features=[8, 16]",
            "data.image_size=[32, 32]",
            "logging.save_sample_images=false",
            "training.amp=false",
        ]
        monkeypatch.setattr(sys, "argv", test_train_args)
        train_main()

        best_ckpt = os.path.join(output_dir, "checkpoints", "checkpoint_best.pt")
        assert os.path.exists(best_ckpt)

        # 2. Test evaluate CLI
        test_eval_args = [
            "sid-eval",
            "--checkpoint", best_ckpt,
            "--override",
            "data.num_workers=0",
            "data.val_samples=2",
            "data.batch_size=2",
            "project.device=cpu",
            "--output_dir", os.path.join(output_dir, "eval_reports"),
        ]
        monkeypatch.setattr(sys, "argv", test_eval_args)
        eval_main()

        assert os.path.exists(os.path.join(output_dir, "eval_reports", "evaluation_report.md"))
        assert os.path.exists(os.path.join(output_dir, "eval_reports", "evaluation_report.json"))

        # 3. Test predict CLI on single image and folder
        sample_img_path = os.path.join(tmpdir, "sample.png")
        Image.new("RGB", (64, 64), color=(128, 64, 32)).save(sample_img_path)

        test_pred_args = [
            "sid-predict",
            "--checkpoint", best_ckpt,
            "--image", sample_img_path,
            "--output_dir", pred_dir,
            "--image_size", "32", "32",
            "--save_overlay",
            "--device", "cpu",
            "--override", "model.dropout=0.0",
        ]
        monkeypatch.setattr(sys, "argv", test_pred_args)
        predict_main()

        expected_mask = os.path.join(pred_dir, "sample_mask.png")
        expected_overlay = os.path.join(pred_dir, "sample_overlay.png")
        assert os.path.exists(expected_mask)
        assert os.path.exists(expected_overlay)


def test_cli_multi_config_training(monkeypatch):
    monkeypatch.setattr("sid_unet.dataset.loader.hf_load_dataset", lambda *a, **kw: MockHFDataset(10))
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "multi_run_outputs")

        test_train_args = [
            "sid-train",
            "--configs", "configs/test_smoke.yaml", "configs/test_quick.yaml",
            "--override",
            f"project.output_dir={output_dir}",
            "project.device=cpu",
            "training.epochs=1",
            "data.num_workers=0",
            "data.train_samples_per_epoch=2",
            "data.val_samples=2",
            "data.batch_size=2",
            "model.features=[8, 16]",
            "data.image_size=[32, 32]",
            "logging.save_sample_images=false",
            "training.amp=false",
        ]
        monkeypatch.setattr(sys, "argv", test_train_args)
        results = train_main()

        assert len(results) == 2
        # Check comparison report files
        comparison_md = os.path.join(output_dir, "multi_experiment_comparison.md")
        comparison_json = os.path.join(output_dir, "multi_experiment_comparison.json")
        assert os.path.exists(comparison_md)
        assert os.path.exists(comparison_json)

        # Check that individual experiment dirs exist
        exp1_dir = os.path.join(output_dir, "exp_01_test_smoke")
        exp2_dir = os.path.join(output_dir, "exp_02_test_quick")
        assert os.path.exists(exp1_dir)
        assert os.path.exists(exp2_dir)
        assert os.path.exists(os.path.join(exp1_dir, "reports", "training_final_report.md"))
        assert os.path.exists(os.path.join(exp2_dir, "reports", "training_final_report.md"))

