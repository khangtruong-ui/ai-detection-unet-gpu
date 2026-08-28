import os
import sys
import tempfile
from PIL import Image
import pytest
import torch

from sid_unet.train import main as train_main
from sid_unet.evaluate import main as eval_main
from sid_unet.predict import main as predict_main


def test_cli_train_eval_predict_pipeline(monkeypatch):
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
        ]
        monkeypatch.setattr(sys, "argv", test_pred_args)
        predict_main()

        expected_mask = os.path.join(pred_dir, "sample_mask.png")
        expected_overlay = os.path.join(pred_dir, "sample_overlay.png")
        assert os.path.exists(expected_mask)
        assert os.path.exists(expected_overlay)
