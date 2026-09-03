import os
import sys
import tempfile
import numpy as np
from PIL import Image
import pytest
import torch

from sid_unet.evaluate import evaluate_single_checkpoint, parse_args as eval_parse_args
from sid_unet.cross_eval import parse_args as cross_parse_args
from sid_unet.predict import parse_args as pred_parse_args
from sid_unet.models.unet import UNet
from sid_unet.utils.config import load_config, save_config


def test_cli_postprocessing_and_illustrations_args():
    orig_argv = sys.argv
    try:
        sys.argv = [
            "sid-eval",
            "--checkpoint", "ckpt.pt",
            "--post-process",
            "--min-area", "128",
            "--fill-holes",
            "--morphology", "close",
            "--illustrations",
            "--max-illustrations", "12",
        ]
        args = eval_parse_args()
        assert args.post_process is True
        assert args.min_area == 128
        assert args.fill_holes is True
        assert args.morphology == "close"
        assert args.save_illustrations is True
        assert args.max_illustrations == 12

        # Test negative flags
        sys.argv = [
            "sid-eval",
            "--checkpoint", "ckpt.pt",
            "--no-post-process",
            "--no-fill-holes",
            "--no-illustrations",
        ]
        args = eval_parse_args()
        assert args.post_process is False
        assert args.fill_holes is False
        assert args.save_illustrations is False

        # Test cross-eval
        sys.argv = [
            "sid-cross-eval",
            "--cross-configs", "conf.yaml",
            "--checkpoints", "ckpt.pt",
            "--post-process",
            "--min-area", "64",
        ]
        cross_args = cross_parse_args()
        assert cross_args.post_process is True
        assert cross_args.min_area == 64

        # Test predict
        sys.argv = [
            "sid-predict",
            "--checkpoint", "ckpt.pt",
            "--image", "test.png",
            "--no-post-process",
        ]
        pred_args = pred_parse_args()
        assert pred_args.post_process is False
    finally:
        sys.argv = orig_argv


def test_diffseg30k_config_validity():
    cfg_path = "configs/cross-eval/diffseg30k.yaml"
    assert os.path.exists(cfg_path)
    cfg = load_config(cfg_path)
    assert cfg.project.name == "eval_diffseg30k"
    assert cfg.data.dataset_name == "KhangTruong/Diffseg30k"
    assert cfg.post_processing.enabled is True
    assert cfg.post_processing.min_area == 64


class MockHFDataset:
    def __init__(self, count=4):
        self.samples = [
            {
                "image": Image.new("RGB", (32, 32), color=(i * 40, 80, 80)),
                "label": i % 3,
                "mask": Image.new("L", (32, 32), color=255 if i % 3 == 2 else 0),
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


class DummyRefiner:
    model_name = "facebook/sam3"
    def refine_single_sample(self, img, u_mask):
        return np.ones((32, 32), dtype=np.float32), {
            "pixel_change_ratio": 0.1,
            "pixels_changed": 100,
        }
    def refine_batch(self, imgs, logits):
        refined = torch.ones_like(logits)
        metrics = [{"pixel_change_ratio": 0.1, "pixels_changed": 100} for _ in range(logits.size(0))]
        return refined, metrics


def test_evaluate_ablation_and_illustrations(monkeypatch, tmp_path):
    monkeypatch.setattr("sid_unet.dataset.loader.hf_load_dataset", lambda *a, **kw: MockHFDataset(4))
    monkeypatch.setattr("sid_unet.evaluate.get_sam_refiner", lambda *a, **kw: DummyRefiner())

    # Build and save a minimal test checkpoint
    ckpt_dir = tmp_path / "RUN_TEST" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt_path = str(ckpt_dir / "checkpoint_best.pt")

    config = load_config(overrides=["model.features=[16, 32]", "model.aux_classifier=false", "training.epochs=1", "data.num_workers=0"])
    model = UNet(in_channels=3, out_channels=1, features=[16, 32], aux_classifier=False)
    torch.save({"model_state_dict": model.state_dict(), "config": config.to_dict()}, ckpt_path)

    out_eval_dir = str(tmp_path / "RUN_TEST" / "eval_reports")

    res = evaluate_single_checkpoint(
        checkpoint_path=ckpt_path,
        split="test",
        samples=4,
        overrides=["data.num_workers=0"],
        output_dir=out_eval_dir,
        segment="facebook/sam3",
        post_process=True,
        min_area=32,
        save_illustrations=True,
        max_illustrations=4,
    )

    assert "ablation_summary" in res["overall_metrics"]
    ablation = res["overall_metrics"]["ablation_summary"]
    assert "Baseline (Raw UNet)" in ablation
    assert "+ Post-Processing" in ablation
    assert "+ SAM Refinement" in ablation
    assert "+ SAM & Post-Processing" in ablation

    # Check report files and content
    report_md_path = os.path.join(out_eval_dir, "evaluation_report.md")
    report_json_path = os.path.join(out_eval_dir, "evaluation_report.json")
    assert os.path.exists(report_md_path)
    assert os.path.exists(report_json_path)

    with open(report_md_path, "r") as f:
        md_text = f.read()

    assert "Pipeline Ablation & Mask Refinement Comparison" in md_text
    assert "Baseline (Raw UNet)" in md_text
    assert "+ Post-Processing" in md_text
    assert "+ SAM Refinement" in md_text
    assert "+ SAM & Post-Processing" in md_text
    assert "Evaluation Illustrations" in md_text

    # Verify illustrations were generated
    ill_dir = os.path.join(out_eval_dir, "illustrations")
    assert os.path.exists(ill_dir)
    assert os.path.exists(os.path.join(ill_dir, "eval_sample_predictions.png"))
    assert os.path.exists(os.path.join(ill_dir, "eval_ablation_comparison.png"))


def test_cross_eval_ablation_and_illustrations(monkeypatch, tmp_path):
    monkeypatch.setattr("sid_unet.dataset.loader.hf_load_dataset", lambda *a, **kw: MockHFDataset(4))
    monkeypatch.setattr("sid_unet.cross_eval.get_sam_refiner", lambda *a, **kw: DummyRefiner())

    ckpt_dir = tmp_path / "RUN_CROSS" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt_path = str(ckpt_dir / "checkpoint_best.pt")

    config = load_config(overrides=["model.features=[16, 32]", "model.aux_classifier=false", "training.epochs=1", "data.num_workers=0"])
    model = UNet(in_channels=3, out_channels=1, features=[16, 32], aux_classifier=False)
    torch.save({"model_state_dict": model.state_dict(), "config": config.to_dict()}, ckpt_path)

    # Save a minimal cross-eval config
    cfg_path = str(tmp_path / "test_cross.yaml")
    save_config(config, cfg_path)

    master_out_dir = str(tmp_path / "master_cross_output")

    from sid_unet.cross_eval import run_cross_evaluation
    results = run_cross_evaluation(
        checkpoint_paths=[ckpt_path],
        config_paths=[cfg_path],
        split="test",
        samples=4,
        overrides=["data.num_workers=0"],
        output_dir=master_out_dir,
        segment="facebook/sam3",
        post_process=True,
        save_illustrations=True,
        max_illustrations=4,
    )

    assert "master_report" in results
    assert os.path.exists(os.path.join(master_out_dir, "master_cross_evaluation_report.md"))
    assert os.path.exists(os.path.join(master_out_dir, "master_cross_evaluation_report.json"))
    assert os.path.exists(os.path.join(master_out_dir, "cross_eval_matrix.json"))

    # Verify illustrations in master output dir
    ill_dir = os.path.join(master_out_dir, "illustrations")
    assert os.path.exists(ill_dir)
    assert os.path.exists(os.path.join(ill_dir, "cross_eval_iou_heatmap.png"))

    # Verify checkpoint neighbor report
    neighbor_dir = str(tmp_path / "RUN_CROSS" / "cross_eval_reports")
    assert os.path.exists(os.path.join(neighbor_dir, "cross_evaluation_report.md"))
