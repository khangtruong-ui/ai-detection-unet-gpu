import os
import numpy as np
import pytest
import torch

from sid_unet.utils.plotting import (
    create_error_map,
    create_mask_overlay,
    plot_cross_eval_heatmaps,
    plot_eval_ablation_bar_chart,
    plot_eval_sample_predictions,
)


def test_error_map_and_overlay():
    gt = np.zeros((32, 32), dtype=np.float32)
    gt[5:15, 5:15] = 1.0

    pred = np.zeros((32, 32), dtype=np.float32)
    pred[10:20, 10:20] = 1.0

    err = create_error_map(pred, gt)
    assert err.shape == (32, 32, 3)
    assert err.dtype == np.uint8

    img = np.ones((32, 32, 3), dtype=np.uint8) * 128
    overlay = create_mask_overlay(img, pred)
    assert overlay.shape == (32, 32, 3)
    assert overlay.dtype == np.uint8


def test_plot_eval_sample_predictions(tmp_path):
    samples = []
    for i in range(3):
        samples.append({
            "image": torch.randn(3, 32, 32),
            "gt_mask": np.zeros((32, 32), dtype=np.float32),
            "unet_mask": np.zeros((32, 32), dtype=np.float32),
            "post_mask": np.zeros((32, 32), dtype=np.float32),
            "sam_mask": np.zeros((32, 32), dtype=np.float32),
            "final_mask": np.zeros((32, 32), dtype=np.float32),
            "label": i % 3,
            "img_id": f"test_img_{i}",
        })

    out_file = str(tmp_path / "sample_grid.png")
    res = plot_eval_sample_predictions(samples, out_file, max_samples=3)
    assert os.path.exists(res)
    assert os.path.getsize(res) > 0


def test_plot_eval_ablation_bar_chart(tmp_path):
    ablation_data = {
        "Baseline (Raw UNet)": {"iou": 0.72, "dice": 0.81, "pixel_acc": 0.91, "precision": 0.85, "recall": 0.78},
        "+ Post-Processing": {"iou": 0.76, "dice": 0.84, "pixel_acc": 0.93, "precision": 0.89, "recall": 0.80},
        "+ SAM Refinement": {"iou": 0.80, "dice": 0.88, "pixel_acc": 0.95, "precision": 0.91, "recall": 0.85},
        "+ SAM & Post-Processing": {"iou": 0.82, "dice": 0.90, "pixel_acc": 0.96, "precision": 0.92, "recall": 0.88},
    }

    out_file = str(tmp_path / "ablation_bar.png")
    res = plot_eval_ablation_bar_chart(ablation_data, out_file)
    assert os.path.exists(res)
    assert os.path.getsize(res) > 0


def test_plot_cross_eval_heatmaps(tmp_path):
    checkpoints = ["ckpt_model_a", "ckpt_model_b"]
    configs = ["dataset_1", "dataset_2", "dataset_3"]

    matrices = {
        "iou": {
            ("ckpt_model_a", "dataset_1"): 0.75,
            ("ckpt_model_a", "dataset_2"): 0.68,
            ("ckpt_model_a", "dataset_3"): 0.82,
            ("ckpt_model_b", "dataset_1"): 0.71,
            ("ckpt_model_b", "dataset_2"): 0.65,
            ("ckpt_model_b", "dataset_3"): 0.79,
        },
        "dice_f1": {
            ("ckpt_model_a", "dataset_1"): 0.85,
            ("ckpt_model_a", "dataset_2"): 0.78,
            ("ckpt_model_a", "dataset_3"): 0.90,
            ("ckpt_model_b", "dataset_1"): 0.82,
            ("ckpt_model_b", "dataset_2"): 0.76,
            ("ckpt_model_b", "dataset_3"): 0.88,
        },
    }

    out_dir = str(tmp_path / "illustrations")
    saved = plot_cross_eval_heatmaps(matrices, checkpoints, configs, out_dir)
    assert len(saved) == 2
    for p in saved:
        assert os.path.exists(p)
        assert os.path.getsize(p) > 0


def test_sid_illu_pipeline(tmp_path, monkeypatch):
    from sid_unet.illustration import run_illustration, expand_patterns
    from sid_unet.models.unet import build_model
    from sid_unet.utils.config import save_config, load_config

    # Create two dummy checkpoints (UNet and EfficientNet)
    cfg1 = {"model": {"name": "unet", "features": [8, 16], "aux_classifier": False, "in_channels": 3, "out_channels": 1}}
    m1 = build_model(cfg1)
    ckpt1 = tmp_path / "ckpt1.pt"
    torch.save({"model_state_dict": m1.state_dict(), "config": cfg1}, str(ckpt1))

    cfg2 = {"model": {"name": "efficientnet", "backbone": "efficientnet_b0", "pretrained": False, "sacrifice_of_pixel": True, "aux_classifier": False, "in_channels": 3, "out_channels": 1}}
    m2 = build_model(cfg2)
    ckpt2 = tmp_path / "ckpt2.pt"
    torch.save({"model_state_dict": m2.state_dict(), "config": cfg2}, str(ckpt2))

    # Create dummy dataset config
    smoke_cfg = load_config("configs/test_smoke.yaml").to_dict()
    dataset_cfg_path = tmp_path / "test_data.yaml"
    smoke_cfg["data"]["batch_size"] = 2
    save_config(smoke_cfg, str(dataset_cfg_path))

    # Mock create_eval_dataloader to return synthetic batches without network
    from torch.utils.data import DataLoader, TensorDataset
    images = torch.randn(4, 3, 64, 64)
    masks = torch.zeros(4, 1, 64, 64)
    masks[0, 0, 10:20, 10:20] = 1.0

    class DummyLoader:
        def __iter__(self):
            yield {
                "image": images[:2],
                "mask": masks[:2],
                "label": torch.tensor([0, 2]),
                "image_id": ["sample_0", "sample_1"],
            }
            yield {
                "image": images[2:],
                "mask": masks[2:],
                "label": torch.tensor([1, 0]),
                "image_id": ["sample_2", "sample_3"],
            }

    import sid_unet.illustration
    monkeypatch.setattr(sid_unet.illustration, "create_eval_dataloader", lambda config, split, samples_override: (DummyLoader(), "test"))

    illu_out = str(tmp_path / "illu_outputs")
    results = run_illustration(
        model_ckpts=[str(ckpt1), str(ckpt2)],
        dataset_configs=[str(dataset_cfg_path)],
        num_samples=2,
        output_dir=illu_out,
        threshold=0.5,
    )

    assert os.path.exists(results["report_path"])
    with open(results["report_path"], "r") as f:
        content = f.read()
        assert "SID-Illu" in content
        assert "ckpt1" in content
        assert "ckpt2" in content

    for cfg_k, d_res in results["dataset_results"].items():
        assert os.path.exists(d_res["figure_path"])
        assert os.path.getsize(d_res["figure_path"]) > 0
        assert len(d_res["samples"]) == 2

