import os
import tempfile
from sid_unet.utils.report import (
    extract_key_hyperparameters,
    format_config_table,
    format_metrics_table,
    generate_evaluation_report,
    generate_multi_experiment_report,
)


def test_extract_key_hyperparameters():
    cfg = {
        "project": {"name": "exp_test"},
        "data": {
            "dataset_name": "saberzl/SID_Set",
            "streaming": True,
            "image_size": [256, 256],
            "batch_size": 8,
            "train_samples_per_epoch": -1,
        },
        "model": {
            "name": "unet",
            "features": [64, 128],
            "bilinear": True,
            "aux_classifier": True,
            "num_classes": 3,
        },
        "loss": {
            "mask_loss_type": "combined",
            "bce_weight": 0.5,
            "dice_weight": 0.5,
            "aux_loss_type": "cross_entropy",
            "aux_weight": 0.2,
        },
        "training": {
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "scheduler": "cosine",
            "epochs": 5,
            "amp": True,
        },
    }

    hp = extract_key_hyperparameters(cfg)
    assert hp["Project Name"] == "exp_test"
    assert "Full Dataset" in hp["Epoch Limit / Steps"]
    assert "unet" in hp["Model"]
    assert "Combined" in hp["Mask Loss"]

    table = format_config_table(cfg)
    assert "### Configuration & Hyperparameters" in table
    assert "exp_test" in table


def test_generate_evaluation_report():
    with tempfile.TemporaryDirectory() as tmpdir:
        overall = {"iou": 0.85, "dice": 0.91, "pixel_acc": 0.98}
        per_label = {0: {"iou": 0.99, "dice": 0.99, "pixel_acc": 0.99, "samples": 10}}
        cfg = {"project": {"name": "test_run"}}

        res = generate_evaluation_report(
            overall_metrics=overall,
            per_label_metrics=per_label,
            config=cfg,
            output_dir=tmpdir,
            report_name="test_report",
        )

        assert os.path.exists(os.path.join(tmpdir, "test_report.md"))
        assert os.path.exists(os.path.join(tmpdir, "test_report.json"))
        assert "Overall Segmentation & Classification Metrics" in res["markdown"]


def test_generate_multi_experiment_report():
    with tempfile.TemporaryDirectory() as tmpdir:
        experiments = [
            {
                "run_name": "exp_baseline",
                "config_path": "configs/test_smoke.yaml",
                "best_epoch": 1,
                "best_score": 0.80,
                "config": {
                    "project": {"name": "exp_baseline"},
                    "training": {"learning_rate": 0.001},
                    "data": {"batch_size": 4},
                },
                "final_metrics": {"val_iou": 0.80, "val_dice": 0.88, "val_accuracy": 0.95},
            },
            {
                "run_name": "exp_dice",
                "config_path": "configs/test_quick.yaml",
                "best_epoch": 2,
                "best_score": 0.85,
                "config": {
                    "project": {"name": "exp_dice"},
                    "training": {"learning_rate": 0.0005},
                    "data": {"batch_size": 4},
                },
                "final_metrics": {"val_iou": 0.85, "val_dice": 0.91, "val_accuracy": 0.97},
            },
        ]

        res = generate_multi_experiment_report(
            experiment_results=experiments,
            output_dir=tmpdir,
            report_name="comparison_test",
        )

        assert os.path.exists(os.path.join(tmpdir, "comparison_test.md"))
        assert os.path.exists(os.path.join(tmpdir, "comparison_test.json"))
        assert "exp_baseline" in res["summary_table"]
        assert "exp_dice" in res["summary_table"]
