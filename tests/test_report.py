import os
import tempfile
from sid_unet.utils.report import (
    extract_key_hyperparameters,
    format_config_table,
    format_metrics_table,
    generate_evaluation_report,
    generate_multi_experiment_report,
    generate_checkpoint_cross_eval_report,
    generate_master_cross_evaluation_report,
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
        # Ensure confusion matrix table header is NOT rendered when confusion_matrix is None or empty
        assert "Auxiliary Classification Confusion Matrix" not in res["markdown"]

        # Test with non-empty confusion matrix
        cm = [[5, 1, 0], [0, 6, 0], [1, 0, 4]]
        history = [
            {"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "val_iou": 0.7, "val_dice": 0.8, "val_pixel_acc": 0.95, "lr": 1e-3},
        ]
        curves_path = os.path.join(tmpdir, "training_curves.png")
        with open(curves_path, "w") as f:
            f.write("fake_img")

        res_with_cm = generate_evaluation_report(
            overall_metrics=overall,
            confusion_matrix=cm,
            config=cfg,
            output_dir=tmpdir,
            report_name="test_report_with_cm",
            history=history,
            curves_path=curves_path,
        )
        assert "Auxiliary Classification Confusion Matrix" in res_with_cm["markdown"]
        assert "Class 0 (Real)" in res_with_cm["markdown"]
        assert "Class 1 (Synthetic)" in res_with_cm["markdown"]
        assert "Training & Validation Curves" in res_with_cm["markdown"]
        assert "training_curves.png" in res_with_cm["markdown"]
        assert "Epoch-by-Epoch Training & Validation Progression" in res_with_cm["markdown"]



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


def test_generate_checkpoint_cross_eval_report():
    with tempfile.TemporaryDirectory() as tmpdir:
        results = [
            {
                "checkpoint_name": "model_exp1",
                "config_name": "config_imd2020",
                "dataset_name": "KhangTruong/IMD2020",
                "eval_split": "test",
                "total_evaluated_samples": 100,
                "metrics": {"eval_total_loss": 0.25, "iou": 0.82, "f1": 0.89, "auroc": 0.95, "pixel_acc": 0.98},
                "per_label_metrics": {
                    0: {"iou": 0.99, "dice": 0.99, "pixel_acc": 0.99, "samples": 50},
                    2: {"iou": 0.65, "dice": 0.79, "pixel_acc": 0.97, "samples": 50},
                },
                "confusion_matrix": [[50, 0], [5, 45]],
                "config_path": "configs/test_smoke.yaml",
            }
        ]

        res = generate_checkpoint_cross_eval_report(
            checkpoint_path="/path/to/checkpoints/checkpoint_best.pt",
            results=results,
            output_dir=tmpdir,
            report_name="cross_eval_report",
        )

        assert os.path.exists(os.path.join(tmpdir, "cross_eval_report.md"))
        assert os.path.exists(os.path.join(tmpdir, "cross_eval_report.json"))
        assert "config_imd2020" in res["markdown"]
        assert "0.8200" in res["summary_table"]


def test_generate_master_cross_evaluation_report():
    with tempfile.TemporaryDirectory() as tmpdir:
        cross_results = [
            {
                "checkpoint_path": "/checkpoints/ckpt1.pt",
                "checkpoint_name": "model_1",
                "config_path": "configs/conf1.yaml",
                "config_name": "conf1",
                "dataset_name": "Dataset_A",
                "eval_split": "test",
                "total_evaluated_samples": 50,
                "metrics": {"eval_total_loss": 0.30, "iou": 0.75, "f1": 0.83, "auroc": 0.91, "pixel_acc": 0.96},
            },
            {
                "checkpoint_path": "/checkpoints/ckpt1.pt",
                "checkpoint_name": "model_1",
                "config_path": "configs/conf2.yaml",
                "config_name": "conf2",
                "dataset_name": "Dataset_B",
                "eval_split": "test",
                "total_evaluated_samples": 50,
                "metrics": {"eval_total_loss": 0.20, "iou": 0.85, "f1": 0.91, "auroc": 0.96, "pixel_acc": 0.98},
            },
            {
                "checkpoint_path": "/checkpoints/ckpt2.pt",
                "checkpoint_name": "model_2",
                "config_path": "configs/conf1.yaml",
                "config_name": "conf1",
                "dataset_name": "Dataset_A",
                "eval_split": "test",
                "total_evaluated_samples": 50,
                "metrics": {"eval_total_loss": 0.28, "iou": 0.78, "f1": 0.86, "auroc": 0.93, "pixel_acc": 0.97},
            },
            {
                "checkpoint_path": "/checkpoints/ckpt2.pt",
                "checkpoint_name": "model_2",
                "config_path": "configs/conf2.yaml",
                "config_name": "conf2",
                "dataset_name": "Dataset_B",
                "eval_split": "test",
                "total_evaluated_samples": 50,
                "metrics": {"eval_total_loss": 0.18, "iou": 0.88, "f1": 0.93, "auroc": 0.97, "pixel_acc": 0.99},
            },
        ]

        res = generate_master_cross_evaluation_report(
            cross_results=cross_results,
            output_dir=tmpdir,
            report_name="master_cross_eval",
        )

        assert os.path.exists(os.path.join(tmpdir, "master_cross_eval.md"))
        assert os.path.exists(os.path.join(tmpdir, "master_cross_eval.json"))
        assert os.path.exists(os.path.join(tmpdir, "cross_eval_matrix.json"))
        assert "Mean IoU (Intersection over Union) Cross-Evaluation Matrix" in res["markdown"]
        assert "model_1" in res["summary_table"]
        assert "model_2" in res["summary_table"]
        assert "0.8800" in res["markdown"]


def test_generate_evaluation_report_with_ablation_and_illustrations():
    with tempfile.TemporaryDirectory() as tmpdir:
        overall = {"eval_total_loss": 0.25, "iou": 0.85, "dice": 0.91, "pixel_acc": 0.96}
        ablation = {
            "Baseline (Raw UNet)": {"iou": 0.78, "dice": 0.85, "pixel_acc": 0.94, "precision": 0.82, "recall": 0.88},
            "+ Post-Processing": {"iou": 0.82, "dice": 0.88, "pixel_acc": 0.95, "precision": 0.86, "recall": 0.90},
            "+ SAM Refinement": {"iou": 0.85, "dice": 0.91, "pixel_acc": 0.96, "precision": 0.89, "recall": 0.93},
        }
        ill_path = os.path.join(tmpdir, "illustrations", "sample.png")
        os.makedirs(os.path.dirname(ill_path), exist_ok=True)
        with open(ill_path, "w") as f:
            f.write("fake-png")

        res = generate_evaluation_report(
            overall_metrics=overall,
            output_dir=tmpdir,
            report_name="eval_test_report",
            ablation_results=ablation,
            illustration_paths=[ill_path],
        )

        md = res["markdown"]
        assert "Pipeline Ablation & Mask Refinement Comparison" in md
        assert "Baseline (Raw UNet)" in md
        assert "+ Post-Processing" in md
        assert "+ SAM Refinement" in md
        assert "0.7800" in md
        assert "Sample" in md
        assert "illustrations/sample.png" in md
        assert os.path.exists(os.path.join(tmpdir, "eval_test_report.md"))
        assert os.path.exists(os.path.join(tmpdir, "eval_test_report.json"))
        assert "ablation_comparison" in res["report_dict"]
        assert "illustration_paths" in res["report_dict"]

