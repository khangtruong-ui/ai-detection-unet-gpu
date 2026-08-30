import os
import tempfile
import pytest
from sid_unet.utils.plotting import (
    plot_training_curves,
    plot_multi_experiment_curves,
    save_history_data,
)


def test_plot_training_curves_multi_epoch():
    history = [
        {
            "epoch": 1,
            "lr": 0.001,
            "train_loss": 0.65,
            "val_loss": 0.55,
            "train_mask_loss": 0.50,
            "val_mask_loss": 0.42,
            "val_iou": 0.50,
            "val_dice": 0.65,
            "val_pixel_acc": 0.88,
            "val_precision": 0.60,
            "val_recall": 0.70,
            "val_aux_accuracy": 0.75,
        },
        {
            "epoch": 2,
            "lr": 0.0008,
            "train_loss": 0.45,
            "val_loss": 0.38,
            "train_mask_loss": 0.35,
            "val_mask_loss": 0.30,
            "val_iou": 0.68,
            "val_dice": 0.80,
            "val_pixel_acc": 0.94,
            "val_precision": 0.78,
            "val_recall": 0.82,
            "val_aux_accuracy": 0.88,
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_png = os.path.join(tmpdir, "curves.png")
        saved = plot_training_curves(history, output_png, formats=["png", "pdf", "jpg"])

        assert len(saved) >= 3
        assert os.path.exists(os.path.join(tmpdir, "curves.png"))
        assert os.path.exists(os.path.join(tmpdir, "curves.pdf"))
        assert os.path.exists(os.path.join(tmpdir, "curves.jpg"))
        assert os.path.getsize(os.path.join(tmpdir, "curves.png")) > 1000


def test_plot_training_curves_single_epoch():
    history = [
        {
            "epoch": 1,
            "lr": 0.001,
            "train_loss": 0.5,
            "val_loss": 0.4,
            "val_iou": 0.7,
            "val_dice": 0.8,
            "val_pixel_acc": 0.95,
        }
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        output_png = os.path.join(tmpdir, "single_ep.png")
        saved = plot_training_curves(history, output_png)
        assert len(saved) == 1
        assert os.path.exists(output_png)


def test_plot_multi_experiment_curves():
    hist1 = [
        {"epoch": 1, "train_loss": 0.6, "val_loss": 0.5, "val_iou": 0.6, "val_dice": 0.7},
        {"epoch": 2, "train_loss": 0.4, "val_loss": 0.3, "val_iou": 0.8, "val_dice": 0.88},
    ]
    hist2 = [
        {"epoch": 1, "train_loss": 0.7, "val_loss": 0.6, "val_iou": 0.5, "val_dice": 0.6},
        {"epoch": 2, "train_loss": 0.5, "val_loss": 0.4, "val_iou": 0.75, "val_dice": 0.82},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_png = os.path.join(tmpdir, "multi_curves.png")
        res = plot_multi_experiment_curves(
            {"Run 1": hist1, "Run 2": hist2},
            output_path=output_png,
        )
        assert res == output_png
        assert os.path.exists(output_png)
        assert os.path.getsize(output_png) > 1000


def test_save_history_data():
    history = [
        {"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "val_iou": 0.7},
        {"epoch": 2, "train_loss": 0.3, "val_loss": 0.2, "val_iou": 0.85},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = save_history_data(history, output_dir=tmpdir, prefix="test_hist")
        assert os.path.exists(paths["json"])
        assert os.path.exists(paths["csv"])
