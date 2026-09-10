"""
Combined end-to-end integration tests running the entire mock loop lifecycle:
- Full process: Train model -> Save checkpoints (best, latest, periodic)
- Inspect checkpoint metadata and embedded config
- Load model from checkpoint (UNet, EfficientNet, simulated QLoRA)
- Resume training from checkpoint and continue for further epochs
- Evaluate model loaded from checkpoint (sid-eval / evaluate_single_checkpoint)
- Run inference prediction on images using checkpoint (sid-predict / predict_main)
- Generate multi-sample illustrations and error maps using checkpoint (sid-illu / run_illustration)
- Compare multiple distinct model checkpoints side-by-side in visualization grids
- Robust state_dict compatibility and strict/non-strict checkpoint loading
"""

from __future__ import annotations

import glob
import os
import sys
import tempfile
from typing import Any, Dict
import numpy as np
from PIL import Image
import pytest
import torch

from sid_unet.models.unet import UNet, build_model
from sid_unet.models.efficientnet import EfficientNetSegmentation
from sid_unet.train import train_single_run, main as train_main
from sid_unet.evaluate import evaluate_single_checkpoint, main as eval_main
from sid_unet.predict import main as predict_main
from sid_unet.illustration import run_illustration, main as illu_main
from sid_unet.training.callbacks import (
    CheckpointManager,
    inspect_checkpoint,
    find_auto_resume_checkpoint,
    load_state_dict_compatible,
)
from sid_unet.utils.config import load_config, save_config, ConfigDict


@pytest.fixture
def mock_dataset_config_path(tmp_path):
    """Create a temporary dataset config using the deterministic in-memory mock dataset."""
    cfg = load_config("configs/test_smoke.yaml").to_dict()
    cfg["project"]["device"] = "cpu"
    cfg["data"]["dataset_name"] = "mock"
    cfg["data"]["streaming"] = False
    cfg["data"]["batch_size"] = 2
    cfg["data"]["num_workers"] = 0
    cfg["data"]["train_samples_per_epoch"] = 4
    cfg["data"]["val_samples"] = 2
    cfg["data"]["test_samples"] = 2
    cfg["data"]["image_size"] = [32, 32]
    cfg["model"]["features"] = [8, 16]
    cfg["training"]["epochs"] = 2
    cfg["training"]["amp"] = False
    cfg["training"]["save_latest"] = True
    cfg["logging"]["save_sample_images"] = False

    cfg_file = tmp_path / "mock_dataset.yaml"
    save_config(cfg, str(cfg_file))
    return str(cfg_file)


def _find_checkpoint(base_dir: str, filename: str = "checkpoint_best.pt") -> str:
    """Helper to find checkpoint file in direct dir or RUN/<exp>/checkpoints subdir."""
    direct = os.path.join(base_dir, "checkpoints", filename)
    if os.path.exists(direct):
        return direct
    matches = glob.glob(os.path.join(base_dir, "**", "checkpoints", filename), recursive=True)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {filename} in {base_dir}")


def test_full_mock_loop_train_load_checkpoint_eval_predict_illu(tmp_path, mock_dataset_config_path, monkeypatch):
    """
    Complete end-to-end mock loop:
    1. Train a model with mock dataset -> produces checkpoints
    2. Inspect checkpoint structure and metadata
    3. Load from checkpoint with UNet.from_checkpoint
    4. Run evaluation on test split using the saved checkpoint
    5. Run prediction on a sample image using the saved checkpoint
    6. Run sid-illu visualization on the saved checkpoint
    """
    output_dir = str(tmp_path / "train_output")
    eval_dir = str(tmp_path / "eval_output")
    pred_dir = str(tmp_path / "pred_output")
    illu_dir = str(tmp_path / "illu_output")

    # -------------------------------------------------------------------------
    # Step 1: Train model on mock dataset (2 epochs)
    # -------------------------------------------------------------------------
    train_results = train_single_run(
        config_path=mock_dataset_config_path,
        overrides=[
            f"project.output_dir={output_dir}",
            "training.epochs=2",
            "training.save_latest=true",
        ],
        auto_resume=False,
    )
    assert train_results is not None
    assert len(train_results["history"]) == 2
    assert train_results["best_epoch"] in (1, 2)

    # Verify saved checkpoint files and config
    best_ckpt = _find_checkpoint(output_dir, "checkpoint_best.pt")
    latest_ckpt = _find_checkpoint(output_dir, "checkpoint_latest.pt")
    ckpt_dir = os.path.dirname(best_ckpt)
    best_cfg = os.path.join(ckpt_dir, "checkpoint_best_config.yaml")
    latest_cfg = os.path.join(ckpt_dir, "checkpoint_latest_config.yaml")

    assert os.path.exists(best_ckpt), "checkpoint_best.pt must exist"
    assert os.path.exists(latest_ckpt), "checkpoint_latest.pt must exist"
    assert os.path.exists(best_cfg), "checkpoint_best_config.yaml must exist"
    assert os.path.exists(latest_cfg), "checkpoint_latest_config.yaml must exist"

    # -------------------------------------------------------------------------
    # Step 2: Inspect checkpoint metadata
    # -------------------------------------------------------------------------
    meta = inspect_checkpoint(best_ckpt)
    assert meta["epoch"] in (1, 2)
    assert "best_score" in meta
    assert meta["has_optimizer"] is True
    assert meta["has_scheduler"] is True
    assert len(meta["history"]) >= 1

    latest_meta = inspect_checkpoint(latest_ckpt)
    assert latest_meta["epoch"] == 2
    assert len(latest_meta["history"]) == 2

    # -------------------------------------------------------------------------
    # Step 3: Load model from checkpoint using UNet.from_checkpoint
    # -------------------------------------------------------------------------
    loaded_model, loaded_cfg = UNet.from_checkpoint(
        best_ckpt,
        device="cpu",
        return_config=True,
    )
    assert isinstance(loaded_model, UNet)
    assert loaded_model.features == [8, 16]
    assert loaded_cfg.model.features == [8, 16]

    # Verify loaded model can perform forward pass and predict_mask
    dummy_input = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out_logits = loaded_model(dummy_input)
        assert out_logits.shape == (1, 1, 32, 32)
        pred_bin = loaded_model.predict_mask(dummy_input, threshold=0.5)
        assert pred_bin.shape == (1, 1, 32, 32)
        assert torch.all((pred_bin == 0.0) | (pred_bin == 1.0))

    # -------------------------------------------------------------------------
    # Step 4: Evaluate the loaded checkpoint using evaluate_single_checkpoint
    # -------------------------------------------------------------------------
    eval_res = evaluate_single_checkpoint(
        checkpoint_path=best_ckpt,
        config_path=mock_dataset_config_path,
        split="test",
        samples=4,
        output_dir=eval_dir,
        threshold=0.5,
        overrides=["project.device=cpu"],
    )
    assert "overall_metrics" in eval_res
    overall = eval_res["overall_metrics"]
    assert "iou" in overall
    assert "dice" in overall
    assert "pixel_acc" in overall or "pixel_accuracy" in overall
    assert os.path.exists(os.path.join(eval_dir, "evaluation_report.md"))
    assert os.path.exists(os.path.join(eval_dir, "evaluation_report.json"))

    # -------------------------------------------------------------------------
    # Step 5: Run prediction on a test image using predict_main CLI
    # -------------------------------------------------------------------------
    sample_img_path = str(tmp_path / "sample_test.png")
    test_img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    test_img.save(sample_img_path)

    pred_args = [
        "sid-predict",
        "--checkpoint", best_ckpt,
        "--image", sample_img_path,
        "--output_dir", pred_dir,
        "--image_size", "32", "32",
        "--device", "cpu",
        "--save_overlay",
    ]
    monkeypatch.setattr(sys, "argv", pred_args)
    predict_main()

    expected_mask = os.path.join(pred_dir, "sample_test_mask.png")
    expected_overlay = os.path.join(pred_dir, "sample_test_overlay.png")
    assert os.path.exists(expected_mask)
    assert os.path.exists(expected_overlay)

    # -------------------------------------------------------------------------
    # Step 6: Run sid-illu visualization on the checkpoint
    # -------------------------------------------------------------------------
    illu_res = run_illustration(
        model_ckpts=[best_ckpt],
        dataset_configs=[mock_dataset_config_path],
        num_samples=2,
        output_dir=illu_dir,
        threshold=0.5,
        device="cpu",
    )
    assert os.path.exists(illu_res["report_path"])
    assert len(illu_res["dataset_results"]) >= 1
    for ds_k, ds_data in illu_res["dataset_results"].items():
        assert os.path.exists(ds_data["figure_path"])
        assert os.path.getsize(ds_data["figure_path"]) > 0
        assert len(ds_data["samples"]) == 2


def test_checkpoint_resume_and_continued_training_lifecycle(tmp_path, mock_dataset_config_path):
    """
    Test checkpoint lifecycle across sequential training sessions:
    1. Train for 1 epoch and save checkpoint_latest.pt
    2. Resume from checkpoint_latest.pt targeting 3 epochs
    3. Verify epoch counter advances to 3 and history seamlessly combines
    4. Verify the updated checkpoint can be loaded and evaluated
    """
    output_dir = str(tmp_path / "resume_output")

    # Session 1: Train 1 epoch
    res1 = train_single_run(
        config_path=mock_dataset_config_path,
        overrides=[
            f"project.output_dir={output_dir}",
            "training.epochs=1",
            "training.save_latest=true",
        ],
        auto_resume=False,
    )
    assert len(res1["history"]) == 1
    latest_ckpt = _find_checkpoint(output_dir, "checkpoint_latest.pt")
    assert os.path.exists(latest_ckpt)

    # Inspect checkpoint at epoch 1
    meta1 = inspect_checkpoint(latest_ckpt)
    assert meta1["epoch"] == 1
    assert len(meta1["history"]) == 1

    # Session 2: Resume training targeting 3 epochs
    res2 = train_single_run(
        config_path=mock_dataset_config_path,
        overrides=[
            f"project.output_dir={output_dir}",
            "training.epochs=3",
            "training.save_latest=true",
        ],
        auto_resume=True,
    )
    assert len(res2["history"]) == 3
    assert res2["history"][0]["epoch"] == 1
    assert res2["history"][1]["epoch"] == 2
    assert res2["history"][2]["epoch"] == 3

    # Inspect resumed checkpoint
    meta2 = inspect_checkpoint(latest_ckpt)
    assert meta2["epoch"] == 3
    assert len(meta2["history"]) == 3

    # Load resumed checkpoint and verify evaluation
    model, cfg = UNet.from_checkpoint(latest_ckpt, device="cpu", return_config=True)
    assert isinstance(model, UNet)

    eval_dir = str(tmp_path / "resumed_eval")
    eval_res = evaluate_single_checkpoint(
        checkpoint_path=latest_ckpt,
        config_path=mock_dataset_config_path,
        split="test",
        samples=4,
        output_dir=eval_dir,
        overrides=["project.device=cpu"],
    )
    assert "overall_metrics" in eval_res
    assert "iou" in eval_res["overall_metrics"]


def test_multi_model_checkpoint_comparison_and_illustration(tmp_path, mock_dataset_config_path):
    """
    Test comparative visualization across multiple distinct checkpoints:
    1. Create Model A (UNet scratch) and save checkpoint
    2. Create Model B (EfficientNet with sacrifice_of_pixel) and save checkpoint
    3. Run run_illustration with BOTH checkpoints simultaneously
    4. Assert comparison grid columns include both models side-by-side
    """
    # 1. Model A: UNet
    cfg_a = {
        "model": {
            "name": "unet",
            "features": [8, 16],
            "aux_classifier": False,
            "in_channels": 3,
            "out_channels": 1,
        },
        "project": {"device": "cpu"},
    }
    model_a = build_model(cfg_a)
    ckpt_a = tmp_path / "model_a_unet.pt"
    torch.save({"model_state_dict": model_a.state_dict(), "config": cfg_a, "epoch": 2}, str(ckpt_a))

    # 2. Model B: EfficientNet
    cfg_b = {
        "model": {
            "name": "efficientnet",
            "backbone": "efficientnet_b0",
            "pretrained": False,
            "sacrifice_of_pixel": True,
            "aux_classifier": False,
            "in_channels": 3,
            "out_channels": 1,
        },
        "project": {"device": "cpu"},
    }
    model_b = build_model(cfg_b)
    ckpt_b = tmp_path / "model_b_effnet.pt"
    torch.save({"model_state_dict": model_b.state_dict(), "config": cfg_b, "epoch": 2}, str(ckpt_b))

    # Verify both can load via UNet.from_checkpoint
    loaded_a = UNet.from_checkpoint(str(ckpt_a), device="cpu")
    loaded_b = UNet.from_checkpoint(str(ckpt_b), device="cpu")
    assert isinstance(loaded_a, UNet)
    assert isinstance(loaded_b, EfficientNetSegmentation)

    # 3. Run multi-model sid-illu comparison
    illu_out = str(tmp_path / "multi_illu_out")
    res = run_illustration(
        model_ckpts=[str(ckpt_a), str(ckpt_b)],
        dataset_configs=[mock_dataset_config_path],
        num_samples=2,
        output_dir=illu_out,
        device="cpu",
    )

    assert os.path.exists(res["report_path"])
    with open(res["report_path"], "r") as f:
        report_text = f.read()
        assert "model_a_unet" in report_text
        assert "model_b_effnet" in report_text

    for ds_k, ds_res in res["dataset_results"].items():
        assert os.path.exists(ds_res["figure_path"])
        # Verify both models have prediction entries for each sample
        for s in ds_res["samples"]:
            assert "model_a_unet" in s["models"]
            assert "model_b_effnet" in s["models"]
            assert "pred_mask" in s["models"]["model_a_unet"]
            assert "pred_mask" in s["models"]["model_b_effnet"]


def test_checkpoint_loading_compatibility_with_metadata_and_quantization_keys(tmp_path):
    """
    Test checkpoint loading robustness against unexpected keys, quantization attributes,
    and strict parameter modes:
    1. Checkpoint with unexpected quantization metadata keys (.absmax, .quant_map, etc.)
    2. Loading with strict=None (default) -> succeeds gracefully
    3. Loading with strict=False -> succeeds cleanly
    4. Loading with strict=True -> raises RuntimeError for strict verification
    """
    # Create a small UNet
    cfg = {"model": {"name": "unet", "features": [8, 16], "aux_classifier": False, "in_channels": 3, "out_channels": 1}}
    model = build_model(cfg)
    sd = model.state_dict()

    # Inject extra unexpected keys (mimicking QLoRA quantization metadata and extra heads)
    augmented_sd = dict(sd)
    augmented_sd["inc.conv.0.weight.absmax"] = torch.tensor([1.0, 2.0])
    augmented_sd["inc.conv.0.weight.quant_map"] = torch.tensor([0.1, 0.2])
    augmented_sd["inc.conv.0.weight.quant_state.bitsandbytes__nf4"] = torch.zeros(4)
    augmented_sd["unknown_unused_layer.weight"] = torch.randn(4, 4)

    ckpt_path = str(tmp_path / "augmented_ckpt.pt")
    torch.save({"model_state_dict": augmented_sd, "config": cfg, "epoch": 1}, ckpt_path)

    # 1. Default strict=None should successfully load without crashing
    loaded_default, cfg_ret = UNet.from_checkpoint(ckpt_path, device="cpu", return_config=True)
    assert isinstance(loaded_default, UNet)
    assert loaded_default.features == [8, 16]

    # 2. Explicit strict=False should succeed cleanly
    loaded_non_strict = UNet.from_checkpoint(ckpt_path, device="cpu", strict=False)
    assert isinstance(loaded_non_strict, UNet)

    # 3. Explicit strict=True should raise RuntimeError due to unexpected keys
    with pytest.raises(RuntimeError) as exc_info:
        UNet.from_checkpoint(ckpt_path, device="cpu", strict=True)
    assert "Unexpected key(s) in state_dict" in str(exc_info.value)


def test_cli_mock_loop_combined_e2e(tmp_path, monkeypatch, mock_dataset_config_path):
    """
    End-to-end CLI workflow test:
    Executes sid-train -> sid-eval -> sid-predict -> sid-illu sequentially
    via the CLI entrypoints on the mock dataset.
    """
    run_output = str(tmp_path / "cli_run")
    pred_dir = str(tmp_path / "cli_pred")
    illu_dir = str(tmp_path / "cli_illu")

    # 1. sid-train CLI
    train_args = [
        "sid-train",
        "--config", mock_dataset_config_path,
        "--output_dir", run_output,
        "--override",
        "training.epochs=1",
        "training.save_latest=true",
        "project.device=cpu",
    ]
    monkeypatch.setattr(sys, "argv", train_args)
    train_res = train_main()
    assert train_res is not None

    # Resolve checkpoint using helper
    ckpt_path = _find_checkpoint(run_output, "checkpoint_best.pt")
    assert os.path.exists(ckpt_path)

    # 2. sid-eval CLI
    eval_args = [
        "sid-eval",
        "--checkpoint", ckpt_path,
        "--config", mock_dataset_config_path,
        "--split", "test",
        "--samples", "2",
        "--output_dir", os.path.join(run_output, "eval_reports"),
        "--override", "project.device=cpu",
    ]
    monkeypatch.setattr(sys, "argv", eval_args)
    eval_main()
    assert os.path.exists(os.path.join(run_output, "eval_reports", "evaluation_report.md"))

    # 3. sid-predict CLI
    sample_img = str(tmp_path / "cli_sample.png")
    Image.new("RGB", (32, 32), color=(80, 120, 160)).save(sample_img)
    pred_args = [
        "sid-predict",
        "--checkpoint", ckpt_path,
        "--image", sample_img,
        "--output_dir", pred_dir,
        "--device", "cpu",
    ]
    monkeypatch.setattr(sys, "argv", pred_args)
    predict_main()
    assert os.path.exists(os.path.join(pred_dir, "cli_sample_mask.png"))

    # 4. sid-illu CLI
    illu_args = [
        "sid-illu",
        "--model-ckpts", ckpt_path,
        "--dataset-configs", mock_dataset_config_path,
        "--num-samples", "2",
        "--output-dir", illu_dir,
        "--device", "cpu",
    ]
    monkeypatch.setattr(sys, "argv", illu_args)
    illu_res = illu_main()
    assert os.path.exists(illu_res["report_path"])
