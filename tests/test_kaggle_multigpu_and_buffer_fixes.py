import os
import sys
import tempfile
import pytest
import torch
import torch.nn as nn

from sid_unet.models.iml_maskformer import SinusoidPositionEmbeddings2D
from sid_unet.models.unet import UNet
from sid_unet.utils.config import ConfigDict, save_config
from sid_unet.utils.distributed import is_kaggle_environment
from sid_unet.cross_eval import (
    evaluate_checkpoint_on_config,
    parse_args as cross_parse_args,
    main as cross_eval_main,
)


def test_sinusoid_position_embeddings_buffer_init_and_device():
    """Verify SinusoidPositionEmbeddings2D pre-registers buffers in __init__ and moves cleanly across devices."""
    embed_dim = 64
    spatial_shape = (16, 16)
    pe_mod = SinusoidPositionEmbeddings2D(embed_dim=embed_dim, spatial_shape=spatial_shape)

    # 1. Buffer must be registered upfront in __init__, not deferred to forward()
    buffers = dict(pe_mod.named_buffers())
    assert "position_embeddings" in buffers
    assert buffers["position_embeddings"].device == torch.device("cpu")

    # 2. Moving model to CUDA must move buffer before any forward pass
    if torch.cuda.is_available():
        pe_mod.to("cuda:0")
        buffers_cuda = dict(pe_mod.named_buffers())
        assert buffers_cuda["position_embeddings"].device == torch.device("cuda:0")

        # 3. Forward pass on CUDA should NOT register new buffers or pollute CPU buffers
        initial_buffer_count = len(list(pe_mod.named_buffers()))
        x = torch.randn(2, embed_dim, 16, 16, device="cuda:0")
        out = pe_mod(x)

        assert out.device == torch.device("cuda:0")
        assert out.shape == (2, embed_dim, 16, 16)
        # Buffer count must remain identical (no dynamic register_buffer during forward)
        assert len(list(pe_mod.named_buffers())) == initial_buffer_count
        # Buffer device must still be CUDA, not CPU
        assert pe_mod.position_embeddings.device == torch.device("cuda:0")

        # 4. Forward with differing spatial shape computes dynamically without modifying buffers
        x_diff = torch.randn(2, embed_dim, 8, 8, device="cuda:0")
        out_diff = pe_mod(x_diff)
        assert out_diff.device == torch.device("cuda:0")
        assert out_diff.shape == (2, embed_dim, 8, 8)
        assert len(list(pe_mod.named_buffers())) == initial_buffer_count


def test_is_kaggle_environment(monkeypatch):
    """Verify Kaggle environment detection with various indicators."""
    # When any Kaggle env var is set
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    assert is_kaggle_environment() is True

    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    monkeypatch.setenv("KAGGLE_CONTAINER_NAME", "kaggle_container_1")
    assert is_kaggle_environment() is True

    # When no Kaggle env vars or path exist
    monkeypatch.delenv("KAGGLE_CONTAINER_NAME", raising=False)
    monkeypatch.delenv("KAGGLE_URL_BASE", raising=False)
    for k in list(os.environ.keys()):
        if k.startswith("KAGGLE_"):
            monkeypatch.delenv(k, raising=False)

    # Monkeypatch os.path.exists to simulate non-Kaggle filesystem
    orig_exists = os.path.exists
    monkeypatch.setattr(os.path, "exists", lambda p: False if p == "/kaggle" else orig_exists(p))
    assert is_kaggle_environment() is False


def test_cross_eval_kaggle_multigpu_utilization():
    """Verify that cross-evaluation uses all available GPU devices on Kaggle and tracks utilization."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Test requires at least 2 CUDA devices.")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model_best.pt")
        cfg_path = os.path.join(tmpdir, "config.yaml")

        cfg = ConfigDict({
            "project": {"name": "test_multigpu_suite", "device": "auto"},
            "model": {"name": "unet", "in_channels": 3, "num_classes": 3, "features": [16, 32]},
            "data": {"dataset_name": "mock", "image_size": [64, 64], "batch_size": 8},
            "loss": {"mask_loss_type": "bce"},
        })
        save_config(cfg, cfg_path)

        model = UNet(in_channels=3, num_classes=3, features=[16, 32])
        torch.save({"model_state_dict": model.state_dict(), "config": cfg}, ckpt_path)

        res = evaluate_checkpoint_on_config(
            checkpoint_path=ckpt_path,
            config_path=cfg_path,
            split="test",
            samples=8,
            batch_size=8,
            save_illustrations=False,
        )

        assert "devices" in res
        # Both CUDA devices should be utilized
        assert len(res["devices"]) == torch.cuda.device_count()
        assert "cuda:0" in res["devices"]
        assert "cuda:1" in res["devices"]

        # GPU utilization stats must be populated
        assert "gpu_utilization" in res
        util = res["gpu_utilization"]
        assert "cuda:0" in util
        assert "cuda:1" in util
        assert util["cuda:0"]["peak_memory_mb"] > 0
        assert util["cuda:1"]["peak_memory_mb"] > 0

        # Overall metrics must reflect multi-device execution
        assert res["overall_metrics"]["num_devices"] == torch.cuda.device_count()
        assert res["overall_metrics"]["eval_total_loss"] > 0


def test_cross_eval_disable_data_parallel():
    """Verify that passing data_parallel=False forces single-device execution."""
    if not torch.cuda.is_available():
        pytest.skip("Test requires CUDA.")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model_single.pt")
        cfg_path = os.path.join(tmpdir, "config_single.yaml")

        cfg = ConfigDict({
            "project": {"name": "test_single_suite", "device": "auto"},
            "model": {"name": "unet", "in_channels": 3, "num_classes": 3, "features": [16, 32]},
            "data": {"dataset_name": "mock", "image_size": [64, 64], "batch_size": 4},
            "loss": {"mask_loss_type": "bce"},
        })
        save_config(cfg, cfg_path)

        model = UNet(in_channels=3, num_classes=3, features=[16, 32])
        torch.save({"model_state_dict": model.state_dict(), "config": cfg}, ckpt_path)

        res = evaluate_checkpoint_on_config(
            checkpoint_path=ckpt_path,
            config_path=cfg_path,
            split="test",
            samples=4,
            batch_size=4,
            save_illustrations=False,
            data_parallel=False,
        )

        assert res.get("devices") == ["cuda"] or len(res.get("devices", [])) == 1
        assert res["overall_metrics"].get("num_devices", 1) == 1


def test_cross_eval_cli_data_parallel_flags(monkeypatch):
    """Verify that CLI flags --data-parallel and --no-data-parallel parse and propagate correctly."""
    orig_argv = sys.argv
    try:
        sys.argv = [
            "sid-cross-eval",
            "--cross-configs", "configs/test_smoke.yaml",
            "--checkpoints", "checkpoint.pt",
            "--data-parallel",
            "--device-ids", "0", "1",
        ]
        args = cross_parse_args()
        assert args.data_parallel is True
        assert args.device_ids == [0, 1]

        sys.argv = [
            "sid-cross-eval",
            "--cross-configs", "configs/test_smoke.yaml",
            "--checkpoints", "checkpoint.pt",
            "--no-data-parallel",
        ]
        args_nodp = cross_parse_args()
        assert args_nodp.data_parallel is False
    finally:
        sys.argv = orig_argv


def test_cross_eval_dataparallel_small_batch_and_nontensor_kwargs():
    """Verify eval_single_batch handles DataParallel with small batches and non-tensor kwargs safely."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Test requires at least 2 CUDA devices.")

    from sid_unet.cross_eval import eval_single_batch
    from sid_unet.metrics.segmentation import SegmentationMetricTracker
    from sid_unet.metrics.classification import ClassificationMetricTracker
    from sid_unet.losses.auxiliary import build_loss
    from sid_unet.utils.config import load_config

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 1, kernel_size=1)
        def forward(self, x, f_r=None, img_ids=None):
            if x.size(0) == 0:
                raise RuntimeError("Batch size 0 should not reach forward!")
            return self.conv(x)

    base_model = SimpleModel().to("cuda:0")
    dp_model = nn.DataParallel(base_model, device_ids=[0, 1])

    cfg = load_config("configs/test_smoke.yaml", overrides=["project.device=cuda"])
    loss_fn = build_loss(cfg).to("cuda:0")

    # 1. Test batch size 1 (< 2 GPUs) with non-tensor img_ids
    batch_1 = {
        "image": torch.randn(1, 3, 32, 32),
        "mask": torch.zeros(1, 1, 32, 32),
        "label": torch.tensor([0]),
        "f_r": torch.randn(1, 16),
        "img_id": ["sample_0"],
    }
    raw_tracker = SegmentationMetricTracker(threshold=0.5)
    cls_tracker = ClassificationMetricTracker(3)

    loss_val, count, _ = eval_single_batch(
        batch_1,
        dp_model,
        loss_fn,
        torch.device("cuda:0"),
        raw_tracker,
        cls_tracker,
    )
    assert count == 1
    assert loss_val >= 0.0

    # 2. Test batch size 4 (>= 2 GPUs) with non-tensor img_ids
    batch_4 = {
        "image": torch.randn(4, 3, 32, 32),
        "mask": torch.zeros(4, 1, 32, 32),
        "label": torch.tensor([0, 1, 0, 1]),
        "f_r": torch.randn(4, 16),
        "img_id": ["s0", "s1", "s2", "s3"],
    }
    loss_val4, count4, _ = eval_single_batch(
        batch_4,
        dp_model,
        loss_fn,
        torch.device("cuda:0"),
        raw_tracker,
        cls_tracker,
    )
    assert count4 == 4
    assert loss_val4 >= 0.0


def test_unet_from_checkpoint_auto_detects_iml(tmp_path):
    """Verify UNet.from_checkpoint correctly infers 'iml' model architecture from state dict keys."""
    ckpt_path = str(tmp_path / "iml_checkpoint.pt")
    # Simulate an IML checkpoint without explicit config.model.name
    state_dict = {
        "pixel_module.input_projs.0.0.weight": torch.randn(256, 192, 1, 1),
        "transformer_decoder.layers.0.self_attn.out_proj.weight": torch.randn(256, 256),
    }
    torch.save({"model_state_dict": state_dict, "config": {}}, ckpt_path)

    # UNet.from_checkpoint with use_dummy for SAM2 so it initializes without network downloads
    model, cfg = UNet.from_checkpoint(
        ckpt_path,
        device="cpu",
        return_config=True,
        override_config={"model": {"use_dummy": True}},
        strict=False,
    )
    assert cfg.model.name == "iml"
    assert model.__class__.__name__ == "IMLMaskformer"


def test_sam2_concurrent_download_safety(tmp_path):
    """Verify SAM2ImageEncoder._download_checkpoint handles concurrent threads safely."""
    import concurrent.futures
    from sid_unet.models.iml_maskformer import SAM2ImageEncoder

    encoder = SAM2ImageEncoder(use_dummy=True)
    target = tmp_path / "checkpoints" / "sam2.1_hiera_test.pt"

    # Mock urllib.request.urlretrieve to simulate writing > 10MB file with brief delay
    import urllib.request
    dummy_data = b"0" * 10_000_100

    def mock_urlretrieve(url, filename):
        import time
        time.sleep(0.01)
        with open(filename, "wb") as f:
            f.write(dummy_data)

    orig_retrieve = urllib.request.urlretrieve
    urllib.request.urlretrieve = mock_urlretrieve

    try:
        def download_task(idx):
            return encoder._download_checkpoint(target, variant="tiny")

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(download_task, i) for i in range(6)]
            results = [f.result() for f in futures]

        assert len(results) == 6
        assert all(r == str(target) for r in results)
        assert target.exists()
        assert target.stat().st_size == len(dummy_data)
    finally:
        urllib.request.urlretrieve = orig_retrieve

