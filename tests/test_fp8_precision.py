"""Tests for FP8 / BF8 (float8_e4m3fn) precision defaults, training, inference, and CLI support."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from sid_unet.models.noise_rag import ForensicVectorDBRetrieval
from sid_unet.utils.config import DEFAULT_CONFIG, load_config
from sid_unet.utils.precision import (
    check_fp8_compatibility,
    is_bf16_supported,
    is_fp8_supported,
    normalize_precision_mode,
    prepare_model_for_fp8_inference,
    prepare_model_for_fp8_training,
    resolve_precision_spec,
)


def test_normalize_precision_mode_aliases():
    assert normalize_precision_mode("fp8") == "fp8"
    assert normalize_precision_mode("bf8") == "fp8"
    assert normalize_precision_mode("E4M3") == "fp8"
    assert normalize_precision_mode("float8_e4m3fn") == "fp8"
    assert normalize_precision_mode("auto") == "auto"
    assert normalize_precision_mode(None) == "auto"
    assert normalize_precision_mode(True) == "auto"
    assert normalize_precision_mode(False) == "fp32"
    assert normalize_precision_mode("bfloat16") == "bf16"
    assert normalize_precision_mode("16") == "fp16"


def test_default_configs_use_auto_fp8_when_supported():
    assert DEFAULT_CONFIG["training"]["mixed_precision"] == "auto"
    assert DEFAULT_CONFIG["training"]["fp8_all_parameters"] is True

    cfg = load_config("configs/experiments/noise_rag/noise_rag_prototype_v1.yaml")
    assert cfg.training.mixed_precision == "auto"
    assert cfg.training.fp8_all_parameters is True
    assert cfg.model.memory_dtype == "auto"

    spec = resolve_precision_spec(config=cfg, device=torch.device("cpu"))
    assert spec.mode == "fp32"
    assert not spec.fp8_enabled


def test_check_fp8_compatibility_fallbacks():
    # CPU target is never FP8
    ok_cpu, diag_cpu = check_fp8_compatibility(device=torch.device("cpu"))
    assert ok_cpu is False
    assert diag_cpu["device_type"] == "cpu"

    # Mock pre-Ada SM 7.5 GPU (Tesla T4) -> must fall back cleanly to bf16/fp16
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_name", return_value="Tesla T4"), \
         patch("torch.cuda.get_device_capability", return_value=(7, 5)), \
         patch("torch.cuda.is_bf16_supported", return_value=False):
        ok_t4, diag_t4 = check_fp8_compatibility(device=torch.device("cuda:0"))
        assert ok_t4 is False
        assert "SM 7.5" in str(diag_t4["reason"])

        spec = resolve_precision_spec(mode="auto", device=torch.device("cuda:0"))
        assert spec.fp8_enabled is False
        assert spec.mode == "fp16"
        assert spec.use_grad_scaler is True


@pytest.mark.skipif(not is_fp8_supported(), reason="Requires SM >= 8.9 CUDA GPU with FP8 E4M3 support")
def test_cuda_sm89_resolves_auto_and_bf8_to_fp8_e4m3fn():
    device = torch.device("cuda:0")
    ok, diag = check_fp8_compatibility(device=device)
    assert ok is True
    assert diag["compute_capability"] >= (8, 9)

    for req in (None, "auto", "fp8", "bf8", "e4m3"):
        spec = resolve_precision_spec(mode=req, device=device)
        assert spec.mode == "fp8"
        assert spec.fp8_enabled is True
        assert spec.fp8_dtype == torch.float8_e4m3fn
        assert spec.amp_enabled is True
        assert spec.amp_dtype == torch.bfloat16
        assert spec.use_grad_scaler is False
        assert spec.fp8_all_parameters is True

    spec_bf16 = resolve_precision_spec(mode="bf16", device=device)
    assert spec_bf16.mode == "bf16"
    assert spec_bf16.fp8_enabled is False
    assert spec_bf16.use_grad_scaler is False


class _MixedAttentionAndLinearBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_in = nn.Linear(64, 64)
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.head_odd = nn.Linear(64, 3)  # non-16-divisible out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.proj_in(x))
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        return self.head_odd(h + attn_out)


@pytest.mark.skipif(not is_fp8_supported(), reason="Requires SM >= 8.9 CUDA GPU with FP8 E4M3 support")
def test_prepare_model_for_fp8_training_and_inference_with_attention_and_odd_linear():
    device = torch.device("cuda:0")
    spec = resolve_precision_spec(mode="fp8", device=device)

    # 1. Training mode
    train_model = _MixedAttentionAndLinearBlock().to(device)
    train_stats = prepare_model_for_fp8_training(train_model, spec)
    assert train_stats["enabled"] is True
    assert train_stats["converted_linear_layers"] >= 2
    assert train_stats["skipped_linear_layers"] >= 1  # head_odd (64 -> 3)

    x = torch.randn(4, 32, 64, device=device)
    opt = torch.optim.AdamW(train_model.parameters(), lr=1e-3)
    opt.zero_grad()
    with torch.amp.autocast("cuda", dtype=spec.amp_dtype, enabled=spec.amp_enabled):
        out = train_model(x)
        loss = out.square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in train_model.parameters())
    opt.step()

    # 2. Inference mode (ensuring MultiheadAttention out_proj works under bfloat16 autocast)
    infer_model = _MixedAttentionAndLinearBlock().to(device).eval()
    infer_model, infer_stats = prepare_model_for_fp8_inference(infer_model, spec)
    assert infer_stats["enabled"] is True
    assert infer_stats["dynamic_fp8_layers"] >= 2
    assert infer_stats["weight_only_fp8_layers"] >= 1

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=spec.amp_dtype, enabled=spec.amp_enabled):
        y = infer_model(x)
    assert y.shape == (4, 32, 3)
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not is_fp8_supported(), reason="Requires SM >= 8.9 CUDA GPU with FP8 E4M3 support")
def test_forensic_vector_db_retrieval_fp8_e4m3_parity(tmp_path):
    device = torch.device("cuda:0")
    rng = torch.Generator().manual_seed(42)
    db_keys = torch.randn(2048, 64, generator=rng, dtype=torch.float16)
    db_labels = torch.randint(0, 2, (2048,), generator=rng, dtype=torch.uint8)
    db_path = tmp_path / "vec_db.pt"
    torch.save({"keys": db_keys, "labels": db_labels}, db_path)

    db_bf16 = ForensicVectorDBRetrieval(
        memory_path=str(db_path),
        memory_dtype="bfloat16",
        retrieval_k=4,
        embed_dim=64,
    ).to(device).eval()

    db_fp8 = ForensicVectorDBRetrieval(
        memory_path=str(db_path),
        memory_dtype="auto",
        retrieval_k=4,
        embed_dim=64,
    )
    # Copy projection weights so comparison isolates retrieval path
    db_fp8.query_proj.load_state_dict(db_bf16.query_proj.state_dict())
    db_fp8.fuse_proj.load_state_dict(db_bf16.fuse_proj.state_dict())
    db_fp8 = db_fp8.to(device).eval()
    assert db_fp8.fp8_retrieval_enabled is True

    queries = torch.randn(2, 64, 8, 8, device=device, dtype=torch.bfloat16)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out_bf16 = db_bf16(queries)
        out_fp8 = db_fp8(queries)

    assert out_fp8.shape == out_bf16.shape
    assert torch.isfinite(out_fp8).all()
    cos_sim = torch.nn.functional.cosine_similarity(
        out_bf16.float().flatten(), out_fp8.float().flatten(), dim=0
    )
    assert float(cos_sim) > 0.95


def test_cli_parsers_expose_precision_flag(monkeypatch):
    import sys
    from sid_unet.cross_eval import parse_args as parse_cross_eval_args
    from sid_unet.evaluate import parse_args as parse_eval_args
    from sid_unet.predict import parse_args as parse_predict_args
    from sid_unet.train import parse_args as parse_train_args

    monkeypatch.setattr(sys, "argv", ["sid-train", "--precision", "bf8"])
    assert parse_train_args().precision == "bf8"

    monkeypatch.setattr(sys, "argv", ["sid-eval", "--checkpoint", "ckpt.pt", "--mixed-precision", "e4m3"])
    assert parse_eval_args().precision == "e4m3"

    parsed_ce = parse_cross_eval_args(["--checkpoint", "ckpt.pt", "--config", "cfg.yaml", "--precision", "fp8"])
    assert parsed_ce.precision == "fp8"

    monkeypatch.setattr(sys, "argv", ["sid-predict", "--checkpoint", "ckpt.pt", "--precision", "bf16"])
    assert parse_predict_args().precision == "bf16"


@pytest.mark.skipif(not is_fp8_supported(), reason="Requires SM >= 8.9 CUDA GPU with FP8 E4M3 support")
def test_trainer_fp8_default_training_and_checkpoint_roundtrip(tmp_path):
    from torch.utils.data import DataLoader, Dataset
    from sid_unet.losses.auxiliary import SIDTotalLoss
    from sid_unet.training.trainer import Trainer

    class _SegDataset(Dataset):
        def __init__(self, size: int = 8):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx: int):
            lbl = idx % 3
            img = torch.randn(3, 64, 64)
            mask = torch.zeros(1, 64, 64) if lbl == 0 else (torch.ones(1, 64, 64) if lbl == 1 else torch.cat([torch.ones(1, 32, 64), torch.zeros(1, 32, 64)], dim=1))
            return {"image": img, "mask": mask, "label": torch.tensor(lbl, dtype=torch.long), "img_id": f"fp8_{idx}"}

    class _MiniFP8Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.proj = nn.Linear(64, 64)
            self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
            self.aux = nn.Linear(64, 3)

        def forward(self, x: torch.Tensor):
            h = torch.relu(self.conv(x))
            h_lin = self.proj(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            mask_logits = self.out_conv(h + h_lin)
            cls_logits = self.aux(h_lin.mean(dim=(-2, -1)))
            return {"mask_logits": mask_logits, "cls_logits": cls_logits}

    cfg = load_config(overrides=[
        f"project.output_dir={tmp_path}",
        "project.device=cuda",
        "training.epochs=1",
        "training.batch_size=4",
        "training.gradient_accumulation_steps=1",
        "data.image_size=[64, 64]",
        "logging.log_interval=1",
        "logging.save_sample_images=false",
    ])
    train_loader = DataLoader(_SegDataset(8), batch_size=4)
    val_loader = DataLoader(_SegDataset(4), batch_size=4)

    trainer = Trainer(
        config=cfg,
        model=_MiniFP8Net(),
        loss_fn=SIDTotalLoss(),
        train_loader=train_loader,
        val_loader=val_loader,
    )
    assert trainer.fp8_enabled is True
    assert trainer.precision_spec.mode == "fp8"
    assert trainer.precision_spec.fp8_dtype == torch.float8_e4m3fn
    assert trainer.scaler.is_enabled() is False
    assert trainer.fp8_info["converted_linear_layers"] >= 1

    res = trainer.train()
    assert len(res["history"]) == 1
    val_m = res["history"][0]["val_metrics"]
    assert "val_iou_ci95_low" in val_m
    assert "val_iou_ci95_high" in val_m
    import os
    ckpt_best = os.path.join(trainer.checkpoint_dir, "checkpoint_best.pt")
    assert os.path.exists(ckpt_best)
    reload_net = _MiniFP8Net().to("cuda")
    trainer.ckpt_manager.load_checkpoint(ckpt_best, reload_net)


def test_emulated_only_bf16_falls_back_to_fp16():
    """
    Pre-Ampere GPUs (e.g. Tesla T4 SM 7.5) return True for torch.cuda.is_bf16_supported()
    when including_emulation=True, but False when including_emulation=False.
    is_bf16_supported() and resolve_precision_spec() must reject emulated-only BF16
    and select native FP16.
    """
    def _fake_is_bf16(including_emulation: bool = True) -> bool:
        return bool(including_emulation)

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_name", return_value="Tesla T4"), \
         patch("torch.cuda.get_device_capability", return_value=(7, 5)), \
         patch("torch.cuda.is_bf16_supported", side_effect=_fake_is_bf16):
        assert is_bf16_supported(device=torch.device("cuda:0")) is False

        spec_auto = resolve_precision_spec(mode="auto", device=torch.device("cuda:0"))
        assert spec_auto.mode == "fp16"
        assert spec_auto.amp_dtype == torch.float16
        assert spec_auto.use_grad_scaler is True

        spec_bf16 = resolve_precision_spec(mode="bf16", device=torch.device("cuda:0"))
        assert spec_bf16.mode == "fp16"
        assert spec_bf16.amp_dtype == torch.float16
        assert spec_bf16.use_grad_scaler is True
        assert spec_bf16.fallback_reason is not None


@pytest.mark.skipif(not is_fp8_supported(), reason="Requires SM >= 8.9 CUDA GPU with FP8 E4M3 support")
def test_fp8_128x128_and_in1_linear_shapes_training_and_inference():
    """
    Regression test for TorchAO PerTensor vs 128x128 blockwise collision
    ('AssertionError: input_tensor must be 1x128 scaled') and PerRow WeightOnly
    failure on in_features == 1 ('RuntimeError: shape [out, 1] is invalid').
    """
    device = torch.device("cuda:0")
    spec = resolve_precision_spec(mode="fp8", device=device)

    class _EdgeShapeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin_128 = nn.Linear(128, 128)
            self.lin_wide = nn.Linear(128, 64)
            self.lin_in1 = nn.Linear(1, 64)
            self.lin_odd = nn.Linear(64, 3)

        def forward(self, x: torch.Tensor, scalar_x: torch.Tensor) -> torch.Tensor:
            h = torch.relu(self.lin_128(x))
            h = self.lin_wide(h) + torch.relu(self.lin_in1(scalar_x))
            return self.lin_odd(h)

    x = torch.randn(2, 32, 128, device=device)
    scalar_x = torch.randn(2, 32, 1, device=device)

    # 1. Training
    train_mod = _EdgeShapeBlock().to(device)
    train_stats = prepare_model_for_fp8_training(train_mod, spec)
    assert train_stats["converted_linear_layers"] == 2
    assert train_stats["skipped_linear_layers"] == 2
    with torch.amp.autocast("cuda", dtype=spec.amp_dtype, enabled=spec.amp_enabled):
        out_tr = train_mod(x, scalar_x)
        loss = out_tr.square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in train_mod.parameters())

    # 2. Inference (batch * seq_len = 64 > 1, where PerTensor 128x128 previously crashed)
    ref_mod = _EdgeShapeBlock().to(device).eval()
    infer_mod = _EdgeShapeBlock().to(device).eval()
    infer_mod.load_state_dict(ref_mod.state_dict())

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        y_bf16 = ref_mod(x, scalar_x)

    infer_mod, infer_stats = prepare_model_for_fp8_inference(infer_mod, spec)
    assert infer_stats["dynamic_fp8_layers"] == 2
    assert infer_stats["weight_only_fp8_layers"] == 2

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=spec.amp_dtype, enabled=spec.amp_enabled):
        y_fp8 = infer_mod(x, scalar_x)
    assert y_fp8.shape == (2, 32, 3)
    assert torch.isfinite(y_fp8).all()
    cos_sim = torch.nn.functional.cosine_similarity(
        y_bf16.float().flatten(), y_fp8.float().flatten(), dim=0
    )
    assert float(cos_sim) > 0.98


@pytest.mark.skipif(not is_fp8_supported(), reason="Requires SM >= 8.9 CUDA GPU with FP8 E4M3 support")
def test_fp8_inference_trufor_and_forensic_moe_v2_cuda():
    """
    Verify TruFor (contains 32x 128x128 stage-2 Linear layers) and ForensicMoEv2
    run cleanly under prepare_model_for_fp8_inference with batch_size=2 (M > 1).
    """
    from sid_unet.models.forensic_moe import ForensicMoEv2
    from sid_unet.models.trufor import TruFor

    device = torch.device("cuda:0")
    spec = resolve_precision_spec(mode="fp8", device=device)
    x = torch.rand(2, 3, 64, 64, device=device)

    tf = TruFor(np_out_channels=1, pretrained=False, allow_random_init=True).to(device).eval()
    tf, tf_stats = prepare_model_for_fp8_inference(tf, spec)
    assert tf_stats["dynamic_fp8_layers"] > 0
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=spec.amp_dtype, enabled=spec.amp_enabled):
        out_tf = tf(x, return_dict=True)
    assert out_tf["mask_logits"].shape == (2, 1, 64, 64)
    assert torch.isfinite(out_tf["mask_logits"]).all()

    moe = ForensicMoEv2(img_size=64, pretrained=False, allow_random_init=True).to(device).eval()
    moe, moe_stats = prepare_model_for_fp8_inference(moe, spec)
    assert moe_stats["dynamic_fp8_layers"] > 0
    assert moe_stats["weight_only_fp8_layers"] > 0
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=spec.amp_dtype, enabled=spec.amp_enabled):
        out_moe = moe(x, return_dict=True)
    assert out_moe["mask_logits"].shape == (2, 1, 64, 64)
    assert torch.isfinite(out_moe["mask_logits"]).all()


def test_fp8_inference_128x128_non_bf16_routes_to_weight_only():
    """
    When compute_dtype is float16 (where TorchAO PerRow dynamic FP8 is unsupported),
    128x128 Linear layers must not receive PerTensor dynamic FP8 and must route to
    Float8WeightOnlyConfig when fp8_all_parameters=True.
    """
    from sid_unet.utils.precision import PrecisionSpec

    pytest.importorskip("torchao.quantization")

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.l_128 = nn.Linear(128, 128)
            self.l_64 = nn.Linear(64, 64)
            self.l_in1 = nn.Linear(1, 32)

    spec_fp16 = PrecisionSpec(
        mode="fp8",
        requested_mode="fp8",
        fp8_enabled=True,
        fp8_dtype=torch.float8_e4m3fn,
        amp_enabled=True,
        amp_dtype=torch.float16,
        amp_dtype_str="fp8_e4m3fn",
        use_grad_scaler=True,
        fp8_all_parameters=True,
    )

    recorded = []

    def _fake_quantize_(root, config, filter_fn=None, **kwargs):
        matched = []
        for n, m in root.named_modules():
            if filter_fn is None or filter_fn(m, n):
                matched.append(n)
                if hasattr(m, "weight") and isinstance(m.weight, nn.Parameter):
                    m.weight = nn.Parameter(torch.zeros_like(m.weight.data, dtype=torch.int8), requires_grad=False)
        recorded.append((type(config).__name__, getattr(config, "granularity", None), matched))

    net = _Net()
    with patch("torchao.quantization.quantize_", side_effect=_fake_quantize_):
        _, stats = prepare_model_for_fp8_inference(net, spec_fp16)

    assert stats["dynamic_fp8_layers"] == 1  # only l_64
    assert stats["weight_only_fp8_layers"] == 2  # l_128 + l_in1
    dyn_entries = [e for e in recorded if e[0] == "Float8DynamicActivationFloat8WeightConfig"]
    wo_entries = [e for e in recorded if e[0] == "Float8WeightOnlyConfig"]
    assert len(dyn_entries) == 1
    assert dyn_entries[0][2] == ["l_64"]
    assert sorted(n for _, _, names in wo_entries for n in names) == ["l_128", "l_in1"]

