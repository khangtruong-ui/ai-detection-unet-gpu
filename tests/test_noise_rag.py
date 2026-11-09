"""
Unit and integration tests for Noise-RAG-Prototype-V1, NoisePrint++,
Pre-TVCA MaskCLIP feature extraction, and Vector Database caching.
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sid_unet.models.noise_rag import (
    NoisePrintPlusPlus,
    SomeConvVisionTransformerBlocks,
    ForensicVectorDBRetrieval,
    NoiseRAGPrototypeV1,
)
from sid_unet.models.forensic_memory import (
    ForensicVectorMemory,
    stop_gradient_query_replacement,
)
from sid_unet.cache.vector_db import MaskCLIPVectorDBBuilder
from sid_unet.losses.auxiliary import SIDTotalLoss
from sid_unet.utils.config import load_config
from sid_unet.models.unet import build_model


def test_noiseprint_strict_anti_fallback():
    """Verify NoisePrint++ rejects uninitialized weights and non-existent files."""
    # 1. Reject pretrained=False (random init strictly disabled)
    with pytest.raises(ValueError, match="strictly disabled"):
        NoisePrintPlusPlus(pretrained=False)

    # 2. Reject non-existent / corrupted file
    with pytest.raises(RuntimeError, match="Failed to download/load NoisePrint"):
        NoisePrintPlusPlus(pretrained=True, weights_path_or_url="/non/existent/path/weights.th")


def test_noiseprint_forward_and_frozen():
    """Verify NoisePrint++ extracts genuine features, outputs 3 channels, and stays frozen."""
    np_net = NoisePrintPlusPlus(pretrained=True, freeze=True)
    np_net.eval()

    # All parameters must have requires_grad=False
    for p in np_net.parameters():
        assert not p.requires_grad

    x = torch.rand(2, 3, 64, 64)
    out = np_net(x)

    assert out.shape == (2, 3, 64, 64)
    assert not out.requires_grad
    # Check output is non-trivial residual
    assert torch.isfinite(out).all()
    assert (out != 0).any()


def test_conv_vit_encoder_shapes():
    """Verify SomeConvVisionTransformerBlocks downsamples and refines tokens."""
    enc = SomeConvVisionTransformerBlocks(
        conv_backbone="mobilenetv3_small_100",
        pretrained=True,
        embed_dim=256,
        num_vit_blocks=2,
        out_stride=8,
    )
    enc.eval()

    x = torch.rand(2, 3, 128, 128)
    out = enc(x)

    # Stride 8 downsamples 128x128 -> 16x16
    assert out.shape == (2, 256, 16, 16)
    assert torch.isfinite(out).all()


def test_mask_clip_pre_tvca_extraction():
    """Verify MaskDecoder and MaskCLIP pre-TVCA feature tap (F_s) shape and properties."""
    from sid_unet.models.mask_clip import MaskDecoder

    decoder = MaskDecoder(in_channels=768, decoder_dim=256)
    x = torch.rand(2, 768, 16, 16)

    # Test extract_pre_tvca directly on decoder
    # Stage 0 ConvTranspose2d with scale 4.0 scales 16x16 -> 64x64
    f_s = decoder.extract_pre_tvca(x)
    assert f_s.shape == (2, 256, 64, 64)

    # Standard forward pass uses extract_pre_tvca internally before cross attention
    query = torch.rand(2, 256)
    mask_out = decoder(query, x)
    assert mask_out.shape == (2, 1, 64, 64)


def test_patch_label_computation_and_sampling():
    """Verify 4x4 avg pool continuous patch labels and 90% forged / 10% authentic sampling."""
    builder = MaskCLIPVectorDBBuilder(
        model=nn.Identity(),  # Placeholder for test
        target_entries=100,
        forged_ratio=0.9,
        authentic_ratio=0.1,
        dtype="float16",
        max_patches_per_image_forged=20,
        max_patches_per_image_authentic=5,
    )

    # Create dummy images and binary ground-truth masks
    # Batch size 4, 128x128 resolution -> 32x32 feature grid
    b_sz = 4
    h_f, w_f = 32, 32
    c_dim = 256

    # Mock extract_pre_tvca_features on builder.model
    builder.model.extract_pre_tvca_features = lambda imgs: torch.randn(imgs.shape[0], c_dim, h_f, w_f)

    # Half the mask is tampered (1.0), half is authentic (0.0)
    masks = torch.zeros(b_sz, 1, 128, 128)
    masks[:, :, :64, :] = 1.0  # Top half forged

    # Process batches until quotas are filled
    for _ in range(10):
        images = torch.rand(b_sz, 3, 128, 128)
        builder.process_batch(images, masks)
        if builder.is_complete:
            break

    assert builder.is_complete
    assert builder.num_forged >= 90
    assert builder.num_authentic >= 10

    consolidated = builder.consolidate()
    keys = consolidated["keys"]
    labels = consolidated["labels"]

    assert keys.shape == (100, 256)
    assert labels.shape == (100, 1)
    assert keys.dtype == torch.float16
    assert labels.dtype == torch.float16

    # Verify 90% forged, 10% authentic
    forged_count = int((labels[:, 0] > 0.0).sum().item())
    authentic_count = int((labels[:, 0] == 0.0).sum().item())
    assert forged_count == 90
    assert authentic_count == 10

    # Test saving and reloading
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        builder.save(tmp_path)
        assert os.path.exists(tmp_path)
        loaded = torch.load(tmp_path, map_location="cpu", weights_only=False)
        assert loaded["keys"].shape == (100, 256)
        assert loaded["labels"].shape == (100, 1)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_surrogate_stop_gradient_replacement():
    """Verify surrogate stop-gradient query replacement forward value and backward flow."""
    x_t = torch.randn(2, 256, 16, 16, requires_grad=True)
    x_r = torch.randn(2, 256, 16, 16, requires_grad=True)

    x_s = stop_gradient_query_replacement(original_query=x_t, retrieved_query=x_r)

    # 1. Forward pass must exactly equal x_r
    assert torch.allclose(x_s, x_r, atol=1e-6)

    # 2. Backward gradient flow: d(x_s)/d(x_t) = I, d(x_s)/d(x_r) = 0
    loss = (x_s ** 2).sum()
    loss.backward()

    # Gradient must reach x_t
    assert x_t.grad is not None
    assert x_t.grad.norm() > 0.0
    # Gradient to x_r must be None (not in autograd graph)
    assert x_r.grad is None


def test_alignment_loss_gradient_flow():
    """Verify alignment loss pulls XT toward StopGrad(XS) and computes MSE & Entropy."""
    model = NoiseRAGPrototypeV1(
        freeze_noiseprint=True,
        align_loss_type="mse",
        align_loss_weight=0.5,
    )

    x_t = torch.randn(2, 256, 16, 16, requires_grad=True)
    x_s = torch.randn(2, 256, 16, 16)

    # MSE Mode
    mse_loss = model.compute_alignment_loss(x_t=x_t, x_s=x_s)
    assert mse_loss.ndim == 0
    assert mse_loss.item() > 0.0
    mse_loss.backward()
    assert x_t.grad is not None
    assert x_t.grad.norm() > 0.0

    # Entropy / KL Mode
    x_t_ent = torch.randn(2, 256, 16, 16, requires_grad=True)
    model.align_loss_type = "entropy"
    ent_loss = model.compute_alignment_loss(x_t=x_t_ent, x_s=x_s)
    assert ent_loss.ndim == 0
    assert ent_loss.item() > 0.0
    ent_loss.backward()
    assert x_t_ent.grad is not None
    assert x_t_ent.grad.norm() > 0.0


def test_noise_rag_end_to_end_forward_and_loss():
    """Verify complete NoiseRAGPrototypeV1 forward pass and integration with SIDTotalLoss."""
    # Create temporary vector DB
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        db_path = tmp.name

    try:
        # Create small test DB
        keys = torch.randn(50, 256, dtype=torch.float16)
        labels = (torch.rand(50, 1) > 0.1).float().to(dtype=torch.float16)
        torch.save({"keys": keys, "values": keys, "labels": labels, "entry_count": 50}, db_path)

        model = NoiseRAGPrototypeV1(
            memory_path=db_path,
            memory_dtype="float16",
            embed_dim=256,
            num_cross_stages=3,
            retrieval_k=4,
            align_loss_type="mse+entropy",
            align_loss_weight=0.2,
            aux_classifier=True,
            num_classes=2,
        )
        model.train()

        x = torch.rand(2, 3, 128, 128)
        out = model(x, return_dict=True)

        assert isinstance(out, dict)
        assert "mask_logits" in out
        assert "multiscale_logits" in out
        assert "artifact_loss" in out
        assert "class_logits" in out

        # 4 scales: O_1 (base) + 3 stages = 4
        assert len(out["multiscale_logits"]) == 4
        assert out["mask_logits"].shape == (2, 1, 128, 128)
        for ml in out["multiscale_logits"]:
            assert ml.shape == (2, 1, 128, 128)

        assert out["class_logits"].shape == (2, 2)
        assert out["artifact_loss"].ndim == 0

        # Verify integration with SIDTotalLoss
        criterion = SIDTotalLoss(
            mask_loss_type="bce_dice",
            aux_classifier=True,
            aux_weight=0.2,
            multiscale_weight=0.4,
            artifact_weight=0.5,
        )

        gt_masks = torch.zeros(2, 1, 128, 128)
        gt_masks[:, :, :64, :] = 1.0
        gt_labels = torch.tensor([1, 0])

        total_loss, metrics = criterion(out, gt_masks, gt_labels)

        assert total_loss.ndim == 0
        assert total_loss.item() > 0.0
        assert "multiscale_loss" in metrics
        assert "artifact_loss" in metrics
        assert "aux_loss" in metrics

        # Test full backprop
        total_loss.backward()

        # Check gradients exist on trainable modules (conv_vit_x, heads)
        assert model.conv_vit_x.proj[0].weight.grad is not None
        assert model.conv_vit_x.proj[0].weight.grad.norm() > 0.0

        # NoisePrint++ should have no gradients (frozen)
        for p in model.noiseprint.parameters():
            assert p.grad is None

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_quantization_fp16_bf16():
    """Verify model forward pass under native mixed precision (torch.amp.autocast)."""
    model = NoiseRAGPrototypeV1(
        freeze_noiseprint=True,
        embed_dim=256,
        num_cross_stages=2,
    )
    model.eval()

    x = torch.rand(1, 3, 128, 128)

    # Test under autocast cpu / bfloat16
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model(x, return_dict=True)
        assert out["mask_logits"].shape == (1, 1, 128, 128)
        assert torch.isfinite(out["mask_logits"]).all()


def test_build_model_from_configs():
    """Verify build_model instantiates noise-rag-prototype-v1 and mask-clip-db-mae configs."""
    # 1. Noise-RAG config
    cfg_rag = load_config("configs/experiments/noise_rag/noise_rag_prototype_v1.yaml")
    model_rag = build_model(cfg_rag)
    assert isinstance(model_rag, NoiseRAGPrototypeV1)

    # 2. MaskCLIP DB config
    cfg_clip = load_config("configs/experiments/mask_clip/mask-clip-db-mae.yaml")
    assert cfg_clip.cache.mode == "vector_db"
    assert cfg_clip.cache.target_entries == 100000
    assert cfg_clip.cache.forged_ratio == 0.9
    assert cfg_clip.cache.authentic_ratio == 0.1
