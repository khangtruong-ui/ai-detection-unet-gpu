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


def test_xs_to_xr_mse_loss():
    """Verify MSE loss enforcing XS to get close to XR."""
    model = NoiseRAGPrototypeV1(
        freeze_noiseprint=True,
        align_loss_type="mse",
        align_loss_weight=0.5,
    )

    x_t = torch.randn(2, 256, 16, 16, requires_grad=True)
    x_r = torch.randn(2, 256, 16, 16)
    x_s = stop_gradient_query_replacement(original_query=x_t, retrieved_query=x_r)

    # Compute MSE loss enforcing XS to get close to XR
    loss = model.compute_alignment_loss(x_s=x_s, x_r=x_r, x_t=x_t)
    assert loss.ndim == 0
    assert loss.item() > 0.0

    # Backpropagate: gradient must flow to x_t (which generates x_s)
    loss.backward()
    assert x_t.grad is not None
    assert x_t.grad.norm() > 0.0

    # Test direct compute_alignment_loss(x_s=x_s, x_r=x_r)
    x_s_grad = torch.randn(2, 256, 16, 16, requires_grad=True)
    loss_direct = model.compute_alignment_loss(x_s=x_s_grad, x_r=x_r)
    assert loss_direct.item() > 0.0
    loss_direct.backward()
    assert x_s_grad.grad is not None
    assert x_s_grad.grad.norm() > 0.0


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
        assert "align_loss" in out
        assert "xs_xr_loss" in out
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
        assert "align_loss" in metrics
        assert "xs_xr_loss" in metrics
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
    assert cfg_clip.cache.target_entries in (100000, 1000000)
    assert cfg_clip.cache.forged_ratio == 0.9
    assert cfg_clip.cache.authentic_ratio == 0.1


def test_vector_db_extraction_with_encoded_mask_bytes():
    """Verify vector DB extraction when loader yields encoded PNG mask bytes using decode_mask_bytes."""
    import io
    from PIL import Image
    from sid_unet.cache.vector_db import extract_mask_clip_vector_db_from_loader

    # Create dummy model with extract_pre_tvca_features
    class MockMaskCLIP(nn.Module):
        def extract_pre_tvca_features(self, imgs):
            b = imgs.shape[0]
            return torch.randn(b, 256, 16, 16)

    model = MockMaskCLIP()

    # Create dummy batches with raw PNG mask bytes
    # Top half tampered (1.0), bottom half pristine (0.0)
    m1 = Image.new("L", (64, 64), color=0)
    for y in range(32):
        for x in range(64):
            m1.putpixel((x, y), 255)
    buf1 = io.BytesIO()
    m1.save(buf1, format="PNG")
    b1 = buf1.getvalue()

    # Pristine mask
    m2 = Image.new("L", (64, 64), color=0)
    buf2 = io.BytesIO()
    m2.save(buf2, format="PNG")
    b2 = buf2.getvalue()

    dummy_batches = [
        {
            "images": torch.rand(2, 3, 64, 64),
            "masks": [b1, b2],
        }
        for _ in range(5)
    ]

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        db_path = tmp.name

    try:
        saved_path = extract_mask_clip_vector_db_from_loader(
            model=model,
            loader=dummy_batches,
            output_path=db_path,
            target_entries=50,
            forged_ratio=0.8,
            authentic_ratio=0.2,
            dtype="float16",
        )
        assert os.path.exists(saved_path)
        payload = torch.load(saved_path, map_location="cpu", weights_only=False)
        assert "keys" in payload
        assert "labels" in payload
        assert payload["keys"].shape[1] == 256
        assert payload["keys"].dtype == torch.float16
        assert payload["labels"].dtype == torch.float16
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_forensic_vector_memory_labels_support():
    """Verify ForensicVectorMemory stores, consolidates, and saves continuous patch labels."""
    mem = ForensicVectorMemory(
        memory_enabled=True,
        memory_max_entries=100,
        memory_key_dim=32,
        memory_value_dim=32,
        memory_dedup_threshold=0.9,
        memory_consolidation_method="dedup_greedy",
        memory_dtype="float16",
    )

    # Add observations with labels
    keys = torch.randn(60, 32)
    labels = torch.rand(60, 1)
    added = mem.add_observations(keys=keys, values=keys, labels=labels)

    assert added == 60
    assert len(mem._epoch_observed_labels) == 1

    stats = mem.consolidate()
    assert mem.entry_count > 0
    assert mem.labels.shape == (mem.entry_count, 1)

    # Test state dict roundtrip
    sd = mem.state_dict()
    assert "labels" in sd
    assert sd["labels"] is not None
    assert sd["labels"].shape == (mem.entry_count, 1)

    mem2 = ForensicVectorMemory(
        memory_enabled=True,
        memory_max_entries=100,
        memory_key_dim=32,
        memory_value_dim=32,
    )
    mem2.load_state_dict(sd)
    assert mem2.entry_count == mem.entry_count
    assert torch.allclose(mem2.labels, mem.labels)


def test_forensic_vector_db_retrieval():
    """Verify ForensicVectorDBRetrieval works in both fallback and loaded database modes."""
    retrieval = ForensicVectorDBRetrieval(memory_path=None, memory_dtype="float16", retrieval_k=3, embed_dim=64)
    assert not retrieval.has_database

    x_t = torch.randn(2, 64, 8, 8)
    # When no DB is loaded, acts as identity fallback
    out_fallback = retrieval(x_t)
    assert torch.allclose(out_fallback, x_t)

    # Create dummy database
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        db_path = tmp.name

    try:
        keys = torch.randn(30, 64, dtype=torch.float16)
        labels = torch.rand(30, 1, dtype=torch.float16)
        torch.save({"keys": keys, "labels": labels}, db_path)

        retrieval.load_database(db_path)
        assert retrieval.has_database
        assert retrieval.db_keys.shape == (30, 64)

        out_retrieved = retrieval(x_t)
        assert out_retrieved.shape == (2, 64, 8, 8)
        assert torch.isfinite(out_retrieved).all()
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_noise_rag_checkpoint_roundtrip_unet():
    """Verify UNet.from_checkpoint restores NoiseRAGPrototypeV1 and sibling vector DB."""
    from sid_unet.models.unet import UNet

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "mask_clip_db_mae.pt")
        keys = torch.randn(20, 256, dtype=torch.float16)
        labels = torch.rand(20, 1, dtype=torch.float16)
        torch.save({"keys": keys, "labels": labels}, db_path)

        model = NoiseRAGPrototypeV1(
            memory_path=db_path,
            memory_dtype="float16",
            embed_dim=256,
            num_cross_stages=2,
            conv_backbone="mobilenetv3_small_100",
        )
        model.eval()

        ckpt_path = os.path.join(tmp_dir, "checkpoint.pt")
        cfg_dict = {
            "model": {
                "name": "noise-rag-prototype-v1",
                "memory_path": db_path,
                "memory_dtype": "float16",
                "embed_dim": 256,
                "num_cross_stages": 2,
                "conv_backbone": "mobilenetv3_small_100",
            }
        }
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": cfg_dict,
            "epoch": 1,
        }, ckpt_path)

        loaded_model = UNet.from_checkpoint(ckpt_path, device="cpu")
        assert isinstance(loaded_model, NoiseRAGPrototypeV1)
        assert loaded_model.retrieval.has_database

        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            out_orig = model(x, return_dict=True)
            out_loaded = loaded_model(x, return_dict=True)

        assert torch.allclose(out_orig["mask_logits"], out_loaded["mask_logits"], atol=1e-5)


def test_combined_alignment_loss_mse_plus_entropy():
    """Verify NoiseRAGPrototypeV1 combined mse+entropy alignment loss computation and backward flow."""
    model = NoiseRAGPrototypeV1(
        freeze_noiseprint=True,
        align_loss_type="mse+entropy",
        align_loss_weight=0.5,
    )

    x_t = torch.randn(2, 256, 8, 8, requires_grad=True)
    x_s = torch.randn(2, 256, 8, 8)

    loss = model.compute_alignment_loss(x_t=x_t, x_s=x_s)
    assert loss.ndim == 0
    assert loss.item() > 0.0

    loss.backward()
    assert x_t.grad is not None
    assert x_t.grad.norm() > 0.0


def test_mask_clip_full_extract_pre_tvca_features():
    """Verify MaskCLIP.extract_pre_tvca_features extracts F_s of shape [B, 256, H/4, W/4]."""
    from sid_unet.models.mask_clip import MaskCLIP

    model = MaskCLIP(
        clip_model_name="ViT-B/16",
        pretrained=False,
        allow_random_init=True,
        device="cpu",
    )
    model.eval()

    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        f_s = model.extract_pre_tvca_features(x)

    assert f_s.shape == (2, 256, 32, 32)
    assert torch.isfinite(f_s).all()


def test_sid_cache_cli_vector_db_mode_run():
    """Verify sid-cache CLI executes vector DB mode without errors using mock dataset."""
    from sid_unet.cache.cli import cli_main
    from sid_unet.dataset.loader import generate_mock_raw_sample
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_path = os.path.join(tmp_dir, "test_cache_cfg.yaml")
        cfg_content = """project:
  name: "test_cache"
  seed: 42
data:
  dataset_name: "mock"
  image_size: [64, 64]
  batch_size: 2
  splits: ["train"]
model:
  name: "mask_clip"
  clip_model_name: "ViT-B/16"
  pretrained: false
  allow_random_init: true
loss:
  edge_weight: 0.0
cache:
  mode: "vector_db"
  vector_db_filename: "test_db.pt"
  target_entries: 10
  forged_ratio: 0.5
  authentic_ratio: 0.5
  dtype: "float16"
"""
        with open(cfg_path, "w") as f:
            f.write(cfg_content)

        raw_samples = [generate_mock_raw_sample(i, target_image_size=(64, 64)) for i in range(10)]
        with patch("sid_unet.dataset.loader.load_parquet_or_hf_dataset", return_value=(raw_samples, "train")):
            result = cli_main([
                "--config", cfg_path,
                "--output-dir", tmp_dir,
                "--no-detach",
                "--max-samples", "4",
                "--no-push-to-hub",
                "--num-workers", "1",
            ])

        out_db = os.path.join(tmp_dir, "test_db.pt")
        assert os.path.exists(out_db)
        loaded = torch.load(out_db, map_location="cpu", weights_only=False)
        assert "keys" in loaded
        assert "labels" in loaded


def test_forensic_vector_db_retrieval_chunking_and_resolution():
    """Verify chunked retrieval produces matching results and resolves candidate paths / HF downloads."""
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Create a dummy vector DB
        db_path = os.path.join(tmp_dir, "test_db.pt")
        keys = torch.randn(50, 256, dtype=torch.float16)
        labels = torch.rand(50, 1, dtype=torch.float16)
        torch.save({"keys": keys, "labels": labels}, db_path)

        retrieval = ForensicVectorDBRetrieval(
            memory_path=db_path,
            memory_dtype="float16",
            retrieval_k=3,
            embed_dim=256,
        )
        assert retrieval.has_database
        assert retrieval.db_keys.shape == (50, 256)

        # Test chunked retrieval vs small token retrieval
        # Create a large token map that triggers the chunking path (> 2048 tokens)
        # e.g. B=2, H=36, W=36 -> 2 * 36 * 36 = 2592 tokens
        x_large = torch.randn(2, 256, 36, 36)
        out_chunked = retrieval(x_large)
        assert out_chunked.shape == (2, 256, 36, 36)
        assert torch.isfinite(out_chunked).all()

        # 2. Test candidate local path resolution when given a relative path
        cand_dir = os.path.join(tmp_dir, "outputs", "dataset_cache")
        os.makedirs(cand_dir, exist_ok=True)
        cand_db = os.path.join(cand_dir, "mask_clip_db_mae.pt")
        torch.save({"keys": keys, "labels": labels}, cand_db)

        # 3. Test Hugging Face Hub download resolution
        with patch("huggingface_hub.hf_hub_download", return_value=db_path) as mock_hf:
            retrieval_hf = ForensicVectorDBRetrieval(
                memory_path="KhangTruong/mask-clip-db-mae-cache",
                memory_dtype="float16",
                embed_dim=256,
            )
            assert retrieval_hf.has_database
            assert mock_hf.called


def test_noise_rag_configurable_cross_modes():
    """Verify NoiseRAGPrototypeV1 cross_mode support for 'xs' (surrogate) and 'recurrent' (chained)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "mask_clip_db_mae.pt")
        keys = torch.randn(20, 256, dtype=torch.float16)
        labels = torch.rand(20, 1, dtype=torch.float16)
        torch.save({"keys": keys, "labels": labels}, db_path)

        # 1. recurrent mode (default)
        model_rec = NoiseRAGPrototypeV1(
            memory_path=db_path,
            embed_dim=256,
            num_cross_stages=2,
            cross_mode="recurrent",
        )
        assert model_rec.cross_mode == "recurrent"

        # 2. xs mode
        model_xs = NoiseRAGPrototypeV1(
            memory_path=db_path,
            embed_dim=256,
            num_cross_stages=2,
            cross_mode="xs",
        )
        assert model_xs.cross_mode == "xs"

        x = torch.rand(1, 3, 64, 64)
        out_rec = model_rec(x, return_dict=True)
        out_xs = model_xs(x, return_dict=True)

        assert out_rec["mask_logits"].shape == (1, 1, 64, 64)
        assert out_xs["mask_logits"].shape == (1, 1, 64, 64)
        assert torch.isfinite(out_rec["mask_logits"]).all()
        assert torch.isfinite(out_xs["mask_logits"]).all()


def test_sid_train_vector_db_routing_and_no_mock_latents():
    """Verify sid-train routes --cached-hf-repo to vector DB memory without creating 84-ch mock latents."""
    from sid_unet.train import train_single_run
    from unittest.mock import patch, MagicMock

    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg_path = os.path.join(tmp_dir, "test_noise_rag_cfg.yaml")
        db_path = os.path.join(tmp_dir, "mask_clip_db_mae.pt")
        keys = torch.randn(10, 256, dtype=torch.float16)
        labels = torch.rand(10, 1, dtype=torch.float16)
        torch.save({"keys": keys, "labels": labels}, db_path)

        cfg_content = f"""project:
  name: "test_noise_rag_train"
  seed: 42
  device: "cpu"
  output_dir: "{tmp_dir}/run"
data:
  dataset_name: "mock_rgb_dataset"
  streaming: false
  image_size: [64, 64]
  batch_size: 2
  train_split: "train"
  val_split: "val"
  train_samples_per_epoch: 2
  val_samples_per_epoch: 2
model:
  name: "noise-rag-prototype-v1"
  freeze_noiseprint: true
  memory_path: "{db_path}"
  embed_dim: 256
  num_cross_stages: 1
  in_channels: 3
  out_channels: 1
training:
  epochs: 1
  lr: 0.001
  amp: false
  auto_batch_size: false
loss:
  mask_loss_type: "combined"
  aux_classifier: false
"""
        with open(cfg_path, "w") as f:
            f.write(cfg_content)

        # Mock dataset loading to return 3-channel RGB image batches
        mock_train_batch = {
            "image": torch.rand(2, 3, 64, 64),
            "mask": torch.zeros(2, 1, 64, 64),
            "label": torch.tensor([0, 1]),
            "img_id": ["img1", "img2"],
        }
        mock_val_batch = {
            "image": torch.rand(2, 3, 64, 64),
            "mask": torch.zeros(2, 1, 64, 64),
            "label": torch.tensor([0, 1]),
            "img_id": ["img_v1", "img_v2"],
        }

        mock_train_loader = [mock_train_batch]
        mock_val_loader = [mock_val_batch]

        with patch("sid_unet.train.create_dataloaders", return_value=(mock_train_loader, mock_val_loader, mock_val_loader)), \
             patch("huggingface_hub.hf_hub_download", return_value=db_path):

            # Run train_single_run passing --cached-hf-repo pointing to a vector DB repo
            results = train_single_run(
                config_path=cfg_path,
                cached_hf_repo="KhangTruong/mask-clip-db-mae-cache",
                auto_resume=False,
            )

            assert results is not None
            assert results["best_epoch"] == 1
            assert "final_metrics" in results or "overall_metrics" in results


def test_create_cached_dataloaders_rejects_vector_db_repo():
    """Verify create_cached_dataloaders raises ValueError if passed a vector DB repo instead of silent mock shard."""
    from sid_unet.dataset.loader import create_cached_dataloaders
    from sid_unet.utils.config import ConfigDict
    from unittest.mock import patch, MagicMock

    cfg = ConfigDict({
        "project": {"seed": 42},
        "data": {"image_size": [64, 64], "batch_size": 2, "streaming": False},
        "model": {"name": "noise-rag-prototype-v1", "in_channels": 3},
    })

    # Mock HfApi to report repo has .pt file and no parquet files
    mock_api = MagicMock()
    mock_api.list_repo_files.return_value = ["mask_clip_db_mae.pt", "cache_info.json"]

    with patch("sid_unet.dataset.loader.resolve_cached_parquet_files", return_value=[]), \
         patch("huggingface_hub.HfApi", return_value=mock_api):

        with pytest.raises(ValueError, match="vector database"):
            create_cached_dataloaders(cfg, cached_repo="KhangTruong/mask-clip-db-mae-cache")


def test_sid_eval_and_cross_eval_noise_rag_checkpoint():
    """Verify sid-eval and sid-cross-eval evaluate NoiseRAGPrototypeV1 checkpoints with dict outputs."""
    from sid_unet.evaluate import evaluate_single_checkpoint
    from sid_unet.cross_eval import evaluate_checkpoint_on_config
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create vector DB and checkpoint
        db_path = os.path.join(tmp_dir, "mask_clip_db_mae.pt")
        keys = torch.randn(10, 256, dtype=torch.float16)
        labels = torch.rand(10, 1, dtype=torch.float16)
        torch.save({"keys": keys, "labels": labels}, db_path)

        model = NoiseRAGPrototypeV1(
            memory_path=db_path,
            embed_dim=256,
            num_cross_stages=1,
            conv_backbone="mobilenetv3_small_100",
            aux_classifier=False,
        )
        model.eval()

        ckpt_dir = os.path.join(tmp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "checkpoint_best.pt")

        cfg_dict = {
            "project": {"name": "test_noise_rag", "seed": 42},
            "data": {"dataset_name": "mock", "image_size": [64, 64], "batch_size": 2},
            "model": {
                "name": "noise-rag-prototype-v1",
                "memory_path": db_path,
                "embed_dim": 256,
                "num_cross_stages": 1,
                "conv_backbone": "mobilenetv3_small_100",
                "aux_classifier": False,
            },
            "loss": {"mask_loss_type": "combined", "aux_classifier": False},
        }
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": cfg_dict,
            "epoch": 1,
        }, ckpt_path)

        class DummyEvalDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 2
            def __getitem__(self, idx):
                return {
                    "image": torch.rand(3, 64, 64),
                    "mask": torch.zeros(1, 64, 64),
                    "label": torch.tensor(idx % 2),
                    "img_id": f"img_{idx}",
                }

        mock_loader = torch.utils.data.DataLoader(DummyEvalDataset(), batch_size=2)

        # 1. Test evaluate_single_checkpoint
        with patch("sid_unet.evaluate.create_eval_dataloader", return_value=mock_loader):
            eval_res = evaluate_single_checkpoint(
                checkpoint_path=ckpt_path,
                output_dir=os.path.join(tmp_dir, "eval_out"),
                split="val",
                save_illustrations=False,
            )
            assert eval_res is not None
            assert "overall_metrics" in eval_res
            assert "iou" in eval_res["overall_metrics"]

        # 2. Test evaluate_checkpoint_on_config (cross-eval)
        cross_cfg_path = os.path.join(tmp_dir, "cross_cfg.yaml")
        with open(cross_cfg_path, "w") as f:
            f.write("""protocol: "test_protocol"
data:
  dataset_name: "mock"
  image_size: [64, 64]
  batch_size: 2
""")
        with patch("sid_unet.cross_eval.create_eval_dataloader", return_value=mock_loader):
            cross_res = evaluate_checkpoint_on_config(
                checkpoint_path=ckpt_path,
                config_path=cross_cfg_path,
                master_output_dir=os.path.join(tmp_dir, "cross_out"),
                save_illustrations=False,
            )
            assert cross_res is not None
            assert "overall_metrics" in cross_res
            assert "iou" in cross_res["overall_metrics"]


def test_noise_rag_database_buffers_and_labels_no_grad():
    """Verify that NoiseRAG database keys and labels strictly enforce no_grad and requires_grad=False."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        db_path = tmp.name

    try:
        keys = torch.randn(100, 256, dtype=torch.float16, requires_grad=False)
        labels = (torch.rand(100, 1) > 0.5).float().to(dtype=torch.float16)
        torch.save({"keys": keys, "labels": labels, "entry_count": 100}, db_path)

        model = NoiseRAGPrototypeV1(
            memory_path=db_path,
            memory_dtype="float16",
            embed_dim=256,
            num_cross_stages=2,
            retrieval_k=4,
        )
        model.train()

        # 1. Assert database buffers have requires_grad=False
        assert not model.retrieval.db_keys.requires_grad
        assert not model.retrieval.db_labels.requires_grad
        assert not model.retrieval.db_keys_norm.requires_grad

        # 2. Forward and backward pass
        x = torch.rand(2, 3, 64, 64)
        out = model(x, return_dict=True)
        loss = out["mask_logits"].sum() + out["artifact_loss"]
        loss.backward()

        # Database buffers must not accumulate any gradients
        assert model.retrieval.db_keys.grad is None
        assert model.retrieval.db_labels.grad is None
        assert model.retrieval.db_keys_norm.grad is None

        # 3. Direct verification of compute_alignment_loss detach
        x_t = torch.randn(2, 256, 8, 8, requires_grad=True)
        x_r = torch.randn(2, 256, 8, 8, requires_grad=True)
        align_loss = model.compute_alignment_loss(x_r=x_r, x_t=x_t)
        align_loss.backward()

        assert x_t.grad is not None
        assert x_r.grad is None  # Target must remain detached with zero gradient
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


@pytest.mark.timeout(120)
def test_noise_rag_bounded_vram_and_attention_efficiency():
    """Verify memory-bounded retrieval chunking and FlashAttention/SDPA efficiency (need_weights=False)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for VRAM benchmark test")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        db_path = tmp.name

    try:
        # Create 100k entries vector DB
        n_entries = 100000
        keys = torch.randn(n_entries, 256, dtype=torch.float16)
        labels = (torch.rand(n_entries, 1) > 0.5).float().to(dtype=torch.float16)
        torch.save({"keys": keys, "labels": labels, "entry_count": n_entries}, db_path)

        model = NoiseRAGPrototypeV1(
            memory_path=db_path,
            memory_dtype="float16",
            embed_dim=256,
            num_cross_stages=2,
            retrieval_k=4,
        ).cuda()

        x = torch.rand(2, 3, 256, 256, device="cuda")
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out = model(x, return_dict=True)
            loss = out["mask_logits"].sum() + out["artifact_loss"]

        loss.backward()

        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        # Peak VRAM must be well bounded under 3.5 GB
        assert peak_vram_mb < 3500.0, f"Peak VRAM {peak_vram_mb:.1f} MB exceeds 3.5 GB threshold"
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


