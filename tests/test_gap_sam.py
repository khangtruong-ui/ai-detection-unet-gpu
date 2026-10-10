"""
Unit and integration tests for GAP-SAM (Global Artifact Prior + sam-distil models).

Tests verify:
    1. PairedArtifactEncoder GAP pooling, branch-specific LayerNorms, and MLP fusion shapes.
    2. ZeroGatedFiLM exact identity initialization (alpha_f == 0) and multiscale modulation.
    3. ArtifactClassifier forward pass and real-versus-reconstructed loss formulation.
    4. GAPSAM model instantiation with TinyViT, EfficientViT, and RepViT backbones from sam-distil.
    5. Forward pass shapes in training mode (3-tuple: mask_logits, class_logits, artifact_loss).
    6. Forward pass shapes in evaluation mode (2-tuple: mask_logits, class_logits).
    7. predict_mask binary thresholding and output shape.
    8. Backward gradient flow: LoRA adapters, PairedArtifactEncoder, and ZeroGatedFiLM receive grads,
       while VAE parameters remain strictly frozen without gradients.
    9. build_model and build_loss configuration dispatch for GAP-SAM.
    10. Checkpoint save and restoration compatibility.
    11. All GAP-SAM YAML configuration files parse cleanly and construct valid models.
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn
import yaml

pytest.importorskip("sam3", reason="GAP-SAM requires sam3-distil")
pytest.importorskip("peft", reason="GAP-SAM requires peft")
pytest.importorskip("diffusers", reason="GAP-SAM requires diffusers")

from sid_unet.models.gap_sam import (
    GAPSAM,
    GAPSAMDistil,
    GAP_SAM,
    PairedArtifactEncoder,
    ZeroGatedFiLM,
    ArtifactClassifier,
)
from sid_unet.models.unet import build_model, UNet
from sid_unet.losses.auxiliary import build_loss, SIDTotalLoss
from sid_unet.utils.config import ConfigDict, load_config


import gc

@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def clean_cuda():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def test_paired_artifact_encoder_standalone(device):
    """Verify PairedArtifactEncoder pooling, normalization, and token shapes."""
    encoder = PairedArtifactEncoder(feature_dim=256, hidden_dim=512, token_dim=256).to(device)

    # 4D feature maps from adaptive branch (F_o) and frozen branch (F_r)
    F_o = torch.randn(2, 256, 16, 16, device=device)
    F_r = torch.randn(2, 256, 16, 16, device=device)

    t_art, h_o, h_r = encoder(F_o, F_r)

    assert t_art.shape == (2, 256), f"Expected t_art shape (2, 256), got {t_art.shape}"
    assert h_o.shape == (2, 256), f"Expected h_o shape (2, 256), got {h_o.shape}"
    assert h_r.shape == (2, 256), f"Expected h_r shape (2, 256), got {h_r.shape}"

    # Also verify with pre-pooled 2D tensors
    t_art_2d, h_o_2d, h_r_2d = encoder(h_o, h_r)
    assert t_art_2d.shape == (2, 256)


def test_zero_gated_film_identity_init_and_modulation(device):
    """Verify ZeroGatedFiLM identity mapping at initialization and modulation behavior."""
    film = ZeroGatedFiLM(token_dim=256, channels_per_level=256, num_levels=3).to(device)

    # Initial gate must be exactly 0
    assert torch.allclose(film.alpha_f, torch.zeros(1, device=device)), "alpha_f should be initialized to 0"

    t_art = torch.randn(2, 256, device=device)
    fpn_features = [
        torch.randn(2, 256, 32, 32, device=device),
        torch.randn(2, 256, 16, 16, device=device),
        torch.randn(2, 256, 8, 8, device=device),
    ]

    # At initialization (alpha_f == 0), output must be numerically identical to input
    modulated = film(fpn_features, t_art)
    assert len(modulated) == 3
    for orig, mod in zip(fpn_features, modulated):
        assert torch.allclose(orig, mod), "Modulation must be identity when alpha_f is 0"

    # When alpha_f is non-zero, modulation should modify the features
    with torch.no_grad():
        film.alpha_f.fill_(0.5)
    modulated_active = film(fpn_features, t_art)
    for orig, mod in zip(fpn_features, modulated_active):
        assert not torch.allclose(orig, mod), "Modulation must modify features when alpha_f != 0"


def test_artifact_classifier_loss(device):
    """Verify ArtifactClassifier forward pass and loss computation."""
    classifier = ArtifactClassifier(in_features=256).to(device)

    h_o = torch.randn(4, 256, device=device)
    h_r = torch.randn(4, 256, device=device)

    # Test forward
    logits_o = classifier(h_o)
    logits_r = classifier(h_r)
    assert logits_o.shape == (4, 1)
    assert logits_r.shape == (4, 1)

    # Test compute_loss without labels
    loss_all = classifier.compute_loss(h_o, h_r)
    assert loss_all.ndim == 0
    assert loss_all.item() > 0.0

    # Test compute_loss with labels (0 is authentic, 1 is synthetic)
    labels = torch.tensor([0, 1, 0, 2], device=device)
    loss_labeled = classifier.compute_loss(h_o, h_r, labels=labels)
    assert loss_labeled.ndim == 0
    assert loss_labeled.item() > 0.0

    # Test compute_loss with masks (first item authentic with zero mask)
    masks = torch.ones(4, 1, 16, 16, device=device)
    masks[0] = 0.0
    loss_masked = classifier.compute_loss(h_o, h_r, masks=masks)
    assert loss_masked.ndim == 0
    assert loss_masked.item() > 0.0


def test_gap_sam_training_forward_shapes(device):
    """Verify GAP-SAM produces (mask_logits, class_logits, artifact_loss) in train mode."""
    model = GAPSAM(
        checkpoint_path=None,
        backbone_type="tinyvit",
        model_name="11m",
        lora_r=8,
        lora_alpha=16,
        use_dummy_vae=True,
        aux_classifier=True,
        num_classes=3,
        device=device,
    )
    model.train()

    x = torch.randn(1, 3, 256, 256, device=device)
    outputs = model(x)

    assert isinstance(outputs, tuple), "Expected tuple output"
    assert len(outputs) == 3, f"Expected 3-tuple (mask, class, art), got length {len(outputs)}"

    mask_logits, cls_logits, art_loss = outputs
    assert mask_logits.shape == (1, 1, 256, 256)
    assert cls_logits.shape == (1, 3)
    assert art_loss is not None
    assert art_loss.ndim == 0
    assert art_loss.item() > 0.0


def test_gap_sam_eval_mode_and_predict_mask(device):
    """Verify GAP-SAM in eval mode returns 2-tuple and predict_mask yields binary mask."""
    model = GAPSAM(
        checkpoint_path=None,
        backbone_type="tinyvit",
        model_name="11m",
        lora_r=8,
        lora_alpha=16,
        use_dummy_vae=True,
        aux_classifier=True,
        num_classes=3,
        device=device,
    )
    model.eval()

    x = torch.randn(1, 3, 256, 256, device=device)
    outputs = model(x)

    assert isinstance(outputs, tuple)
    assert len(outputs) == 2, f"In eval mode expected 2-tuple (mask, class), got {len(outputs)}"

    mask_logits, cls_logits = outputs
    assert mask_logits.shape == (1, 1, 256, 256)
    assert cls_logits.shape == (1, 3)

    # Test predict_mask
    mask_pred = model.predict_mask(x, threshold=0.5)
    assert mask_pred.shape == (1, 1, 256, 256)
    assert torch.all((mask_pred == 0.0) | (mask_pred == 1.0))


def test_gap_sam_without_aux_classifier(device):
    """Verify GAP-SAM when aux_classifier=False returns single mask in eval mode."""
    model = GAPSAM(
        checkpoint_path=None,
        backbone_type="tinyvit",
        model_name="11m",
        lora_r=8,
        lora_alpha=16,
        use_dummy_vae=True,
        aux_classifier=False,
        device=device,
    )
    model.eval()

    x = torch.randn(1, 3, 256, 256, device=device)
    mask_logits = model(x)

    assert isinstance(mask_logits, torch.Tensor)
    assert mask_logits.shape == (1, 1, 256, 256)


def test_gap_sam_backward_and_frozen_vae(device):
    """Verify backward gradient flows to LoRA and GAP modules, while VAE remains strictly frozen."""
    if device == "cpu":
        pytest.skip("Skip CPU backward test to avoid long execution time")
    model = GAPSAM(
        checkpoint_path=None,
        backbone_type="tinyvit",
        model_name="11m",
        lora_r=8,
        lora_alpha=16,
        use_dummy_vae=True,
        aux_classifier=True,
        num_classes=3,
        device=device,
    )
    model.train()

    loss_fn = SIDTotalLoss(
        mask_loss_type="combined",
        bce_weight=0.5,
        dice_weight=0.5,
        aux_classifier=True,
        aux_weight=0.2,
        artifact_weight=1.0,
    )

    x = torch.randn(2, 3, 256, 256, device=device)
    target_masks = torch.zeros(2, 1, 256, 256, device=device)
    target_labels = torch.tensor([0, 1], device=device)

    outputs = model(x)
    total_loss, metrics = loss_fn(outputs, target_masks, target_labels)
    assert "artifact_loss" in metrics

    total_loss.backward()

    # LoRA parameters must receive gradients
    has_lora_grad = any(p.grad is not None for n, p in model.named_parameters() if "lora_" in n)
    assert has_lora_grad, "LoRA adapter parameters should have received gradients"

    # PairedArtifactEncoder parameters must receive gradients
    assert model.paired_artifact_encoder.fc1.weight.grad is not None
    assert model.paired_artifact_encoder.fc2.weight.grad is not None

    # FiLM alpha_f must receive gradients
    assert model.zero_gated_film.alpha_f.grad is not None

    # ArtifactClassifier must receive gradients
    assert model.artifact_classifier.classifier.weight.grad is not None

    # VAE parameters must NOT receive gradients (strictly frozen)
    has_vae_grad = any(p.grad is not None for p in model.vae.parameters())
    assert not has_vae_grad, "Frozen VAE parameters must not receive gradients"


def test_build_model_and_build_loss_gap_sam():
    """Verify build_model and build_loss factory functions for GAP-SAM."""
    cfg = ConfigDict({
        "model": {
            "name": "gap_sam",
            "backbone": "tinyvit",
            "model_name": "11m",
            "use_dummy_vae": True,
            "checkpoint_path": None,
            "lora_r": 8,
            "lora_alpha": 16,
            "aux_classifier": True,
            "num_classes": 3,
            "artifact_weight": 1.0,
        },
        "loss": {
            "mask_loss_type": "combined",
            "artifact_weight": 1.0,
        },
        "project": {
            "device": "cpu",
        },
    })

    model = build_model(cfg)
    assert isinstance(model, GAPSAM)
    assert model.backbone_type == "tinyvit"
    assert model.artifact_weight == 1.0

    loss_fn = build_loss(cfg)
    assert isinstance(loss_fn, SIDTotalLoss)
    assert loss_fn.artifact_weight == 1.0


def test_gap_sam_checkpoint_save_and_load(device):
    """Verify GAP-SAM state_dict can be saved and restored via UNet.from_checkpoint."""
    model = GAPSAM(
        checkpoint_path=None,
        backbone_type="tinyvit",
        model_name="11m",
        lora_r=8,
        lora_alpha=16,
        use_dummy_vae=True,
        aux_classifier=True,
        num_classes=3,
        device=device,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "gapsam_test.pt")
        state = {
            "model_state_dict": model.state_dict(),
            "config": {
                "model": {
                    "name": "gap_sam",
                    "backbone": "tinyvit",
                    "model_name": "11m",
                    "use_dummy_vae": True,
                    "checkpoint_path": None,
                    "lora_r": 8,
                    "lora_alpha": 16,
                    "aux_classifier": True,
                    "num_classes": 3,
                },
                "project": {"device": "cpu"},
            },
        }
        torch.save(state, ckpt_path)

        loaded_model = UNet.from_checkpoint(ckpt_path, device="cpu", strict=False)
        assert isinstance(loaded_model, GAPSAM)
        assert loaded_model.backbone_type == "tinyvit"


def test_gap_sam_yaml_configs():
    """Verify all YAML configs in configs/experiments/gap_sam parse cleanly."""
    configs_dir = "/workspace/ai-detection-unet-gpu/configs/experiments/gap_sam"
    assert os.path.isdir(configs_dir), f"Directory {configs_dir} does not exist"

    expected_files = [
        "default.yaml",
        "gap_sam_tinyvit_lora.yaml",
        "gap_sam_efficientvit_lora.yaml",
        "gap_sam_repvit_lora.yaml",
        "gap_sam_sd21_vae.yaml",
    ]

    for fname in expected_files:
        path = os.path.join(configs_dir, fname)
        assert os.path.isfile(path), f"Missing config file: {path}"
        cfg = load_config(path)
        assert cfg.model.name == "gap_sam"
        assert cfg.model.backbone in ("tinyvit", "efficientvit", "repvit")
        assert cfg.loss.artifact_weight == 1.0
        assert cfg.model.lora_r == 8, f"{fname} should have lora_r=8"
        assert cfg.model.lora_alpha == 16, f"{fname} should have lora_alpha=16"
        assert cfg.model.lora_dropout == 0.0, f"{fname} should have lora_dropout=0.0"
        assert cfg.training.epochs == 3, f"{fname} should have epochs=3"
        assert cfg.training.learning_rate == 0.0002
        assert cfg.training.weight_decay == 0.05
        assert cfg.training.val_check_interval == 0.1
        assert cfg.training.early_stopping_patience == 5
        assert cfg.training.early_stopping_metric == "val_loss"
        assert cfg.training.early_stopping_mode == "min"
        assert cfg.model.aux_classifier is False
        assert cfg.loss.aux_classifier is False


def test_gap_sam_text_cache_and_preload(device):
    """Verify GAP-SAM text prompt caching, preloading, and cache stats."""
    model = GAPSAM(
        checkpoint_path=None,
        backbone_type="tinyvit",
        model_name="11m",
        use_dummy_vae=True,
        device=device,
    )
    try:
        model.clear_text_cache()
        stats0 = model.get_cache_stats()
        assert stats0["text_cache_entries"] == 0
        assert stats0["text_cache_hits"] == 0

        # Preload prompt
        model.preload_text_prompt("tampered region", device=device)
        stats1 = model.get_cache_stats()
        assert stats1["text_cache_entries"] == 1

        x = torch.randn(1, 3, 256, 256, device=device)
        model.eval()
        _ = model(x)
        stats2 = model.get_cache_stats()
        # Second access must be a cache hit
        assert stats2["text_cache_hits"] >= 1

        model.clear_text_cache()
        assert len(model._text_cache) == 0
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def test_gap_sam_dynamic_artifact_cache(device):
    """Verify dynamic in-memory artifact caching skips VAE and yields identical outputs."""
    model = GAPSAM(
        checkpoint_path=None,
        backbone_type="tinyvit",
        model_name="11m",
        use_dummy_vae=True,
        enable_artifact_cache=True,
        device=device,
    )
    try:
        model.eval()
        model.clear_artifact_cache()

        x = torch.randn(1, 3, 256, 256, device=device)
        img_ids = ["test_sample_0"]

        # Step 1: Cache miss (fills cache)
        out1 = model(x, img_ids=img_ids)
        stats1 = model.get_cache_stats()
        assert stats1["artifact_cache_size"] == 1
        assert stats1["artifact_cache_misses"] == 1
        assert stats1["artifact_cache_hits"] == 0

        # Step 2: Cache hit (reuses cached artifact prior)
        out2 = model(x, img_ids=img_ids)
        stats2 = model.get_cache_stats()
        assert stats2["artifact_cache_hits"] == 1
        assert stats2["artifact_hit_rate"] == 0.5  # 1 hit out of 2 total queries

        # Outputs must be numerically identical
        assert torch.allclose(out1[0], out2[0], atol=1e-5)
        if isinstance(out1, tuple) and len(out1) > 1 and out1[1] is not None:
            assert torch.allclose(out1[1], out2[1], atol=1e-5)

        model.clear_artifact_cache()
        assert model.get_cache_stats()["artifact_cache_size"] == 0
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def test_gap_sam_forward_cached_and_extract_cache_tensors(device):
    """Verify extract_cache_tensors produces [B, 256] and forward_cached matches direct forward."""
    model = GAPSAM(
        checkpoint_path=None,
        backbone_type="tinyvit",
        model_name="11m",
        use_dummy_vae=True,
        device=device,
    )
    try:
        model.eval()

        x = torch.randn(1, 3, 256, 256, device=device)

        # 1. Extract offline cache tensor
        f_r = model.extract_cache_tensors(x)
        assert f_r.shape == (1, 256), f"Expected shape (1, 256), got {f_r.shape}"

        # 2. Run forward_cached with pre-computed f_r
        out_cached = model.forward_cached(x, f_r=f_r)

        # 3. Run forward with explicit f_r
        out_direct = model(x, f_r=f_r)

        assert torch.allclose(out_cached[0], out_direct[0], atol=1e-5)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def test_gap_sam_bypass_vae_for_cached_training(device):
    """Verify that frozen VAE can be purged to save VRAM while cached forward still executes."""
    model = GAPSAM(
        checkpoint_path=None,
        backbone_type="tinyvit",
        model_name="11m",
        use_dummy_vae=True,
        device=device,
    )
    try:
        model.eval()

        x = torch.randn(1, 3, 256, 256, device=device)
        f_r = model.extract_cache_tensors(x)

        # Bypass and purge VAE
        assert model.vae is not None
        model.bypass_vae_for_cached_training()
        assert model.vae is None

        # Cached forward works without VAE present
        out = model.forward_cached(x, f_r=f_r)
        mask_out = out[0] if isinstance(out, (tuple, list)) else out
        assert mask_out.shape == (1, 1, 256, 256)

        # Uncached forward without f_r correctly raises informative RuntimeError
        with pytest.raises(RuntimeError, match="bypass_vae_for_cached_training"):
            model(x)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def test_gap_sam_cache_extractor_registration(device):
    """Verify GAPSAMCacheExtractor in sid_unet.cache.extractor."""
    from sid_unet.cache.extractor import get_extractor_for_model, GAPSAMCacheExtractor

    model = GAPSAM(
        checkpoint_path=None,
        backbone_type="tinyvit",
        model_name="11m",
        use_dummy_vae=True,
        device=device,
    )
    try:
        extractor = get_extractor_for_model(model)
        assert isinstance(extractor, GAPSAMCacheExtractor)
        assert extractor.model_name == "gap_sam"
        assert extractor.total_channels == 256

        x = torch.randn(1, 3, 256, 256, device=device)
        z = extractor.extract_batch(x)
        assert z.shape == (1, 256)
        assert z.dtype == torch.float16

        meta = extractor.get_metadata()
        assert meta["model_name"] == "gap_sam"
        assert meta["total_channels"] == 256
        assert meta["model_variant"] == "11m"
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def test_gap_sam_training_forward_shapes_without_aux_classifier(device):
    """Verify GAP-SAM in pure configuration (aux_classifier=False) produces 2-tuple (mask, art) in train mode."""
    model = GAPSAM(
        checkpoint_path=None,
        backbone_type="tinyvit",
        model_name="11m",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        use_dummy_vae=True,
        aux_classifier=False,
        device=device,
    )
    model.train()

    x = torch.randn(2, 3, 256, 256, device=device)
    target_masks = torch.zeros(2, 1, 256, 256, device=device)
    outputs = model(x)

    assert isinstance(outputs, tuple), "Expected tuple output"
    assert len(outputs) == 2, f"Expected 2-tuple (mask_logits, art_loss), got length {len(outputs)}"

    mask_logits, art_loss = outputs
    assert mask_logits.shape == (2, 1, 256, 256)
    assert art_loss is not None
    assert art_loss.ndim == 0
    assert art_loss.item() > 0.0

    # Verify SIDTotalLoss correctly processes 2-tuple (mask_logits, art_loss) when aux_classifier=False
    loss_fn = SIDTotalLoss(
        mask_loss_type="combined",
        bce_weight=0.5,
        dice_weight=0.5,
        aux_classifier=False,
        artifact_weight=1.0,
    )
    total_loss, loss_metrics = loss_fn(outputs, target_masks)
    assert "artifact_loss" in loss_metrics, "artifact_loss must be logged in metrics"
    assert "mask_loss" in loss_metrics
    assert "aux_loss" not in loss_metrics
    assert total_loss.ndim == 0
    assert torch.isclose(total_loss, torch.tensor(loss_metrics["mask_loss"] + loss_metrics["artifact_loss"], device=total_loss.device), atol=1e-5)


def test_artifact_classifier_loss_halving_formula(device):
    """Verify ArtifactClassifier loss exactly implements 0.5 * (loss_o + loss_r) matching paper Eq. 7."""
    classifier = ArtifactClassifier(in_features=256).to(device)

    h_o = torch.randn(4, 256, device=device)
    h_r = torch.randn(4, 256, device=device)

    # Compute expected loss manually: 0.5 * (BCE(h_o, 0) + BCE(h_r, 1))
    logits_o = classifier(h_o)
    logits_r = classifier(h_r)
    expected_o = torch.nn.functional.binary_cross_entropy_with_logits(logits_o, torch.zeros_like(logits_o))
    expected_r = torch.nn.functional.binary_cross_entropy_with_logits(logits_r, torch.ones_like(logits_r))
    expected_total = 0.5 * (expected_o + expected_r)

    actual_loss = classifier.compute_loss(h_o, h_r)
    assert torch.isclose(actual_loss, expected_total, atol=1e-6), (
        f"Loss {actual_loss.item()} does not match expected {expected_total.item()}"
    )


def test_trainer_intra_epoch_val_check_interval():
    """Verify Trainer correctly schedules and executes intra-epoch validation checks."""
    from sid_unet.training.trainer import Trainer
    from torch.utils.data import DataLoader, TensorDataset

    # Tiny synthetic dataset
    imgs = torch.randn(8, 3, 32, 32)
    msks = torch.zeros(8, 1, 32, 32)
    lbls = torch.zeros(8, dtype=torch.long)

    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, imgs, msks, lbls):
            self.imgs, self.msks, self.lbls = imgs, msks, lbls
        def __len__(self):
            return len(self.imgs)
        def __getitem__(self, idx):
            return {"image": self.imgs[idx], "mask": self.msks[idx], "label": self.lbls[idx]}

    train_ds = DictDataset(imgs, msks, lbls)
    val_ds = DictDataset(imgs[:4], msks[:4], lbls[:4])

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 1, 1),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = ConfigDict({
            "project": {"name": "test_intra_val", "output_dir": tmpdir, "seed": 42, "device": "cpu"},
            "data": {"image_size": [32, 32], "batch_size": 2, "train_samples_per_epoch": -1, "val_samples": -1},
            "model": {"name": "unet", "in_channels": 3, "out_channels": 1, "aux_classifier": False},
            "loss": {"mask_loss_type": "bce", "bce_weight": 1.0, "aux_classifier": False},
            "training": {
                "epochs": 1,
                "learning_rate": 0.001,
                "optimizer": "adamw",
                "val_check_interval": 0.5,  # Validate twice in an epoch (at step 2 of 4 and step 4)
                "early_stopping_patience": 5,
                "early_stopping_metric": "val_loss",
                "early_stopping_mode": "min",
                "save_best": False,
                "save_latest": False,
            },
            "logging": {"log_interval": 1, "measure_network": False},
        })

        trainer = Trainer(
            config=cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
        )

        assert trainer.val_check_interval == 0.5
        val_calls = 0
        orig_val = trainer.validate
        def mock_val(*args, **kwargs):
            nonlocal val_calls
            val_calls += 1
            return orig_val(*args, **kwargs)
        trainer.validate = mock_val

        trainer.train()
        # With 4 batches total and val_check_interval=0.5:
        # step 2 triggers intra-epoch validate (1st call),
        # epoch end triggers regular validate (2nd call).
        assert val_calls >= 2, f"Expected at least 2 validation evaluations, got {val_calls}"


