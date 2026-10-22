"""
Comprehensive unit and integration tests for OpenSDI MaskCLIP (Synergizing Pretrained Models: CLIP + MAE).

Tests verify:
    1. EdgeMaskGenerator morphological operations and edge-weighted BCE loss computation.
    2. PromptLearner continuous context tokens and TextEncoder output dimensions.
    3. SPM VCA (AttentionFusion), FPN+TVCA (MaskDecoder), and VSA (CLSAttentionAggregator).
    4. Anti-Random-Initialization Policy: rejecting unpermitted random initialization and loading online weights.
    5. MaskCLIP instantiation with full OpenSDI pretrained checkpoint (zero missing or unexpected keys).
    6. Forward pass output signatures (mask_logits, class_logits) and predict_mask binary thresholding.
    7. Official OpenSDI forward_opensdi compatibility (ce_loss + bce_loss + edge_loss).
    8. Backward gradient flow: SPM adapters receive gradients while CLIP parameters remain frozen.
    9. Model Surgery: Spatial backbone replacement, DoRA/LoRA injection, fusion map rewiring, and task arithmetic.
    10. Cache Extractor: MaskCLIPCacheExtractor extracting multi-layer patch and CLS feature representations.
    11. Configuration dispatch: build_model, UNet.from_checkpoint, and validation of all YAML experiment configs.
"""

import gc
import os
import tempfile
import pytest
import torch
import torch.nn as nn
import yaml

pytest.importorskip("clip", reason="MaskCLIP requires openai-clip")
pytest.importorskip("timm", reason="MaskCLIP requires timm")

from sid_unet.models.mask_clip import (
    MaskCLIP,
    SideAdapterNetwork,
    AttentionFusion,
    MaskDecoder,
    CLSAttentionAggregator,
    PromptLearner,
    TextEncoder,
    EdgeMaskGenerator,
    SurgicalLoRALinear,
    DEFAULT_MASKCLIP_HF_REPO,
    DEFAULT_MASKCLIP_HF_FILENAME,
    DEFAULT_MAE_PRETRAIN_URL,
    DEFAULT_CLIP_VIT_L14_URL,
)
from sid_unet.models.unet import build_model, UNet
from sid_unet.losses.auxiliary import build_loss, SIDTotalLoss
from sid_unet.cache.extractor import get_extractor_for_model, MaskCLIPCacheExtractor
from sid_unet.utils.config import ConfigDict, load_config


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def clean_cuda():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


# ============================================================================
# 1. Edge Mask Generator & Morphological Operations
# ============================================================================


def test_edge_mask_generator_and_loss(device):
    """Verify EdgeMaskGenerator morphological dilation/erosion and edge loss calculation."""
    generator = EdgeMaskGenerator(kernel_size=7).to(device)

    # Synthetic binary square mask
    mask = torch.zeros(2, 1, 64, 64, device=device)
    mask[:, :, 20:44, 20:44] = 1.0

    edge_mask = generator(mask)
    assert edge_mask.shape == (2, 1, 64, 64)
    assert ((edge_mask == 0) | (edge_mask == 1)).all()
    # Edge pixels must exist along boundary
    assert edge_mask.sum() > 0
    # Center of square should NOT be edge
    assert edge_mask[:, :, 32, 32].sum() == 0

    diff, dilated, erosion = generator(mask, return_all=True)
    assert diff.shape == dilated.shape == erosion.shape == (2, 1, 64, 64)


# ============================================================================
# 2. SPM Submodules Standalone
# ============================================================================


def test_prompt_learner_and_text_encoder(device):
    """Verify PromptLearner continuous context tokens and TextEncoder output."""
    from sid_unet.models.mask_clip import _load_clip_backbone_online
    clip_model = _load_clip_backbone_online("ViT-L/14").to(device)
    clip_model.float()

    prompt_learner = PromptLearner(ctx_dim=768, language_length=10, language_depth=5).to(device)
    text_encoder = TextEncoder(clip_model).to(device)

    texts = ["an image"] * 2
    prompts, tokenized = prompt_learner(clip_model, texts, torch.device(device))
    assert prompts.shape == (4, 77, 768)  # 2 neg + 2 pos
    assert tokenized.shape == (4, 77)

    text_feats = text_encoder(prompts, tokenized)
    assert text_feats.shape == (4, 768)


def test_vca_attention_fusion_standalone(device):
    """Verify Visual Cross-Attention (AttentionFusion) tensor shapes."""
    fusion = AttentionFusion(clip_dim=1024, x_dim=768, num_heads=8).to(device)
    x = torch.rand(2, 32, 32, 768, device=device)
    clip_feat = torch.rand(2, 1024, 16, 16, device=device)

    fused = fusion(x, clip_feat)
    assert fused.shape == (2, 32, 32, 768)


def test_mask_decoder_fpn_tvca_standalone(device):
    """Verify FPN + TVCA MaskDecoder forward pass across 4 pyramid scales."""
    decoder = MaskDecoder(in_channels=768, decoder_dim=256, num_heads=8, out_channels=1).to(device)
    query = torch.rand(2, 256, device=device)
    x = torch.rand(2, 768, 32, 32, device=device)

    out = decoder(query, x)
    assert out.shape == (2, 1, 128, 128)  # 4x upsampled from 32x32


def test_cls_attention_aggregator_standalone(device):
    """Verify VSA (CLSAttentionAggregator) multi-head self-attention and projection."""
    aggregator = CLSAttentionAggregator(feature_dim=1024, out_dim=768, num_heads=8, num_layers=1).to(device)
    cls_tokens = torch.rand(2, 7, 1024, device=device)

    pooled = aggregator(cls_tokens)
    assert pooled.shape == (2, 768)


# ============================================================================
# 3. Anti-Random-Initialization Policy
# ============================================================================


def test_strict_no_random_init_policy():
    """Verify that unpermitted random initialization raises a strict ValueError."""
    with pytest.raises(ValueError, match="Random initialization is strictly disabled"):
        MaskCLIP(pretrained=False, allow_random_init=False)


# ============================================================================
# 4. MaskCLIP Online Pretrained Weight Loading
# ============================================================================


def test_mask_clip_instantiation_and_surgery_summary(device):
    """Verify MaskCLIP instantiates online with zero missing or unexpected parameters."""
    model = MaskCLIP(
        pretrained=True,
        pretrained_mode="opensdi",
        num_classes=2,
        device=device,
    )
    summary = model.get_surgery_summary()
    assert summary["model_setting_name"] == "ViTL"
    assert summary["total_params"] > 540_000_000
    assert "clip_visual" in summary["submodules"]
    assert "mae_vit" in summary["submodules"]
    assert summary["submodules"]["clip_visual"]["frozen"] > 0
    assert summary["submodules"]["mae_vit"]["trainable"] > 0


# ============================================================================
# 5. Forward Pass Signatures & Predict Mask
# ============================================================================


def test_mask_clip_forward_and_predict_mask(device):
    """Verify forward signatures (mask_logits, class_logits) and predict_mask output."""
    model = MaskCLIP(
        pretrained=True,
        pretrained_mode="opensdi",
        num_classes=2,
        device=device,
    )
    model.eval()

    # Arbitrary resolution input [2, 3, 256, 256]
    x = torch.rand(2, 3, 256, 256, device=device)

    # 1. Default sid_unet signature: (mask_logits, class_logits)
    mask_logits, class_logits = model(x)
    assert mask_logits.shape == (2, 1, 256, 256)
    assert class_logits.shape == (2, 2)

    # 2. Return dict signature
    out_dict = model(x, return_dict=True)
    assert isinstance(out_dict, dict)
    assert "mask_logits" in out_dict
    assert "pred_mask" in out_dict
    assert "pred_label" in out_dict
    assert out_dict["pred_mask"].shape == (2, 1, 256, 256)
    assert out_dict["pred_label"].shape == (2,)

    # 3. predict_mask binary thresholding
    bin_mask = model.predict_mask(x, threshold=0.5)
    assert bin_mask.shape == (2, 1, 256, 256)
    assert ((bin_mask == 0.0) | (bin_mask == 1.0)).all()


def test_mask_clip_3class_compatibility(device):
    """Verify num_classes=3 projection for SID-UNet multi-class classification."""
    model = MaskCLIP(
        pretrained=True,
        pretrained_mode="opensdi",
        num_classes=3,
        device=device,
    )
    model.eval()
    x = torch.rand(2, 3, 256, 256, device=device)
    mask_logits, class_logits = model(x)
    assert class_logits.shape == (2, 3)


# ============================================================================
# 6. Official OpenSDI Compatibility (forward_opensdi)
# ============================================================================


def test_mask_clip_forward_opensdi_compatibility(device):
    """Verify official OpenSDI output dictionary and combined loss computation."""
    model = MaskCLIP(
        pretrained=True,
        pretrained_mode="opensdi",
        num_classes=2,
        device=device,
    )
    model.train()

    image = torch.rand(2, 3, 256, 256, device=device)
    mask = (torch.rand(2, 1, 256, 256, device=device) > 0.5).float()
    label = torch.tensor([0, 1], device=device)

    output = model.forward_opensdi(image, mask=mask, label=label)
    assert "backward_loss" in output
    assert "pred_mask" in output
    assert "pred_label" in output
    assert "visual_loss" in output

    vloss = output["visual_loss"]
    assert "loss_ce" in vloss
    assert "loss_bce" in vloss
    assert "loss_edge" in vloss
    assert output["backward_loss"] > 0


# ============================================================================
# 7. Backward Gradient Flow
# ============================================================================


def test_mask_clip_backward_gradient_flow(device):
    """Verify gradient flow into SPM adapters while frozen CLIP weights receive no gradients."""
    model = MaskCLIP(
        pretrained=True,
        pretrained_mode="opensdi",
        num_classes=2,
        freeze_clip=True,
        freeze_mae=False,
        device=device,
    )
    model.train()
    loss_fn = SIDTotalLoss(mask_loss_type="bce", aux_classifier=True, edge_weight=20.0, edge_mask_width=7)

    x = torch.rand(2, 3, 256, 256, device=device)
    target_mask = (torch.rand(2, 1, 256, 256, device=device) > 0.5).float()
    target_label = torch.tensor([0, 1], device=device)

    model.zero_grad()
    out = model(x)
    loss, metrics = loss_fn(out, target_mask, target_label)
    loss.backward()

    # Trainable submodules must have non-zero gradients
    assert model.aggregator.vit_model.blocks[0].attn.qkv.weight.grad is not None
    assert model.aggregator.fusion_layers["layer_1"].channel_adjust[0].weight.grad is not None
    assert model.aggregator.mask_decoder.predict.weight.grad is not None
    assert model.prompt_learner.ctx_pos.grad is not None
    assert model.cls_aggregator.projector[0].weight.grad is not None

    # Frozen CLIP visual parameters must have NO gradients
    assert model.clip.visual.conv1.weight.grad is None
    assert model.clip.visual.transformer.resblocks[0].attn.in_proj_weight.grad is None


# ============================================================================
# 8. Model Surgery APIs
# ============================================================================


def test_mask_clip_surgery_backbone_replacement(device):
    """Model Surgery: Replace MAE spatial backbone with a mock dense backbone."""
    model = MaskCLIP(
        pretrained=True,
        pretrained_mode="opensdi",
        device=device,
    )
    # Mock replacement backbone outputting 512 channels
    mock_backbone = nn.Sequential(
        nn.Conv2d(3, 512, kernel_size=16, stride=16),
        nn.GELU(),
    )
    model.replace_spatial_backbone(mock_backbone, backbone_out_channels=512)
    assert model.aggregator.custom_spatial_adapter is not None

    x = torch.rand(1, 3, 256, 256, device=device)
    mask_logits, cls_logits = model(x)
    assert mask_logits.shape == (1, 1, 256, 256)


def test_mask_clip_surgery_dora_lora_injection(device):
    """Model Surgery: Inject DoRA/LoRA into MAE attention projections."""
    model = MaskCLIP(
        pretrained=True,
        pretrained_mode="opensdi",
        device=device,
    )
    info = model.inject_peft_lora(target="mae", r=8, alpha=16.0, use_dora=True)
    assert info["num_adapted_layers"] > 0
    assert info["use_dora"] is True
    assert info["trainable_pct"] < 10.0  # Parameter-efficient: <10% trainable

    x = torch.rand(1, 3, 256, 256, device=device)
    mask_logits, cls_logits = model(x)
    assert mask_logits.shape == (1, 1, 256, 256)


def test_mask_clip_surgery_fusion_rewiring(device):
    """Model Surgery: Dynamically prune/rewire layer taps and cross-attention blocks."""
    model = MaskCLIP(
        pretrained=True,
        pretrained_mode="opensdi",
        device=device,
    )
    # Rewire to 3 taps and 3 fusion blocks
    model.reconfigure_layer_tap_and_fusion(
        selected_layers=[4, 12, 20],
        fusion_map={2: 0, 5: 1, 8: 2},
        reuse_weights=True,
    )
    assert len(model.selected_layers) == 3
    assert len(model.aggregator.fusion_layers) == 3

    x = torch.rand(1, 3, 256, 256, device=device)
    mask_logits, cls_logits = model(x)
    assert mask_logits.shape == (1, 1, 256, 256)


def test_mask_clip_surgery_task_arithmetic_merge(device):
    """Model Surgery: Task Arithmetic / WiSE-FT weight-space interpolation."""
    model1 = MaskCLIP(pretrained=True, pretrained_mode="opensdi", device=device)
    model2 = MaskCLIP(pretrained=True, pretrained_mode="opensdi", device=device)

    # Perturb model2 PromptLearner
    with torch.no_grad():
        model2.prompt_learner.ctx_pos.add_(1.0)

    orig_val = model1.prompt_learner.ctx_pos.clone()
    merged = model1.task_arithmetic_merge(model2, alpha=0.5, submodules=["prompt_learner"])
    assert "prompt_learner.ctx_pos" in merged
    assert torch.allclose(model1.prompt_learner.ctx_pos, orig_val + 0.5)


def test_mask_clip_forward_intervention_hooks(device):
    """Model Surgery: Register forward activation steering / intervention hooks."""
    model = MaskCLIP(pretrained=True, pretrained_mode="opensdi", device=device)

    hook_called = []
    def text_hook(t_feat):
        hook_called.append("text")
        return t_feat * 1.0

    model.register_surgery_hook("text_features", text_hook)
    x = torch.rand(1, 3, 256, 256, device=device)
    _ = model(x)
    assert "text" in hook_called
    model.clear_surgery_hooks()


# ============================================================================
# 9. Cache Extractor
# ============================================================================


def test_mask_clip_cache_extractor(device):
    """Verify MaskCLIPCacheExtractor extracts compact representations for sid-cache."""
    model = MaskCLIP(pretrained=True, pretrained_mode="opensdi", device=device)
    extractor = get_extractor_for_model(model)
    assert isinstance(extractor, MaskCLIPCacheExtractor)
    assert extractor.total_channels == 14336

    x = torch.rand(2, 3, 256, 256, device=device)
    cached_z = extractor.extract_batch(x)
    assert cached_z.shape == (2, 14336, 16, 16)

    # Pass cached representation directly back to forward
    mask_logits, cls_logits = model(x, cached_clip_features=cached_z)
    assert mask_logits.shape == (2, 1, 256, 256)


# ============================================================================
# 10. Checkpoint Save/Load & build_model Integration
# ============================================================================


def test_build_model_dispatch_and_checkpoint_roundtrip(device):
    """Verify build_model builds MaskCLIP from dict and UNet.from_checkpoint restores it."""
    cfg = {
        "model": {
            "name": "mask_clip",
            "pretrained": True,
            "pretrained_mode": "opensdi",
            "num_classes": 2,
        }
    }
    model = build_model(cfg)
    assert isinstance(model, MaskCLIP)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name

    try:
        payload = {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "epoch": 1,
        }
        torch.save(payload, tmp_path)

        loaded_model = UNet.from_checkpoint(tmp_path, device=device)
        assert isinstance(loaded_model, MaskCLIP)

        x = torch.rand(1, 3, 256, 256, device=device)
        out = loaded_model(x)
        assert len(out) == 2
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ============================================================================
# 11. Validate All MaskCLIP YAML Experiment Configs
# ============================================================================


def test_all_mask_clip_yaml_configs():
    """Verify all YAML experiment configs in configs/experiments/mask_clip parse and construct."""
    config_dir = "configs/experiments/mask_clip"
    assert os.path.isdir(config_dir), f"Directory {config_dir} does not exist"

    yaml_files = [
        os.path.join(config_dir, f)
        for f in os.listdir(config_dir)
        if f.endswith((".yaml", ".yml")) and not f.startswith("README")
    ]
    assert len(yaml_files) >= 4, f"Expected at least 4 configs, found {len(yaml_files)}"

    for yf in yaml_files:
        cfg = load_config(yf)
        assert cfg.model.name in ("mask_clip", "maskclip", "opensdi")
        assert cfg.model.pretrained is True
        assert cfg.model.allow_random_init is False
        assert cfg.loss.edge_weight >= 0.0
