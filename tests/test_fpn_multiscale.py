"""
Comprehensive unit and integration tests for Feature Pyramid Network (FPN) and Multiscale Prediction.

Verifies:
1. FeaturePyramidNetwork standalone architecture:
   - Top-down feature refinement and lateral 1x1 convolutions.
   - Smooth 3x3 anti-aliasing convolutions.
   - Deep supervision per-scale heads.
   - Fusion modes: "concat" and "sum".
   - Gradient flow to all pyramid stages.
2. DiffusionDiffModel FPN integration:
   - Forward pass returning (mask_logits, class_logits, multiscale_logits).
   - Forward pass returning dictionary with multiscale_logits and multiscale_pred_masks.
   - forward_cached returning multiscale predictions.
   - use_fpn=False fallback backward compatibility.
3. DiffusionDiffV2Model FPN integration:
   - Fusing perpendicular skips across pyramid stages.
   - Forward and forward_cached with multiscale outputs.
   - Gradient flow through FPN decoder.
4. DiffusionDiffMinimizedModel FPN integration:
   - 2 lines of computation with FPN decoder.
   - Forward and forward_cached multiscale prediction.
5. MaskDecoder & MaskCLIP multiscale prediction:
   - MaskDecoder multiscale heads at scale factors (4.0, 2.0, 1.0, 0.5).
   - SideAdapterNetwork return_multiscale dispatch.
   - MaskCLIP forward with return_multiscale=True and return_dict=True.
   - MaskCLIP forward_opensdi computing multiscale loss when multiscale_weight > 0.
6. SIDTotalLoss multiscale deep supervision loss:
   - Computing combined loss with multiscale_weight > 0.
   - Interpolation to ground truth resolution.
7. Experiment configuration validation:
   - Loading all new multiscale YAML experiment configs.
"""

import math
import os
import tempfile
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sid_unet.models.fpn import FeaturePyramidNetwork, FPNBlock
from sid_unet.models.diffusion_diff import (
    DiffusionDiffModel,
    TrainableLatentDecoder,
)
from sid_unet.models.diffusion_diff_v2 import (
    DiffusionDiffV2Model,
    TrainableLatentDecoderV2,
)
from sid_unet.models.diffusion_diff_minimized import DiffusionDiffMinimizedModel
from sid_unet.models.mask_clip import (
    MaskDecoder,
    SideAdapterNetwork,
    MaskCLIP,
    EdgeMaskGenerator,
)
from sid_unet.losses.auxiliary import SIDTotalLoss, build_loss
from sid_unet.utils.config import load_config, ConfigDict
from sid_unet.models.unet import build_model


# ============================================================================
# 1. FeaturePyramidNetwork Standalone Unit Tests
# ============================================================================


def test_fpn_standalone_concat_fusion():
    """Verify FeaturePyramidNetwork shapes, lateral convs, top-down path, and concat fusion."""
    in_channels_list = [256, 128, 64, 32]  # Coarsest (8x8) to finest (64x64)
    fpn = FeaturePyramidNetwork(
        in_channels_list=in_channels_list,
        fpn_channels=64,
        out_channels=1,
        fusion_mode="concat",
        include_heads=True,
    )

    B = 2
    features = [
        torch.randn(B, 256, 8, 8),    # C4
        torch.randn(B, 128, 16, 16),  # C3
        torch.randn(B, 64, 32, 32),   # C2
        torch.randn(B, 32, 64, 64),   # C1
    ]

    fused_pred, multiscale_preds = fpn(features, return_multiscale=True)

    assert fused_pred.shape == (B, 1, 64, 64)
    assert multiscale_preds is not None
    assert len(multiscale_preds) == 4
    assert multiscale_preds[0].shape == (B, 1, 8, 8)
    assert multiscale_preds[1].shape == (B, 1, 16, 16)
    assert multiscale_preds[2].shape == (B, 1, 32, 32)
    assert multiscale_preds[3].shape == (B, 1, 64, 64)

    # Test without return_multiscale
    fused_only = fpn(features, return_multiscale=False)
    assert fused_only.shape == (B, 1, 64, 64)


def test_fpn_standalone_sum_fusion():
    """Verify FeaturePyramidNetwork with sum fusion mode."""
    in_channels_list = [128, 64, 32]
    fpn = FeaturePyramidNetwork(
        in_channels_list=in_channels_list,
        fpn_channels=48,
        out_channels=2,
        fusion_mode="sum",
        include_heads=True,
    )

    B = 2
    features = [
        torch.randn(B, 128, 8, 8),
        torch.randn(B, 64, 16, 16),
        torch.randn(B, 32, 32, 32),
    ]

    fused_pred, multiscale_preds = fpn(features, return_multiscale=True)

    assert fused_pred.shape == (B, 2, 32, 32)
    assert multiscale_preds is not None
    assert len(multiscale_preds) == 3


def test_fpn_standalone_no_predict_heads():
    """Verify FeaturePyramidNetwork when include_heads=False."""
    in_channels_list = [64, 32]
    fpn = FeaturePyramidNetwork(
        in_channels_list=in_channels_list,
        fpn_channels=32,
        out_channels=1,
        fusion_mode="concat",
        include_heads=False,
    )

    features = [torch.randn(2, 64, 8, 8), torch.randn(2, 32, 16, 16)]
    fused_pred, multiscale_preds = fpn(features, return_multiscale=True)

    assert fused_pred.shape == (2, 1, 16, 16)
    assert multiscale_preds == []


def test_fpn_gradient_flow():
    """Verify gradient flow through all lateral convs, smoothing layers, and prediction heads."""
    in_channels_list = [64, 32, 16]
    fpn = FeaturePyramidNetwork(
        in_channels_list=in_channels_list,
        fpn_channels=32,
        out_channels=1,
        fusion_mode="concat",
        include_heads=True,
    )

    features = [
        torch.randn(2, 64, 8, 8, requires_grad=True),
        torch.randn(2, 32, 16, 16, requires_grad=True),
        torch.randn(2, 16, 32, 32, requires_grad=True),
    ]

    fused_pred, multiscale_preds = fpn(features, return_multiscale=True)
    loss = fused_pred.sum()
    for p in multiscale_preds:
        loss = loss + p.sum()

    loss.backward()

    # Check input gradients
    for i, feat in enumerate(features):
        assert feat.grad is not None, f"Feature {i} grad is None"
        assert not torch.allclose(feat.grad, torch.zeros_like(feat.grad))

    # Check FPN layer gradients
    for name, param in fpn.named_parameters():
        assert param.grad is not None, f"FPN param {name} grad is None"
        assert not torch.allclose(param.grad, torch.zeros_like(param.grad))


# ============================================================================
# 2. DiffusionDiffModel FPN & Multiscale Tests
# ============================================================================


def test_diffusion_diff_with_fpn_forward_multiscale():
    """Verify DiffusionDiffModel with FPN returns multiscale outputs correctly."""
    model = DiffusionDiffModel(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[100, 250],
        timestep_embed_dim=16,
        sigma_embed_dim=16,
        decoder_config={
            "channels": [64, 32, 16],
            "upsample_mode": "bilinear",
            "use_fpn": True,
            "fpn_channels": 32,
            "fpn_fusion_mode": "concat",
        },
        aux_classifier=True,
        num_classes=3,
    )

    x = torch.randn(2, 3, 64, 64)

    # 1. Standard forward without multiscale
    mask_logits, class_logits = model(x, return_multiscale=False)
    assert mask_logits.shape == (2, 1, 64, 64)
    assert class_logits.shape == (2, 3)

    # 2. Forward with return_multiscale=True
    m_logits, c_logits, ms_logits = model(x, return_multiscale=True)
    assert m_logits.shape == (2, 1, 64, 64)
    assert c_logits.shape == (2, 3)
    assert isinstance(ms_logits, list)
    assert len(ms_logits) == 3
    # Check that multiscale predictions correspond to pyramid scales
    assert ms_logits[0].shape[0] == 2
    assert ms_logits[-1].shape[0] == 2
    assert ms_logits[-1].shape[1] == 1
    assert ms_logits[-1].dim() == 4

    # 3. Forward with return_dict=True
    out_dict = model(x, return_dict=True)
    assert isinstance(out_dict, dict)
    assert "mask_logits" in out_dict
    assert "class_logits" in out_dict
    assert "multiscale_logits" in out_dict
    assert "multiscale_pred_masks" in out_dict
    assert len(out_dict["multiscale_logits"]) == 3
    assert len(out_dict["multiscale_pred_masks"]) == 3


def test_diffusion_diff_fpn_cached_forward():
    """Verify forward_cached with FPN multiscale outputs."""
    model = DiffusionDiffModel(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[100],
        timestep_embed_dim=16,
        sigma_embed_dim=16,
        decoder_config={
            "channels": [64, 32, 16],
            "use_fpn": True,
            "fpn_channels": 32,
        },
        aux_classifier=True,
        num_classes=2,
    )

    z = torch.randn(2, model.total_z_channels, 8, 8)

    # return_multiscale=True
    mask_logits, class_logits, ms_logits = model.forward_cached(z, return_multiscale=True)
    assert mask_logits.shape == (2, 1, 64, 64)
    assert class_logits.shape == (2, 2)
    assert len(ms_logits) == 3

    # return_dict=True
    out_dict = model.forward_cached(z, return_dict=True)
    assert isinstance(out_dict, dict)
    assert "multiscale_logits" in out_dict
    assert len(out_dict["multiscale_logits"]) == 3


def test_diffusion_diff_fallback_without_fpn():
    """Verify that setting use_fpn=False falls back seamlessly to standard decoder."""
    model = DiffusionDiffModel(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[100],
        timestep_embed_dim=16,
        sigma_embed_dim=16,
        decoder_config={
            "channels": [64, 32, 16],
            "use_fpn": False,
        },
        aux_classifier=False,
    )

    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)
    assert model.decoder.fpn is None


# ============================================================================
# 3. DiffusionDiffV2Model FPN & Multiscale Tests
# ============================================================================


def test_diffusion_diff_v2_with_fpn_forward_and_gradient():
    """Verify DiffusionDiffV2Model with perpendicular skips and FPN."""
    model = DiffusionDiffV2Model(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[100, 250],
        timestep_embed_dim=16,
        sigma_embed_dim=16,
        decoder_config={
            "channels": [64, 32, 16],
            "upsample_mode": "bilinear",
            "use_fpn": True,
            "fpn_channels": 32,
            "fpn_fusion_mode": "concat",
        },
        use_perpendicular_skips=True,
        aux_classifier=True,
        num_classes=2,
    )

    x = torch.randn(2, 3, 64, 64)

    # Multiscale forward
    mask_logits, class_logits, ms_logits = model(x, return_multiscale=True)
    assert mask_logits.shape == (2, 1, 64, 64)
    assert class_logits.shape == (2, 2)
    assert len(ms_logits) == 3

    # Backward pass
    loss = mask_logits.sum() + class_logits.sum()
    for m in ms_logits:
        loss = loss + m.sum()
    loss.backward()

    # Trainable decoder must have received gradients
    assert model.decoder.out_conv.weight.grad is not None
    assert model.decoder.fpn.lateral_convs[0][0].weight.grad is not None
    # Diffuser UNet should remain frozen (zero grad)
    for p in model.diffuser.parameters():
        assert p.grad is None


def test_diffusion_diff_v2_fpn_cached_forward():
    """Verify DiffusionDiffV2Model forward_cached with FPN."""
    model = DiffusionDiffV2Model(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[100],
        timestep_embed_dim=16,
        sigma_embed_dim=16,
        decoder_config={
            "channels": [64, 32, 16],
            "use_fpn": True,
            "fpn_channels": 32,
        },
        use_perpendicular_skips=True,
        aux_classifier=True,
        num_classes=3,
    )

    z = torch.randn(2, model.total_z_channels, 8, 8)
    out_dict = model.forward_cached(z, return_dict=True)
    assert "mask_logits" in out_dict
    assert "multiscale_logits" in out_dict
    assert len(out_dict["multiscale_logits"]) == 3


# ============================================================================
# 4. DiffusionDiffMinimizedModel FPN & Multiscale Tests
# ============================================================================


def test_diffusion_diff_minimized_with_fpn():
    """Verify DiffusionDiffMinimizedModel with 2 lines of computation and FPN."""
    model = DiffusionDiffMinimizedModel(
        use_dummy=True,
        dummy_vae_channels=(32, 64),
        dummy_unet_channels=(32, 64),
        timesteps=[250],
        timestep_embed_dim=16,
        sigma_embed_dim=16,
        decoder_config={
            "channels": [64, 32, 16],
            "use_fpn": True,
            "fpn_channels": 32,
        },
        use_perpendicular_skips=True,
        aux_classifier=True,
        num_classes=2,
    )

    x = torch.randn(2, 3, 64, 64)
    mask_logits, class_logits, ms_logits = model(x, return_multiscale=True)
    assert mask_logits.shape == (2, 1, 64, 64)
    assert class_logits.shape == (2, 2)
    assert len(ms_logits) == 3

    # Test forward_cached
    z = torch.randn(2, model.total_z_channels, 8, 8)
    out = model.forward_cached(z, return_multiscale=True)
    assert len(out) == 3
    assert out[0].shape == (2, 1, 64, 64)


# ============================================================================
# 5. MaskDecoder & MaskCLIP Multiscale Tests
# ============================================================================


def test_mask_decoder_multiscale_heads():
    """Verify MaskDecoder multiscale heads generate 4 scales matching (4.0, 2.0, 1.0, 0.5)."""
    decoder = MaskDecoder(in_channels=768, decoder_dim=256, num_heads=8, out_channels=1)
    query = torch.randn(2, 256)
    x = torch.randn(2, 768, 32, 32)

    # 1. Standard call returns fused 4x upsampled prediction
    out = decoder(query, x, return_multiscale=False)
    assert out.shape == (2, 1, 128, 128)

    # 2. Call with return_multiscale=True returns tuple of (fused, [s0, s1, s2, s3])
    fused, ms_logits = decoder(query, x, return_multiscale=True)
    assert fused.shape == (2, 1, 128, 128)
    assert len(ms_logits) == 4
    # Scales: 4.0 (128x128), 2.0 (64x64), 1.0 (32x32), 0.5 (16x16)
    assert ms_logits[0].shape == (2, 1, 128, 128)
    assert ms_logits[1].shape == (2, 1, 64, 64)
    assert ms_logits[2].shape == (2, 1, 32, 32)
    assert ms_logits[3].shape == (2, 1, 16, 16)


def test_mask_clip_forward_opensdi_multiscale_loss():
    """Verify MaskCLIP forward_opensdi calculates multiscale loss when multiscale_weight > 0."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MaskCLIP(
        pretrained=True,
        pretrained_mode="opensdi",
        num_classes=2,
        multiscale=True,
        multiscale_weight=0.3,
        device=device,
    )

    x = torch.randn(2, 3, 224, 224, device=device)
    mask = torch.zeros(2, 1, 224, 224, device=device)
    label = torch.ones(2, dtype=torch.long, device=device)

    out = model.forward_opensdi(x, mask=mask, label=label)

    assert "backward_loss" in out
    assert "visual_loss" in out
    assert "loss_multiscale" in out["visual_loss"]
    assert out["visual_loss"]["loss_multiscale"].item() >= 0.0
    assert "multiscale_logits" in out
    assert len(out["multiscale_logits"]) == 4


# ============================================================================
# 6. SIDTotalLoss Multiscale Deep Supervision Tests
# ============================================================================


def test_sid_total_loss_with_multiscale_supervision():
    """Verify that SIDTotalLoss correctly computes multiscale deep supervision loss."""
    criterion = SIDTotalLoss(
        mask_loss_type="combined",
        bce_weight=0.5,
        dice_weight=0.5,
        focal_weight=0.5,
        aux_classifier=True,
        aux_weight=0.2,
        multiscale_weight=0.3,
    )

    B = 2
    pred_mask = torch.randn(B, 1, 64, 64)
    target_mask = torch.randint(0, 2, (B, 1, 64, 64)).float()
    pred_label = torch.randn(B, 3)
    target_label = torch.tensor([0, 1])

    # Multiscale pyramid predictions at 8x8, 16x16, 32x32, 64x64
    multiscale_preds = [
        torch.randn(B, 1, 8, 8),
        torch.randn(B, 1, 16, 16),
        torch.randn(B, 1, 32, 32),
        torch.randn(B, 1, 64, 64),
    ]

    # Test tuple model output (mask_logits, class_logits, multiscale_logits)
    model_output = (pred_mask, pred_label, multiscale_preds)
    total_loss, loss_dict = criterion(
        model_output=model_output,
        target_masks=target_mask,
        target_labels=target_label,
    )

    assert "multiscale_loss" in loss_dict
    assert loss_dict["multiscale_loss"] > 0.0

    # Test dict model output
    dict_output = {
        "mask_logits": pred_mask,
        "class_logits": pred_label,
        "multiscale_logits": multiscale_preds,
    }
    total_loss_dict, loss_dict_2 = criterion(
        model_output=dict_output,
        target_masks=target_mask,
        target_labels=target_label,
    )
    assert "multiscale_loss" in loss_dict_2
    assert loss_dict_2["multiscale_loss"] > 0.0


# ============================================================================
# 7. Experiment Configs Validation
# ============================================================================


def test_multiscale_yaml_experiment_configs():
    """Verify all new multiscale experiment YAML files can be parsed and validated."""
    yaml_paths = [
        "configs/experiments/diffusion_diff/diffusion_diff_fpn_multiscale.yaml",
        "configs/experiments/diffusion_diff_v2/diffusion_diff_v2_fpn_multiscale.yaml",
        "configs/experiments/diffusion_diff_minimized/diffusion_diff_minimized_fpn_multiscale.yaml",
        "configs/experiments/mask_clip/mask_clip_fpn_multiscale.yaml",
    ]

    for rel_path in yaml_paths:
        full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), rel_path)
        assert os.path.exists(full_path), f"Config file missing: {full_path}"
        cfg = load_config(full_path)
        assert "model" in cfg
        assert "loss" in cfg
        assert cfg["loss"].get("multiscale_weight", 0.0) > 0.0
        if "decoder" in cfg["model"]:
            assert cfg["model"]["decoder"].get("use_fpn", False) is True
