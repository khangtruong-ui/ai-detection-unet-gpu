"""
Tests for SAM3-Distil (EfficientSAM3) + LoRA integration.
Verifies model instantiation, PEFT parameter efficiency, forward/backward passes,
predict_mask thresholding, build_model dispatch, and checkpoint restoration.
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn

from sid_unet.models.sam3_distil import SAM3DistilLoRA, SAM3Distil
from sid_unet.models.unet import build_model, UNet
from sid_unet.utils.config import ConfigDict


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_sam3_distil_init_and_forward_shapes(device):
    """Verify SAM3DistilLoRA forward produces expected mask and aux class shapes."""
    model = SAM3DistilLoRA(
        checkpoint_path=None,
        backbone_type="tinyvit",
        model_name="11m",
        text_encoder_type="MobileCLIP-S0",
        text_encoder_context_length=16,
        lora_r=8,
        lora_alpha=16,
        aux_classifier=True,
        num_classes=3,
        device=device,
    )

    x = torch.randn(2, 3, 256, 256, device=device)
    outputs = model(x)
    assert isinstance(outputs, tuple), "Expected tuple (mask_logits, class_logits)"
    mask_logits, cls_logits = outputs

    assert mask_logits.shape == (2, 1, 256, 256)
    assert cls_logits.shape == (2, 3)


def test_sam3_distil_without_aux_classifier(device):
    """Verify that disabling aux_classifier returns a single mask tensor."""
    model = SAM3DistilLoRA(
        checkpoint_path=None,
        backbone_type="tinyvit",
        model_name="11m",
        text_encoder_type="MobileCLIP-S0",
        text_encoder_context_length=16,
        lora_r=8,
        lora_alpha=16,
        aux_classifier=False,
        device=device,
    )

    x = torch.randn(1, 3, 256, 256, device=device)
    outputs = model(x)
    assert isinstance(outputs, torch.Tensor), "Expected single tensor when aux_classifier=False"
    assert outputs.shape == (1, 1, 256, 256)


def test_sam3_distil_predict_mask(device):
    """Verify predict_mask produces binary masks with correct shape and range."""
    model = SAM3DistilLoRA(
        checkpoint_path=None,
        backbone_type="tinyvit",
        model_name="11m",
        text_encoder_type="MobileCLIP-S0",
        text_encoder_context_length=16,
        lora_r=8,
        lora_alpha=16,
        device=device,
    )

    x = torch.randn(1, 3, 256, 256, device=device)
    pred = model.predict_mask(x, threshold=0.5)
    assert pred.shape == (1, 1, 256, 256)
    unique_vals = set(torch.unique(pred).cpu().tolist())
    assert unique_vals.issubset({0.0, 1.0}), f"Expected binary mask values, got {unique_vals}"


def test_sam3_distil_lora_trainable_parameters(device):
    """Verify that only LoRA and classifier head parameters are trainable."""
    model = SAM3DistilLoRA(
        checkpoint_path=None,
        backbone_type="tinyvit",
        model_name="11m",
        text_encoder_type="MobileCLIP-S0",
        text_encoder_context_length=16,
        lora_r=8,
        lora_alpha=16,
        aux_classifier=True,
        device=device,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100.0 * trainable_params / total_params

    assert trainable_params > 0, "Expected trainable LoRA parameters"
    assert trainable_pct < 3.0, f"Expected < 3% trainable parameters, got {trainable_pct:.2f}%"

    # Base parameters must have requires_grad=False
    for name, param in model.peft_model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad is True, f"LoRA parameter {name} should require grad"
        else:
            assert param.requires_grad is False, f"Base parameter {name} should be frozen"


def test_sam3_distil_backward_gradient_flow(device):
    """Verify that gradients compute and flow exclusively into LoRA adapters."""
    if device == "cpu":
        pytest.skip("Skip CPU backward test to avoid long execution time")

    model = SAM3DistilLoRA(
        checkpoint_path=None,
        backbone_type="tinyvit",
        model_name="11m",
        text_encoder_type="MobileCLIP-S0",
        text_encoder_context_length=16,
        lora_r=8,
        lora_alpha=16,
        aux_classifier=True,
        device=device,
    )

    x = torch.randn(1, 3, 256, 256, device=device)
    mask_logits, cls_logits = model(x)
    loss = mask_logits.sum() + cls_logits.sum()
    loss.backward()

    lora_grads = [p.grad for n, p in model.peft_model.named_parameters() if "lora_" in n and p.grad is not None]
    base_grads = [p.grad for n, p in model.peft_model.named_parameters() if "lora_" not in n and p.grad is not None]

    assert len(lora_grads) > 0, "Expected computed gradients on LoRA parameters"
    assert len(base_grads) == 0, "Base parameters must not have gradients"
    if model.classifier_head is not None:
        assert model.classifier_head.weight.grad is not None


def test_sam3_distil_build_model_and_checkpoint(device):
    """Verify build_model dispatch and UNet.from_checkpoint serialization round-trip."""
    cfg = ConfigDict({
        "project": {"device": device},
        "model": {
            "name": "sam3_distil",
            "backbone": "tinyvit",
            "model_name": "11m",
            "checkpoint_path": None,
            "lora_r": 8,
            "lora_alpha": 16,
            "aux_classifier": True,
        }
    })

    model = build_model(cfg)
    assert isinstance(model, SAM3DistilLoRA)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "sam3_distil_test_ckpt.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": cfg.to_dict(),
            "epoch": 1,
        }, ckpt_path)

        loaded_model, loaded_cfg = UNet.from_checkpoint(
            ckpt_path,
            device=device,
            return_config=True,
        )
        assert isinstance(loaded_model, SAM3DistilLoRA)
        assert loaded_cfg.model.name == "sam3_distil"
