"""
Unit and integration tests for SAM2 model integration and weights management in SID-UNet.
Covers small and large model variants, LoRA parameter adaptation, dense mask prediction,
checkpoint restoration, weight pulling, and SAMRefiner integration.
"""

import os
import tempfile
import pytest
import numpy as np
import torch
from PIL import Image

pytest.importorskip("transformers", reason="SAM2 requires transformers")
pytest.importorskip("peft", reason="SAM2 requires peft")

from sid_unet.models.sam2 import (
    SAM2Segmenter,
    DEFAULT_SAM2_SMALL_CHECKPOINT,
    DEFAULT_SAM2_LARGE_CHECKPOINT,
    resolve_sam2_checkpoint,
    pull_sam2_weights,
)
from sid_unet.models.unet import build_model
from sid_unet.models.sam3_refiner import SAMRefiner
from sid_unet.utils.config import ConfigDict


def test_resolve_sam2_checkpoint():
    """Verify SAM2 checkpoint resolution for small, large, tiny, and custom paths."""
    assert resolve_sam2_checkpoint("small") == DEFAULT_SAM2_SMALL_CHECKPOINT
    assert resolve_sam2_checkpoint(variant="small") == DEFAULT_SAM2_SMALL_CHECKPOINT
    assert resolve_sam2_checkpoint("large") == DEFAULT_SAM2_LARGE_CHECKPOINT
    assert resolve_sam2_checkpoint(variant="large") == DEFAULT_SAM2_LARGE_CHECKPOINT
    assert resolve_sam2_checkpoint("custom/repo") == "custom/repo"


def test_sam2_segmenter_init_and_forward_shapes():
    """Verify SAM2Segmenter initialization and forward pass shapes with small variant."""
    model = SAM2Segmenter(
        pretrained_model_name_or_path=DEFAULT_SAM2_SMALL_CHECKPOINT,
        load_in_4bit=False,
        lora_r=4,
        lora_alpha=8,
        aux_classifier=False,
        device="cpu",
    )
    assert model is not None
    assert model.classifier_head is None

    # Forward pass on [1, 3, 64, 64]
    x = torch.randn(1, 3, 64, 64)
    mask_logits = model(x)
    assert isinstance(mask_logits, torch.Tensor)
    assert mask_logits.shape == (1, 1, 64, 64)


def test_sam2_segmenter_with_aux_classifier():
    """Verify auxiliary classifier head outputs [B, num_classes]."""
    model = SAM2Segmenter(
        pretrained_model_name_or_path=DEFAULT_SAM2_SMALL_CHECKPOINT,
        load_in_4bit=False,
        lora_r=4,
        lora_alpha=8,
        aux_classifier=True,
        num_classes=3,
        device="cpu",
    )
    assert model.classifier_head is not None

    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert isinstance(out, tuple)
    mask_logits, cls_logits = out
    assert mask_logits.shape == (1, 1, 64, 64)
    assert cls_logits.shape == (1, 3)


def test_sam2_segmenter_predict_mask():
    """Verify predict_mask produces binary output {0.0, 1.0}."""
    model = SAM2Segmenter(
        pretrained_model_name_or_path=DEFAULT_SAM2_SMALL_CHECKPOINT,
        load_in_4bit=False,
        lora_r=4,
        lora_alpha=8,
        device="cpu",
    )
    x = torch.randn(1, 3, 64, 64)
    pred = model.predict_mask(x, threshold=0.5)
    assert pred.shape == (1, 1, 64, 64)
    unique_vals = set(pred.unique().cpu().numpy())
    assert unique_vals.issubset({0.0, 1.0})


def test_sam2_segmenter_lora_trainable_parameters():
    """Verify that only LoRA parameters are trainable and base SAM2 parameters are frozen."""
    model = SAM2Segmenter(
        pretrained_model_name_or_path=DEFAULT_SAM2_SMALL_CHECKPOINT,
        load_in_4bit=False,
        lora_r=4,
        lora_alpha=8,
        device="cpu",
    )
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen = [p for p in model.parameters() if not p.requires_grad]

    assert len(trainable) > 0, "Expected active LoRA parameters"
    assert len(frozen) > 0, "Base SAM2 parameters should be frozen"

    # Forward + backward pass
    x = torch.randn(1, 3, 64, 64)
    mask_logits = model(x)
    loss = mask_logits.mean()
    loss.backward()

    active_grads = [p.grad for p in trainable if p.grad is not None]
    assert len(active_grads) > 0, "Expected active LoRA parameters to receive gradients"
    for p in frozen:
        assert p.grad is None, "Frozen base parameter unexpectedly received gradient"


def test_sam2_build_model_and_checkpoint():
    """Verify building SAM2 from config and restoring from checkpoint."""
    cfg = ConfigDict({
        "project": {"device": "cpu"},
        "model": {
            "name": "sam2",
            "variant": "small",
            "load_in_4bit": False,
            "lora_r": 4,
            "lora_alpha": 8,
            "aux_classifier": False,
        }
    })
    model = build_model(cfg)
    assert isinstance(model, SAM2Segmenter)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "sam2_test_ckpt.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": cfg.to_dict(),
        }, ckpt_path)

        loaded_model = SAM2Segmenter.from_checkpoint(ckpt_path, device="cpu")
        assert isinstance(loaded_model, SAM2Segmenter)
        x = torch.randn(1, 3, 64, 64)
        out = loaded_model(x)
        assert out.shape == (1, 1, 64, 64)


def test_sam2_refiner_integration():
    """Verify SAMRefiner integration using SAM2 model."""
    refiner = SAMRefiner(
        model_name=DEFAULT_SAM2_SMALL_CHECKPOINT,
        device="cpu",
        threshold=0.5,
        min_pixels=16,
    )
    assert refiner.is_sam2 is True

    img = Image.new("RGB", (64, 64), (100, 150, 200))
    unet_mask = np.zeros((64, 64), dtype=np.float32)
    unet_mask[10:40, 10:40] = 1.0

    refined, metrics = refiner.refine_single_sample(img, unet_mask)
    assert refined.shape == (64, 64)
    assert "pixels_changed" in metrics
    assert "error" not in metrics


def test_pull_sam2_weights_both_variants():
    """Verify weight pulling and local caching for both small and large SAM2 models."""
    small_repo = pull_sam2_weights(variant="small")
    assert small_repo == DEFAULT_SAM2_SMALL_CHECKPOINT

    large_repo = pull_sam2_weights(variant="large")
    assert large_repo == DEFAULT_SAM2_LARGE_CHECKPOINT


def test_sam2_decoder_unfrozen_trainable():
    """Verify that freeze_decoder=False sets mask_decoder parameters to requires_grad=True and receives gradients."""
    model = SAM2Segmenter(
        pretrained_model_name_or_path=DEFAULT_SAM2_SMALL_CHECKPOINT,
        load_in_4bit=False,
        lora_r=4,
        lora_alpha=8,
        freeze_decoder=False,
        device="cpu",
    )
    dec = model.get_mask_decoder()
    assert dec is not None
    dec_params = [p for p in dec.parameters()]
    assert len(dec_params) > 0
    assert all(p.requires_grad for p in dec_params)

    # Forward + backward pass to ensure gradients flow into mask_decoder
    model.train()
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    loss = out.mean()
    loss.backward()

    dec_grads = [p.grad for p in dec_params if p.grad is not None]
    assert len(dec_grads) > 0, "Active mask_decoder parameters should receive gradients"


def test_sam2_decoder_frozen_flag():
    """Verify that freeze_decoder=True keeps all mask_decoder parameters frozen."""
    model = SAM2Segmenter(
        pretrained_model_name_or_path=DEFAULT_SAM2_SMALL_CHECKPOINT,
        load_in_4bit=False,
        lora_r=4,
        lora_alpha=8,
        freeze_decoder=True,
        device="cpu",
    )
    dec = model.get_mask_decoder()
    assert dec is not None
    dec_params = [p for p in dec.parameters()]
    assert len(dec_params) > 0
    assert all(not p.requires_grad for p in dec_params)


def test_sam2_decoder_reinitialization():
    """Verify that reinit_decoder=True re-initializes mask_decoder weights."""
    model_pretrained = SAM2Segmenter(
        pretrained_model_name_or_path=DEFAULT_SAM2_SMALL_CHECKPOINT,
        load_in_4bit=False,
        lora_r=4,
        reinit_decoder=False,
        device="cpu",
    )
    model_reinit = SAM2Segmenter(
        pretrained_model_name_or_path=DEFAULT_SAM2_SMALL_CHECKPOINT,
        load_in_4bit=False,
        lora_r=4,
        reinit_decoder=True,
        device="cpu",
    )
    dec_pre = model_pretrained.get_mask_decoder()
    dec_re = model_reinit.get_mask_decoder()
    assert dec_pre is not None and dec_re is not None

    # Compare iou_token or mask_tokens weights
    pre_iou = dec_pre.iou_token.weight.detach().cpu()
    re_iou = dec_re.iou_token.weight.detach().cpu()
    assert not torch.allclose(pre_iou, re_iou), "Reinitialized weights should differ from pretrained"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU for 4-bit QLoRA")
def test_sam2_qlora_4bit_unfrozen_decoder_gpu():
    """Verify QLoRA 4-bit with unfrozen decoder executes backward pass without bitsandbytes dtype crash."""
    model = SAM2Segmenter(
        pretrained_model_name_or_path=DEFAULT_SAM2_SMALL_CHECKPOINT,
        load_in_4bit=True,
        lora_r=4,
        lora_alpha=8,
        freeze_decoder=False,
        device="cuda",
    )
    dec = model.get_mask_decoder()
    assert dec is not None
    # Mask decoder must NOT be quantized to 4-bit; must be float
    for p in dec.parameters():
        assert p.is_floating_point()
        assert p.requires_grad

    x = torch.randn(1, 3, 64, 64, device="cuda")
    model.train()
    out = model(x)
    loss = out.mean()
    loss.backward()

    grads = [p.grad for p in dec.parameters() if p.grad is not None]
    assert len(grads) > 0, "Decoder parameters should receive gradients in 4-bit QLoRA mode"


def test_sam2_mask_adapter_balanced_init():
    """Verify that mask_adapter is initialized with balanced zero bias and Xavier weights to avoid 0.0000 IoU trap."""
    model = SAM2Segmenter(
        pretrained_model_name_or_path=DEFAULT_SAM2_SMALL_CHECKPOINT,
        load_in_4bit=False,
        lora_r=4,
        lora_alpha=8,
        use_mask_adapter=True,
        device="cpu",
    )
    assert model.mask_adapter is not None
    last_conv = model.mask_adapter[-1]
    assert isinstance(last_conv, torch.nn.Conv2d)
    assert last_conv.bias is not None
    # Bias must be initialized to 0.0, avoiding the -2.0 bias trap
    assert torch.allclose(last_conv.bias, torch.zeros_like(last_conv.bias), atol=1e-5)
    # Weights must have normal/Xavier variance, not all zero
    assert last_conv.weight.std().item() > 0.01
