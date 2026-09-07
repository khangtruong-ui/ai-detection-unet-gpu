import os
import tempfile
import pytest
import torch
from PIL import Image

from sid_unet.models.sam3_qlora import SAM3QLoRA
from sid_unet.models.unet import build_model
from sid_unet.utils.config import ConfigDict


def test_sam3_qlora_init_and_forward_shapes():
    """Verify SAM3-QLoRA model initialization, forward pass, and spatial resizing."""
    model = SAM3QLoRA(
        pretrained_model_name_or_path="yujiepan/sam3-tiny-random",
        load_in_4bit=False,
        lora_r=4,
        lora_alpha=8,
        aux_classifier=False,
        device="cpu",
    )
    assert model is not None
    assert model.classifier_head is None

    # Test forward pass with arbitrary spatial resolution (e.g. 128x128)
    x = torch.randn(2, 3, 128, 128)
    mask_logits = model(x)
    assert isinstance(mask_logits, torch.Tensor)
    assert mask_logits.shape == (2, 1, 128, 128)


def test_sam3_qlora_with_aux_classifier():
    """Verify auxiliary classifier head outputs [B, num_classes]."""
    model = SAM3QLoRA(
        pretrained_model_name_or_path="yujiepan/sam3-tiny-random",
        load_in_4bit=False,
        lora_r=4,
        lora_alpha=8,
        aux_classifier=True,
        num_classes=3,
        device="cpu",
    )
    assert model.classifier_head is not None

    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert isinstance(out, tuple)
    mask_logits, cls_logits = out
    assert mask_logits.shape == (2, 1, 64, 64)
    assert cls_logits.shape == (2, 3)


def test_sam3_qlora_predict_mask():
    """Verify predict_mask produces binary output {0.0, 1.0}."""
    model = SAM3QLoRA(
        pretrained_model_name_or_path="yujiepan/sam3-tiny-random",
        load_in_4bit=False,
        lora_r=4,
        lora_alpha=8,
        device="cpu",
    )
    x = torch.randn(2, 3, 64, 64)
    pred = model.predict_mask(x, threshold=0.5)
    assert pred.shape == (2, 1, 64, 64)
    unique_vals = set(pred.unique().cpu().numpy())
    assert unique_vals.issubset({0.0, 1.0})


def test_sam3_qlora_backward_and_lora_trainable_params():
    """Verify that only LoRA parameters are trainable and receive gradients."""
    model = SAM3QLoRA(
        pretrained_model_name_or_path="yujiepan/sam3-tiny-random",
        load_in_4bit=False,
        lora_r=4,
        lora_alpha=8,
        device="cpu",
    )
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    frozen = [p for p in model.parameters() if not p.requires_grad]

    assert len(trainable) > 0, "Expected at least one trainable LoRA parameter"
    assert len(frozen) > 0, "Base SAM3 parameters should be frozen"

    # Forward + backward pass
    x = torch.randn(1, 3, 64, 64)
    mask_logits = model(x)
    loss = mask_logits.mean()
    loss.backward()

    active_grads = [p.grad for p in trainable if p.grad is not None]
    assert len(active_grads) > 0, "Expected active LoRA parameters to receive gradients"

    for p in frozen:
        assert p.grad is None, "Frozen base parameter unexpectedly received gradient"


def test_sam3_qlora_build_model_and_checkpoint():
    """Verify building SAM3-QLoRA via configuration and restoring from checkpoint."""
    cfg = ConfigDict({
        "project": {"device": "cpu"},
        "model": {
            "name": "sam3_qlora",
            "pretrained_model_name_or_path": "yujiepan/sam3-tiny-random",
            "load_in_4bit": False,
            "lora_r": 4,
            "lora_alpha": 8,
            "aux_classifier": False,
        }
    })
    model = build_model(cfg)
    assert isinstance(model, SAM3QLoRA)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "sam3_test_ckpt.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": cfg.to_dict(),
        }, ckpt_path)

        loaded_model = SAM3QLoRA.from_checkpoint(ckpt_path, device="cpu")
        assert isinstance(loaded_model, SAM3QLoRA)
        x = torch.randn(1, 3, 64, 64)
        out = loaded_model(x)
        assert out.shape == (1, 1, 64, 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required for memory fit benchmark")
def test_sam3_qlora_gpu_memory_fit():
    """
    Test if the machine's GPU (e.g. 12GB RTX 3060) has sufficient VRAM to train
    full-scale SAM3 with 4-bit QLoRA.
    Evaluates:
      1. Loading 4-bit NF4 quantized base model.
      2. Attaching LoRA adapters.
      3. Executing forward pass on 256x256 image (interpolated to 1008x1008).
      4. Executing backward pass for gradient computation.
      5. Executing AdamW optimizer step.
    """
    torch.cuda.empty_cache()
    start_free, total_vram = torch.cuda.mem_get_info()
    print(f"\nInitial GPU VRAM: Free={start_free / 1e9:.2f} GB, Total={total_vram / 1e9:.2f} GB")

    try:
        model = SAM3QLoRA(
            pretrained_model_name_or_path="jetjodh/sam3",
            load_in_4bit=True,
            lora_r=8,
            lora_alpha=16,
            lora_target_modules=["q_proj", "v_proj"],
            aux_classifier=False,
            device="cuda",
        )
        model.train()

        allocated_after_load = torch.cuda.memory_allocated() / (1024**2)
        print(f"Allocated VRAM after loading 4-bit SAM3 + LoRA: {allocated_after_load:.1f} MB")

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4,
        )

        # Forward pass on [1, 3, 256, 256]
        x = torch.randn(1, 3, 256, 256, device="cuda")
        out = model(x)
        loss = out.mean()

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        peak_allocated = torch.cuda.max_memory_allocated() / (1024**2)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)
        print(f"SAM3-QLoRA Training Step SUCCESS! Peak Allocated: {peak_allocated:.1f} MB, Peak Reserved: {peak_reserved:.1f} MB")

        assert peak_reserved <= total_vram, "Peak reserved memory exceeded total GPU capacity"

    except torch.cuda.OutOfMemoryError as oom_err:
        peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)
        print(f"\nCUDA OutOfMemory: GPU VRAM insufficient for full SAM3+QLoRA forward/backward: {oom_err}")
        print(f"Peak Reserved before OOM: {peak_reserved:.1f} MB / Total: {total_vram / 1e6:.1f} MB")
        pytest.fail(f"GPU memory insufficient to fit SAM3+QLoRA training (12GB boundary reached): {oom_err}")
