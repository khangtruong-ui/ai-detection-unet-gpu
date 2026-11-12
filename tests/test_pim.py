import pytest
import torch
import torch.nn as nn
from sid_unet.models.pim import PIM, DEFAULT_PIM_HF_REPO
from sid_unet.models.unet import build_model
from sid_unet.losses import SIDTotalLoss


class TestPIMModel:
    """Comprehensive correctness and anti-fallback test suite for PIM model."""

    def test_strict_anti_fallback_pretrained_false_raises_value_error(self):
        """Verifies strict zero-fallback policy: pretrained=False must raise ValueError immediately."""
        with pytest.raises(ValueError, match="strictly disabled"):
            _ = PIM(pretrained=False)

    def test_strict_anti_fallback_missing_checkpoint_raises_runtime_error(self):
        """Verifies strict zero-fallback policy: missing/corrupt weights must raise RuntimeError immediately."""
        with pytest.raises(RuntimeError, match="Failed to download/load|Failed to resolve"):
            _ = PIM(
                pretrained=True,
                checkpoint_path="/non/existent/path/to/pim_ckpt.pth",
                hf_repo="non_existent_hf_repo_xyz_98765",
            )

    @pytest.mark.timeout(180)
    def test_pim_genuine_loading_and_forward(self):
        """Loads genuine PIM weights from Hugging Face KhangTruong/PIM and verifies all output heads."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PIM(pretrained=True).to(device)
        model.eval()

        b, c, h, w = 2, 3, 256, 256
        x = torch.randn(b, c, h, w, device=device)

        with torch.no_grad():
            out = model(x, return_dict=True)

        assert isinstance(out, dict)
        assert "mask_logits" in out
        assert "pred_mask" in out
        assert "boundary_logits" in out
        assert "recon_img" in out
        assert "raw_seg" in out

        # Check mask logits
        assert out["mask_logits"].shape == (b, 1, h, w)
        assert not torch.isnan(out["mask_logits"]).any()
        assert not torch.isinf(out["mask_logits"]).any()

        # Check probability bounds
        pred_mask = out["pred_mask"]
        assert pred_mask.shape == (b, 1, h, w)
        assert (pred_mask >= 0.0).all() and (pred_mask <= 1.0).all()

        # Check boundary logits shape
        assert out["boundary_logits"].shape == (b, 1, h, w)
        assert not torch.isnan(out["boundary_logits"]).any()

        # Check image reconstruction shape
        assert out["recon_img"].shape == (b, 3, h, w)
        assert not torch.isnan(out["recon_img"]).any()

        # Check binary prediction helper
        bin_mask = model.predict_mask(x, threshold=0.5)
        assert bin_mask.shape == (b, 1, h, w)
        assert torch.all((bin_mask == 0.0) | (bin_mask == 1.0))

    @pytest.mark.timeout(180)
    def test_pim_return_tuple_forward(self):
        """Verifies forward pass with return_dict=False returns 3 output tensors."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PIM(pretrained=True).to(device)
        model.eval()

        b, c, h, w = 1, 3, 128, 128
        x = torch.randn(b, c, h, w, device=device)

        with torch.no_grad():
            mask_logits, boundary_logits, recon_img = model(x, return_dict=False)

        assert mask_logits.shape == (b, 1, h, w)
        assert boundary_logits.shape == (b, 1, h, w)
        assert recon_img.shape == (b, 3, h, w)

    @pytest.mark.timeout(180)
    def test_pim_backward_pass(self):
        """Verifies gradient flow through PIM during training mode."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PIM(pretrained=True).to(device)
        model.train()

        b, c, h, w = 2, 3, 128, 128
        x = torch.randn(b, c, h, w, device=device, requires_grad=True)
        target_mask = torch.randint(0, 2, (b, 1, h, w), dtype=torch.float32, device=device)

        out = model(x, return_dict=True)
        logits = out["mask_logits"]

        loss_fn = SIDTotalLoss(mask_loss_type="combined", bce_weight=0.5, dice_weight=0.5)
        total_loss, loss_dict = loss_fn(logits, target_mask)

        assert not torch.isnan(total_loss)
        total_loss.backward()

        # Ensure gradients propagated to decode_head conv_seg
        assert model.net.decode_head.conv_seg.weight.grad is not None
        assert not torch.isnan(model.net.decode_head.conv_seg.weight.grad).any()

    def test_pim_build_model_integration(self):
        """Verifies integration with sid_unet build_model factory."""
        cfg_pim = {
            "model": {
                "name": "pim",
                "pretrained": True,
            }
        }
        model = build_model(cfg_pim)
        assert isinstance(model, PIM)

    def test_pim_from_checkpoint_and_reformatted_loading(self, tmp_path):
        """Verifies UNet.from_checkpoint loads PIM correctly with embedded config and model_state_dict."""
        from sid_unet.models.unet import UNet
        model = PIM(pretrained=True)
        ckpt_path = tmp_path / "checkpoint_best.pt"
        payload = {
            "epoch": 5,
            "step": 500,
            "model_state_dict": model.state_dict(),
            "config": {
                "model": {
                    "name": "pim",
                    "pretrained": True,
                }
            },
            "best_score": 0.85,
        }
        torch.save(payload, str(ckpt_path))

        loaded_model, ckpt_cfg = UNet.from_checkpoint(str(ckpt_path), return_config=True)
        assert isinstance(loaded_model, PIM)
        assert ckpt_cfg.model.name == "pim"
