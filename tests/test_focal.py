import pytest
import torch
import torch.nn as nn
from sid_unet.models.focal import FOCAL, FOCALHRNet, FOCALViT, DEFAULT_FOCAL_HF_REPO
from sid_unet.models.unet import build_model
from sid_unet.losses import SIDTotalLoss


class TestFOCALModel:
    """Comprehensive correctness and anti-fallback test suite for FOCAL model."""

    def test_strict_anti_fallback_pretrained_false_raises_value_error(self):
        """Verifies strict zero-fallback policy: pretrained=False must raise ValueError immediately."""
        with pytest.raises(ValueError, match="strictly disabled"):
            _ = FOCAL(backbone="hrnet", pretrained=False)

        with pytest.raises(ValueError, match="strictly disabled"):
            _ = FOCAL(backbone="vit", pretrained=False)

    def test_strict_anti_fallback_missing_checkpoint_raises_runtime_error(self):
        """Verifies strict zero-fallback policy: missing/corrupt weights must raise RuntimeError immediately."""
        with pytest.raises(RuntimeError, match="Failed to download/load|Failed to resolve"):
            _ = FOCAL(
                backbone="hrnet",
                pretrained=True,
                checkpoint_path="/non/existent/path/to/weights.pth",
                hf_repo="non_existent_hf_repo_xyz_12345",
            )

    def test_unsupported_backbone_raises_value_error(self):
        """Unsupported backbone types must raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported FOCAL backbone"):
            _ = FOCAL(backbone="resnet50", pretrained=True)

    @pytest.mark.timeout(180)
    def test_focal_hrnet_genuine_loading_and_forward(self):
        """Loads genuine FOCAL HRNet weights and verifies output shapes, ranges, and types."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FOCAL(backbone="hrnet", pretrained=True).to(device)
        model.eval()

        b, c, h, w = 2, 3, 256, 256
        x = torch.randn(b, c, h, w, device=device)

        with torch.no_grad():
            out = model(x, return_dict=True)

        assert isinstance(out, dict)
        assert "mask_logits" in out
        assert "pred_mask" in out
        assert "features" in out

        # Check mask logits shape and validity
        assert out["mask_logits"].shape == (b, 1, h, w)
        assert not torch.isnan(out["mask_logits"]).any()
        assert not torch.isinf(out["mask_logits"]).any()

        # Check probability bounds
        pred_mask = out["pred_mask"]
        assert pred_mask.shape == (b, 1, h, w)
        assert (pred_mask >= 0.0).all() and (pred_mask <= 1.0).all()

        # Check feature dimension (HRNet-W32 feature channels = 32)
        assert out["features"].shape[1] == 32

        # Check binary prediction helper
        bin_mask = model.predict_mask(x, threshold=0.5)
        assert bin_mask.shape == (b, 1, h, w)
        assert torch.all((bin_mask == 0.0) | (bin_mask == 1.0))

        # Check clustering mask prediction
        cluster_mask = model.predict_mask_clustering(x)
        assert cluster_mask.shape == (b, 1, h, w)
        assert not torch.isnan(cluster_mask).any()

    @pytest.mark.timeout(180)
    def test_focal_hrnet_backward_pass(self):
        """Verifies gradient flow through FOCAL HRNet during training mode."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FOCAL(backbone="hrnet", pretrained=True).to(device)
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

        # Ensure gradients propagated to the convolutional classification head
        assert model.fc.weight.grad is not None
        assert not torch.isnan(model.fc.weight.grad).any()

    @pytest.mark.timeout(240)
    def test_focal_vit_genuine_loading_and_forward(self):
        """Loads genuine FOCAL ViT weights and verifies output shapes and clustering."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FOCAL(backbone="vit", pretrained=True).to(device)
        model.eval()

        b, c, h, w = 1, 3, 256, 256
        x = torch.randn(b, c, h, w, device=device)

        with torch.no_grad():
            out = model(x, return_dict=True)

        assert out["mask_logits"].shape == (b, 1, h, w)
        assert out["pred_mask"].shape == (b, 1, h, w)
        assert (out["pred_mask"] >= 0.0).all() and (out["pred_mask"] <= 1.0).all()
        assert out["features"].shape[1] == 256

        bin_mask = model.predict_mask(x)
        assert bin_mask.shape == (b, 1, h, w)

        cl_mask = model.predict_mask_clustering(x)
        assert cl_mask.shape == (b, 1, h, w)

    def test_focal_build_model_integration(self):
        """Verifies integration with sid_unet build_model factory."""
        cfg_hrnet = {
            "model": {
                "name": "focal",
                "backbone": "hrnet",
                "pretrained": True,
            }
        }
        model = build_model(cfg_hrnet)
        assert isinstance(model, FOCAL)
        assert model.backbone_name == "hrnet"

        # Direct shortcuts
        m_hr = FOCALHRNet(pretrained=True)
        assert isinstance(m_hr, FOCAL)
