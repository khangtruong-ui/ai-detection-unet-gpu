import pytest
import torch

from sid_unet.losses import SIDTotalLoss
from sid_unet.models.msccnet import MSCCNet
from sid_unet.models.unet import UNet, build_model


class TestMSCCNetModel:
    """Comprehensive unit and integration test suite for MSCCNet."""

    @pytest.mark.parametrize("hw", [(256, 256), (512, 512), (256, 384)])
    def test_msccnet_forward_resolutions_and_dict_output(self, hw):
        model = MSCCNet(
            backbone_depth=50,
            pretrained=False,
            backbone_pretrained=False,
            num_classes=2,
            spectral_k=2,
        )
        model.eval()
        h, w = hw
        x = torch.randn(2, 3, h, w)
        with torch.no_grad():
            out = model(x, return_dict=True)

        assert isinstance(out, dict)
        assert out["mask_logits"].shape == (2, 1, h, w)
        assert out["pred_mask"].shape == (2, 1, h, w)
        assert out["class_logits"].shape == (2, 2)
        assert out["pred_label_probs"].shape == (2, 2)
        assert out["fres_map_logits"].shape == (2, 1, h, w)
        assert out["fres_map_prob"].shape == (2, 1, h, w)
        assert torch.all((out["pred_mask"] >= 0.0) & (out["pred_mask"] <= 1.0))
        assert torch.allclose(out["pred_label_probs"].sum(dim=-1), torch.ones(2), atol=1e-4)
        assert not torch.isnan(out["mask_logits"]).any()

    def test_msccnet_tuple_and_mask_only_forward(self):
        model_aux = MSCCNet(backbone_depth=18, pretrained=False, aux_classifier=True, feats_channels=256, out_channels=256)
        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            mask_logits, cls_logits = model_aux(x, return_dict=False)
        assert mask_logits.shape == (1, 1, 128, 128)
        assert cls_logits.shape == (1, 2)

        model_no_aux = MSCCNet(backbone_depth=18, pretrained=False, aux_classifier=False, feats_channels=256, out_channels=256)
        with torch.no_grad():
            only_mask = model_no_aux(x, return_dict=False)
        assert torch.is_tensor(only_mask)
        assert only_mask.shape == (1, 1, 128, 128)

        bin_mask = model_aux.predict_mask(x, threshold=0.5)
        assert bin_mask.shape == (1, 1, 128, 128)
        assert torch.all((bin_mask == 0.0) | (bin_mask == 1.0))

    def test_msccnet_backward_pass(self):
        model = MSCCNet(backbone_depth=18, pretrained=False, feats_channels=256, out_channels=256)
        model.train()
        x = torch.randn(2, 3, 128, 128, requires_grad=True)
        target_mask = torch.randint(0, 2, (2, 1, 128, 128), dtype=torch.float32)
        target_label = torch.tensor([0, 1], dtype=torch.long)

        outputs = model(x)
        loss_fn = SIDTotalLoss(mask_loss_type="combined", bce_weight=0.5, dice_weight=0.5, aux_classifier=True, aux_weight=0.2)
        total_loss, _ = loss_fn(outputs, target_mask, target_label)
        assert not torch.isnan(total_loss)
        total_loss.backward()

        assert model.decoder[-1].weight.grad is not None
        assert model.classifer.fc.weight.grad is not None
        assert model.mscc.gcn.conv2.weight.grad is not None

    def test_msccnet_build_model_and_from_checkpoint(self, tmp_path):
        cfg = {
            "model": {
                "name": "msccnet",
                "backbone_depth": 50,
                "pretrained": False,
                "backbone_pretrained": False,
                "spectral_k": 2,
                "num_classes": 2,
            }
        }
        model = build_model(cfg)
        assert isinstance(model, MSCCNet)

        ckpt_path = tmp_path / "msccnet_ckpt.pt"
        torch.save({"model_state_dict": model.state_dict(), "config": cfg}, str(ckpt_path))
        loaded, loaded_cfg = UNet.from_checkpoint(str(ckpt_path), device="cpu", return_config=True)
        assert isinstance(loaded, MSCCNet)
        assert loaded_cfg.model.name == "msccnet"

        # Also test raw state_dict auto-detection without embedded config
        raw_ckpt = tmp_path / "msccnet_raw.pth"
        torch.save(model.state_dict(), str(raw_ckpt))
        loaded_raw, raw_cfg = UNet.from_checkpoint(str(raw_ckpt), device="cpu", return_config=True)
        assert isinstance(loaded_raw, MSCCNet)
        assert raw_cfg.model.name == "msccnet"

    def test_msccnet_missing_checkpoint_raises(self):
        with pytest.raises(FileNotFoundError):
            _ = MSCCNet(checkpoint_path="/nonexistent/msccnet_best.pth")
