import os

import pytest
import torch

from sid_unet.losses import SIDTotalLoss
from sid_unet.models.omnifd import OmniFD
from sid_unet.models.unet import UNet, build_model


class TestOmniFDModel:
    """Comprehensive unit and integration test suite for OmniFD."""

    @pytest.mark.parametrize("hw", [(224, 224), (256, 256), (299, 299)])
    def test_omnifd_swin_t_forward_resolutions(self, hw):
        model = OmniFD(variant="swin-t", pretrained=False, allow_random_init=True)
        model.eval()
        h, w = hw
        x = torch.randn(1, 3, h, w)
        with torch.no_grad():
            out = model(x, return_dict=True)

        assert isinstance(out, dict)
        assert out["mask_logits"].shape == (1, 1, h, w)
        assert out["pred_mask"].shape == (1, 1, h, w)
        assert out["class_logits"].shape == (1, 2)
        assert out["pred_label_probs"].shape == (1, 2)
        assert torch.all((out["pred_mask"] >= 0.0) & (out["pred_mask"] <= 1.0))
        assert torch.allclose(out["pred_label_probs"].sum(dim=-1), torch.ones(1), atol=1e-4)
        assert not torch.isnan(out["mask_logits"]).any()

    def test_omnifd_task_head_variants(self):
        x = torch.randn(1, 3, 128, 128)

        # Unified (image + spatial)
        m_both = OmniFD(
            variant="swin-t",
            pretrained=False,
            allow_random_init=True,
            has_image_head=True,
            has_spatial_head=True,
        )
        with torch.no_grad():
            mask_l, cls_l = m_both(x, return_dict=False)
        assert mask_l.shape == (1, 1, 128, 128)
        assert cls_l.shape == (1, 2)

        # Spatial-only
        m_spatial = OmniFD(
            variant="swin-t",
            pretrained=False,
            allow_random_init=True,
            has_image_head=False,
            has_spatial_head=True,
        )
        with torch.no_grad():
            only_mask = m_spatial(x, return_dict=False)
        assert torch.is_tensor(only_mask)
        assert only_mask.shape == (1, 1, 128, 128)

        # Image-only
        m_img = OmniFD(
            variant="swin-t",
            pretrained=False,
            allow_random_init=True,
            has_image_head=True,
            has_spatial_head=False,
        )
        with torch.no_grad():
            m_zero, img_cls = m_img(x, return_dict=False)
        assert m_zero.shape == (1, 1, 128, 128)
        assert img_cls.shape == (1, 2)

    def test_omnifd_backward_pass(self):
        model = OmniFD(variant="swin-t", pretrained=False, allow_random_init=True)
        model.train()
        x = torch.randn(2, 3, 128, 128, requires_grad=True)
        target_mask = torch.randint(0, 2, (2, 1, 128, 128), dtype=torch.float32)
        target_label = torch.tensor([0, 1], dtype=torch.long)

        outputs = model(x)
        loss_fn = SIDTotalLoss(mask_loss_type="combined", bce_weight=0.5, dice_weight=0.5, aux_classifier=True, aux_weight=0.2)
        total_loss, _ = loss_fn(outputs, target_mask, target_label)
        assert not torch.isnan(total_loss)
        total_loss.backward()

        assert model.head.decoder_modules["spatial"].conv_seg.weight.grad is not None
        assert model.head.decoder_modules["image"].fc.weight.grad is not None
        assert model.neck.module.latents.grad is not None

    def test_omnifd_build_model_and_from_checkpoint(self, tmp_path):
        cfg = {
            "model": {
                "name": "omnifd",
                "variant": "swin-t",
                "pretrained": False,
                "allow_random_init": True,
                "has_image_head": True,
                "has_spatial_head": True,
            }
        }
        model = build_model(cfg)
        assert isinstance(model, OmniFD)

        ckpt_path = tmp_path / "omnifd_ckpt.pt"
        torch.save({"model_state_dict": model.state_dict(), "config": cfg}, str(ckpt_path))
        loaded, loaded_cfg = UNet.from_checkpoint(str(ckpt_path), device="cpu", return_config=True)
        assert isinstance(loaded, OmniFD)
        assert loaded_cfg.model.name == "omnifd"

        # Test raw mmengine-style state_dict auto-detection
        raw_ckpt = tmp_path / "omnifd_raw.pth"
        torch.save({"state_dict": model.state_dict()}, str(raw_ckpt))
        loaded_raw, raw_cfg = UNet.from_checkpoint(str(raw_ckpt), device="cpu", return_config=True)
        assert isinstance(loaded_raw, OmniFD)
        assert raw_cfg.model.name == "omnifd"

    def test_omnifd_strict_zero_random_init_guards(self):
        with pytest.raises(ValueError, match="allow_random_init=True"):
            _ = OmniFD(variant="swin-t", pretrained=False)

        with pytest.raises(RuntimeError, match="Unable to resolve OmniFD checkpoint"):
            _ = OmniFD(
                variant="swin-t",
                checkpoint_name="nonexistent_omnifd_ckpt_xyz",
                checkpoint_path="/nonexistent/omnifd.pth",
                pretrained=True,
            )

    @pytest.mark.timeout(180)
    def test_omnifd_load_cached_gdrive_checkpoints_if_present(self):
        cases = [
            ("/tmp/omnifd_weights/omnifd_swin-t_50k_forgerynet.pth", "swin-t", True, True),
            ("/tmp/omnifd_weights/omnifd_swin-s_20k_ff_spatial.pth", "swin-s", False, True),
            ("/tmp/omnifd_weights/omnifd_swin-s_20k_ff_image.pth", "swin-s", True, False),
        ]
        found = [c for c in cases if os.path.isfile(c[0])]
        if not found:
            pytest.skip("Pretrained OmniFD checkpoints not cached locally.")
        x = torch.randn(1, 3, 224, 224)
        for cand, variant, has_img, has_spa in found:
            model = OmniFD(
                variant=variant,
                checkpoint_path=cand,
                pretrained=True,
                allow_random_init=False,
                has_image_head=has_img,
                has_spatial_head=has_spa,
            )
            model.eval()
            with torch.no_grad():
                out = model(x, return_dict=True)
            assert out["mask_logits"].shape == (1, 1, 224, 224)
            if has_img:
                assert out["class_logits"].shape == (1, 2)
                assert out["pred_label_probs"].shape == (1, 2)
            else:
                assert "class_logits" not in out
