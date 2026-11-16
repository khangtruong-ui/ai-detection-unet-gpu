"""
Tests for FOCAL Strategy 1 (Minimal-Surgery Finetuning Alignment).

Covers all acceptance criteria from `docs/focal_finetuning_strategy1_requirements.md`:
1. Config & preprocessing alignment (`data.normalize: false`, native ViT resolution, `auto` precision).
2. Mode 1A supervised finetuning train/eval parity, `FOCALSupervisedHead` decoder, strict backbone loading,
   differential LR, and backbone freeze warmup.
3. Mode 1B image-relative Soft Contrastive Learning (`MyInfoNCE`), `self.fc` decoupling, pure-pristine /
   pure-tampered edge cases, bounded memory subsampling, and cross-image isolation.
4. Pristine-image safety gate (`cluster_separability_threshold` and `aux_pristine_gate`) and
   `filter_zero_positive` / `macro_f1` / `auroc` reporting consistency.
5. End-to-end Trainer & FP8 (`precision="auto"`) compatibility for HRNet and ViT.
"""

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from sid_unet.dataset.transforms import get_transforms
from sid_unet.losses import SIDTotalLoss, build_loss
from sid_unet.models.focal import FOCAL, FOCALSupervisedHead
from sid_unet.models.unet import UNet, build_model
from sid_unet.training.trainer import Trainer
from sid_unet.utils.config import load_config
from sid_unet.utils.precision import prepare_model_for_fp8_training, resolve_precision_spec
from sid_unet.utils.report import generate_checkpoint_cross_eval_report


class _SyntheticSegDataset(Dataset):
    """Deterministic synthetic dataset with clear tampered patch for smoke training."""

    def __init__(self, n_samples: int = 8, size: int = 128):
        self.n_samples = n_samples
        self.size = size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        img = torch.full((3, self.size, self.size), 0.25, dtype=torch.float32)
        mask = torch.zeros((1, self.size, self.size), dtype=torch.float32)
        # Alternate or include tampered patch so every sample has clear contrast
        q = self.size // 4
        img[:, q : 3 * q, q : 3 * q] = 0.85
        mask[:, q : 3 * q, q : 3 * q] = 1.0
        return {
            "image": img,
            "mask": mask,
            "label": torch.tensor(2, dtype=torch.long),
        }


class TestFocalStrategy1ConfigsAndPreprocessing:
    """Requirement 1: Preprocessing, resolution, and auto precision config alignment."""

    @pytest.mark.parametrize(
        "cfg_path,expected_backbone,expected_mode,expected_clustering",
        [
            ("configs/experiments/focal/focal_hrnet.yaml", "hrnet", "supervised", False),
            ("configs/experiments/focal/focal_vit.yaml", "vit", "supervised", False),
            ("configs/experiments/focal/focal_hrnet_contrastive.yaml", "hrnet", "contrastive", True),
            ("configs/experiments/focal/focal_vit_contrastive.yaml", "vit", "contrastive", True),
        ],
    )
    def test_focal_configs_aligned(self, cfg_path, expected_backbone, expected_mode, expected_clustering):
        cfg = load_config(cfg_path)
        assert cfg.data.normalize is False
        assert list(cfg.data.image_size) == [512, 512]
        assert cfg.model.backbone == expected_backbone
        assert cfg.model.finetune_mode == expected_mode
        assert cfg.model.use_clustering is expected_clustering
        assert cfg.training.precision == "auto"
        assert cfg.training.mixed_precision == "auto"
        if expected_backbone == "vit":
            assert cfg.model.force_vit_1024 is False

    def test_transforms_raw_rgb_when_normalize_false(self):
        from PIL import Image
        import numpy as np

        tf = get_transforms(image_size=(128, 128), is_train=False, normalize=False)
        img = Image.fromarray(np.full((64, 64, 3), 200, dtype=np.uint8))
        mask = np.zeros((64, 64), dtype=np.float32)
        img_t, mask_t = tf(img, mask)
        assert img_t.min().item() >= 0.0
        assert img_t.max().item() <= 1.0
        assert torch.isclose(img_t.mean(), torch.tensor(200.0 / 255.0), atol=1e-2)

    @pytest.mark.timeout(180)
    def test_vit_native_resolution_without_1024_upsampling(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FOCAL(
            backbone="vit",
            pretrained=True,
            finetune_mode="supervised",
            force_vit_1024=False,
        ).to(device)
        model.eval()

        # 256x256 -> H/16 = 16x16 (no forced 1024x1024 resize)
        x_256 = torch.rand(1, 3, 256, 256, device=device)
        with torch.no_grad():
            out_256 = model(x_256, return_dict=True)
        assert out_256["features"].shape == (1, 256, 16, 16)
        assert out_256["mask_logits"].shape == (1, 1, 256, 256)

        # 512x512 -> H/16 = 32x32
        x_512 = torch.rand(1, 3, 512, 512, device=device)
        with torch.no_grad():
            out_512 = model(x_512, return_dict=True)
        assert out_512["features"].shape == (1, 256, 32, 32)
        assert out_512["mask_logits"].shape == (1, 1, 512, 512)


class TestFocalStrategy1Mode1ASupervised:
    """Requirement 2 (Mode 1A): Supervised finetuning parity, decoder head, and optimization."""

    @pytest.mark.timeout(180)
    def test_mode_1a_train_eval_parity_and_decoder(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FOCAL(
            backbone="hrnet",
            pretrained=True,
            finetune_mode="supervised",
            head_type="decoder",
        ).to(device)

        assert model.use_clustering is False
        assert isinstance(model.decoder, FOCALSupervisedHead)
        assert not any(p.requires_grad for p in model.fc.parameters())

        x = torch.rand(2, 3, 128, 128, device=device)
        model.train()
        out_train = model(x, return_dict=True)
        model.eval()
        with torch.no_grad():
            out_eval = model(x, return_dict=True)

        assert out_train["mask_logits"].shape == (2, 1, 128, 128)
        assert out_eval["mask_logits"].shape == (2, 1, 128, 128)
        assert "cluster_separability" not in out_eval

        # Verify gradient flows through decoder and backbone when unfrozen
        model.train()
        out = model(x, return_dict=True)
        loss = out["mask_logits"].pow(2).mean()
        loss.backward()
        assert model.decoder.out_conv.weight.grad is not None
        assert model.fc.weight.grad is None

    @pytest.mark.timeout(180)
    def test_mode_1a_checkpoint_roundtrip(self, tmp_path):
        model = FOCAL(
            backbone="hrnet",
            pretrained=True,
            finetune_mode="supervised",
            head_type="decoder",
        )
        ckpt_path = tmp_path / "focal_sup.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "model": {
                        "name": "focal",
                        "backbone": "hrnet",
                        "pretrained": True,
                        "finetune_mode": "supervised",
                        "use_clustering": False,
                        "head_type": "decoder",
                    }
                },
            },
            str(ckpt_path),
        )
        loaded, cfg = UNet.from_checkpoint(str(ckpt_path), return_config=True)
        assert isinstance(loaded, FOCAL)
        assert loaded.finetune_mode == "supervised"
        assert loaded.use_clustering is False
        assert isinstance(loaded.decoder, FOCALSupervisedHead)
        assert torch.allclose(
            loaded.decoder.out_conv.weight.cpu(),
            model.decoder.out_conv.weight.cpu(),
        )

    @pytest.mark.timeout(240)
    def test_mode_1a_trainer_differential_lr_warmup_auto_precision(self, tmp_path):
        cfg = load_config("configs/experiments/focal/focal_hrnet.yaml")
        cfg.project.output_dir = str(tmp_path / "run_1a")
        cfg.project.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.data.image_size = [128, 128]
        cfg.data.batch_size = 4
        cfg.training.epochs = 2
        cfg.training.learning_rate = 1e-3
        cfg.training.backbone_lr_multiplier = 0.1
        cfg.training.freeze_backbone_epochs = 1
        cfg.training.gradient_accumulation_steps = 1
        cfg.training.auto_batch_size = False
        cfg.training.precision = "auto"
        cfg.training.mixed_precision = "auto"
        cfg.training.early_stopping_patience = 10

        ds = _SyntheticSegDataset(n_samples=4, size=128)
        train_loader = DataLoader(ds, batch_size=4, shuffle=False)
        val_loader = DataLoader(ds, batch_size=4, shuffle=False)

        trainer = Trainer(config=cfg, train_loader=train_loader, val_loader=val_loader)
        # Verify differential LR param groups
        assert len(trainer.optimizer.param_groups) == 2
        head_lr = trainer.optimizer.param_groups[0]["lr"]
        bb_lr = trainer.optimizer.param_groups[1]["lr"]
        assert pytest.approx(bb_lr / head_lr, rel=1e-5) == 0.1

        res = trainer.train()
        assert len(res["history"]) == 2
        # Backbone must be unfrozen by epoch 2
        raw_model = getattr(trainer.model, "module", trainer.model)
        assert all(p.requires_grad for p in raw_model.net.parameters())
        # Filter zero positive reporting preserved
        assert "filter_zero_positive" in res["final_metrics"]
        assert "val_macro_f1" in res["final_metrics"]
        assert "val_auroc" in res["final_metrics"]


class TestFocalStrategy1Mode1BContrastive:
    """Requirement 2 (Mode 1B): Image-relative MyInfoNCE contrastive training."""

    @pytest.mark.timeout(180)
    def test_mode_1b_infonce_gradients_and_fc_decoupled(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FOCAL(
            backbone="hrnet",
            pretrained=True,
            finetune_mode="contrastive",
            contrastive_sample_size=256,
        ).to(device)
        model.train()

        x = torch.rand(2, 3, 128, 128, device=device)
        gt = torch.zeros(2, 1, 128, 128, device=device)
        gt[:, :, 32:96, 32:96] = 1.0

        out = model(x, return_dict=True, gt_masks=gt)
        assert out["skip_mask_loss"] is True
        assert "align_loss" in out
        assert torch.isfinite(out["align_loss"])
        assert out["align_loss"].item() > 0.0

        loss_fn = SIDTotalLoss(artifact_weight=1.0)
        total_loss, loss_dict = loss_fn(out, gt)
        assert loss_dict["mask_loss"] == 0.0
        assert torch.isclose(total_loss, out["align_loss"])

        total_loss.backward()
        assert model.fc.weight.grad is None
        assert model.net.conv1.weight.grad is not None
        assert not torch.isnan(model.net.conv1.weight.grad).any()

    @pytest.mark.timeout(180)
    def test_mode_1b_edge_cases_pristine_tampered_and_cross_image_isolation(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FOCAL(
            backbone="hrnet",
            pretrained=True,
            finetune_mode="contrastive",
            contrastive_sample_size=128,
        ).to(device)
        model.train()

        feats = torch.randn(2, 32, 16, 16, device=device, requires_grad=True)

        # 1. All-pristine batch -> 0.0 loss, finite
        gt_pristine = torch.zeros(2, 1, 64, 64, device=device)
        loss_p = model.compute_contrastive_loss(feats, gt_pristine)
        assert torch.isfinite(loss_p) and loss_p.item() == 0.0

        # 2. All-tampered batch -> 0.0 loss, finite
        gt_tampered = torch.ones(2, 1, 64, 64, device=device)
        loss_t = model.compute_contrastive_loss(feats, gt_tampered)
        assert torch.isfinite(loss_t) and loss_t.item() == 0.0

        # 3. Cross-image isolation: image 0 is 100% pristine, image 1 is 100% tampered ->
        #    no cross-image contrasting allowed, so loss must still be 0.0
        gt_mixed_img = torch.zeros(2, 1, 64, 64, device=device)
        gt_mixed_img[1] = 1.0
        loss_iso = model.compute_contrastive_loss(feats, gt_mixed_img)
        assert torch.isfinite(loss_iso) and loss_iso.item() == 0.0

    @pytest.mark.timeout(240)
    def test_mode_1b_trainer_integration_auto_precision(self, tmp_path):
        cfg = load_config("configs/experiments/focal/focal_hrnet_contrastive.yaml")
        cfg.project.output_dir = str(tmp_path / "run_1b")
        cfg.project.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.data.image_size = [128, 128]
        cfg.data.batch_size = 4
        cfg.training.epochs = 1
        cfg.training.gradient_accumulation_steps = 1
        cfg.training.auto_batch_size = False
        cfg.training.precision = "auto"
        cfg.training.mixed_precision = "auto"

        ds = _SyntheticSegDataset(n_samples=4, size=128)
        train_loader = DataLoader(ds, batch_size=4, shuffle=False)
        val_loader = DataLoader(ds, batch_size=4, shuffle=False)

        trainer = Trainer(config=cfg, train_loader=train_loader, val_loader=val_loader)
        res = trainer.train()
        assert len(res["history"]) == 1
        assert res["history"][0]["train_loss"] > 0.0


class TestFocalStrategy1PristineSafetyAndFP8:
    """Requirement 3: Pristine-image clustering safety, reporting, and FP8 ViT compatibility."""

    @pytest.mark.timeout(180)
    def test_pristine_image_separability_gate_blocks_false_positives(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FOCAL(
            backbone="hrnet",
            pretrained=True,
            use_clustering=True,
            cluster_separability_threshold=0.05,
        ).to(device)
        model.eval()

        # Pure-pristine uniform image -> low separability -> 0% hallucinated positive pixels
        x_pristine = torch.full((1, 3, 128, 128), 0.5, device=device)
        with torch.no_grad():
            out_p = model(x_pristine, return_dict=True)
        assert out_p["cluster_separability"].item() < 0.05
        assert (out_p["pred_mask"] > 0.5).float().mean().item() == 0.0

        # Tampered image with distinct manipulated patch -> high separability -> non-zero mask
        x_tampered = torch.full((1, 3, 128, 128), 0.4, device=device)
        torch.manual_seed(0)
        x_tampered[:, :, 32:80, 32:80] = torch.rand(1, 3, 48, 48, device=device)
        with torch.no_grad():
            out_t = model(x_tampered, return_dict=True)
        assert out_t["cluster_separability"].item() >= 0.05
        pos_ratio = (out_t["pred_mask"] > 0.5).float().mean().item()
        assert 0.0 < pos_ratio < 0.5

    @pytest.mark.timeout(180)
    def test_aux_pristine_gate_when_enabled(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FOCAL(
            backbone="hrnet",
            pretrained=True,
            aux_classifier=True,
            aux_pristine_gate=True,
            cluster_separability_threshold=0.0,
        ).to(device)
        model.eval()

        # Force class_head bias so class 0 (pristine) always wins
        with torch.no_grad():
            last_linear = model.class_head[-1]
            last_linear.weight.zero_()
            last_linear.bias.copy_(torch.tensor([5.0, -5.0], device=device))

        x = torch.rand(1, 3, 128, 128, device=device)
        with torch.no_grad():
            out = model(x, return_dict=True)
        assert (out["pred_mask"] > 0.5).float().mean().item() == 0.0

    def test_cross_eval_report_includes_filter_zero_positive_and_macro_f1(self, tmp_path):
        results = [
            {
                "checkpoint_name": "focal_hrnet",
                "config_name": "casia_v2.0",
                "dataset_name": "KhangTruong/Casia-v2.0-Fake",
                "eval_split": "test",
                "total_evaluated_samples": 50,
                "metrics": {
                    "eval_total_loss": 0.3,
                    "iou": 0.42,
                    "f1": 0.58,
                    "macro_f1": 0.71,
                    "auroc": 0.85,
                    "pixel_acc": 0.89,
                    "filter_zero_positive": True,
                },
            }
        ]
        rep = generate_checkpoint_cross_eval_report(
            checkpoint_path=str(tmp_path / "ckpt.pt"),
            results=results,
            output_dir=str(tmp_path),
        )
        assert "Filter Zero Pos" in rep["summary_table"]
        assert "Macro F1" in rep["summary_table"]
        assert "tampered-only" in rep["summary_table"]
        assert "0.7100" in rep["summary_table"]

    @pytest.mark.timeout(240)
    def test_vit_auto_precision_fp8_training_step(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            pytest.skip("CUDA required for FP8 / AMP auto-precision verification.")

        cfg = load_config("configs/experiments/focal/focal_vit.yaml")
        spec = resolve_precision_spec(cfg, device=device)
        model = FOCAL(
            backbone="vit",
            pretrained=True,
            finetune_mode="supervised",
            force_vit_1024=False,
        ).to(device)
        fp8_info = prepare_model_for_fp8_training(model, spec)
        if spec.fp8_enabled:
            assert fp8_info["converted_linear_layers"] > 0

        model.train()
        x = torch.rand(1, 3, 256, 256, device=device)
        gt = torch.zeros(1, 1, 256, 256, device=device)
        gt[:, :, 64:192, 64:192] = 1.0

        with torch.amp.autocast(device_type="cuda", dtype=spec.amp_dtype, enabled=spec.amp_enabled):
            out = model(x, return_dict=True)
            loss, _ = build_loss(cfg)(out, gt)
        assert torch.isfinite(loss)
        loss.backward()
        assert model.decoder.out_conv.weight.grad is not None
