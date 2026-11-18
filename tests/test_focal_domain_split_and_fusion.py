"""
Tests for:
1. FOCAL Domain-Split 4-Class / 4-Cluster Finetuning on KhangTruong/IMD2020-CASIAv2-Joint
   (0: casia_pristine, 1: casia_tampered, 2: imd_pristine, 3: imd_tampered)
   with 2-class (pristine vs tampered) inference collapse.
2. FOCAL Independent Dual-Backbone Fusion (frozen primary backbone + trainable secondary backbone).
"""

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from sid_unet.dataset.loader import process_raw_sample
from sid_unet.losses import SIDTotalLoss, build_loss
from sid_unet.models.focal import (
    FOCAL,
    DualBackboneFusionModule,
    _collapse_domain_4_to_2_mask_logits,
    encode_domain_sources,
)
from sid_unet.models.unet import UNet, build_model
from sid_unet.training.trainer import Trainer
from sid_unet.utils.config import load_config


class _SyntheticJointDomainDataset(Dataset):
    """Synthetic joint CASIA + IMD2020 dataset with pristine and tampered samples."""

    def __init__(self, n_samples: int = 8, size: int = 128):
        self.n_samples = n_samples
        self.size = size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        # 0: casia pristine, 1: casia tampered, 2: imd2020 pristine, 3: imd2020 tampered
        kind = idx % 4
        domain = "casia" if kind < 2 else "imd2020"
        is_tampered = (kind % 2) == 1

        base_val = 0.25 if domain == "casia" else 0.45
        img = torch.full((3, self.size, self.size), base_val, dtype=torch.float32)
        mask = torch.zeros((1, self.size, self.size), dtype=torch.float32)
        if is_tampered:
            q = self.size // 4
            img[:, q : 3 * q, q : 3 * q] = 0.9 if domain == "casia" else 0.1
            mask[:, q : 3 * q, q : 3 * q] = 1.0
        return {
            "image": img,
            "mask": mask,
            "label": torch.tensor(2 if is_tampered else 0, dtype=torch.long),
            "dataset": domain,
        }


class TestDomainSourcePipeline:
    """Verify dataset_source column extraction and integer domain encoding."""

    def test_process_raw_sample_reads_dataset_source_column(self):
        from PIL import Image
        import numpy as np

        img = Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8))
        mask = Image.fromarray(np.zeros((64, 64), dtype=np.uint8))
        out_casia = process_raw_sample(
            {"image": img, "mask": mask, "dataset_source": "casia"},
            transform=None,
            target_image_size=(64, 64),
        )
        out_imd = process_raw_sample(
            {"image": img, "mask": mask, "dataset_source": "imd2020"},
            transform=None,
            target_image_size=(64, 64),
        )
        assert out_casia["dataset"] == "casia"
        assert out_imd["dataset"] == "imd2020"

    def test_encode_domain_sources_mapping(self):
        ids = encode_domain_sources(["casia", "imd2020", "CASIAv2", "IMD", "other", ""])
        assert ids.tolist() == [0, 1, 0, 1, -1, -1]


class TestFOCALDomainSplit4ClassAnd2ClassInference:
    """Task 1: 4-class/4-cluster training with 2-class (pristine vs tampered) inference."""

    def test_logsumexp_4_to_2_collapse_exact_probability_identity(self):
        logits_4 = torch.randn(2, 4, 16, 16)
        mask_logits, logits_2 = _collapse_domain_4_to_2_mask_logits(logits_4)
        assert mask_logits.shape == (2, 1, 16, 16)
        assert logits_2.shape == (2, 2, 16, 16)

        probs_4 = torch.softmax(logits_4, dim=1)
        expected_tampered_prob = probs_4[:, 1:2] + probs_4[:, 3:4]
        assert torch.allclose(torch.sigmoid(mask_logits), expected_tampered_prob, atol=1e-5)

    @pytest.mark.timeout(180)
    def test_supervised_domain_split_train_and_2class_inference(self, tmp_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FOCAL(
            backbone="hrnet",
            pretrained=True,
            finetune_mode="supervised",
            head_type="decoder",
            domain_split_mode=True,
            aux_classifier=True,
            peft_enabled=True,
            hrnet_trainable_stages=["stage4"],
            freeze_backbone_bn=True,
        ).to(device)
        assert model.decoder.out_conv.out_channels == 4
        assert model.requires_gt_masks is True
        assert model.requires_domain_info is True

        # Training pass with 4-class domain split targets
        model.train()
        x = torch.rand(4, 3, 128, 128, device=device)
        gt = torch.zeros(4, 1, 128, 128, device=device)
        gt[1, :, 32:96, 32:96] = 1.0  # casia tampered
        gt[3, :, 32:96, 32:96] = 1.0  # imd tampered
        out_train = model(
            x,
            return_dict=True,
            gt_masks=gt,
            dataset_sources=["casia", "casia", "imd2020", "imd2020"],
        )
        assert out_train["mask_logits"].shape == (4, 1, 128, 128)
        assert out_train["class_logits"].shape == (4, 2)
        assert out_train["skip_mask_loss"] is False
        assert "align_loss" in out_train and torch.isfinite(out_train["align_loss"])

        loss_fn = SIDTotalLoss(aux_classifier=True, artifact_weight=1.0).to(device)
        labels = torch.tensor([0, 2, 0, 2], dtype=torch.long, device=device)
        loss, _ = loss_fn(out_train, gt, labels)
        loss.backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert missing == []

        # Inference pass WITHOUT any domain labels -> strictly 2 classes (pristine vs tampered)
        model.eval()
        with torch.no_grad():
            out_eval = model(x, return_dict=True)
        assert out_eval["mask_logits"].shape == (4, 1, 128, 128)
        assert out_eval["raw_logits"].shape == (4, 2, 128, 128)
        assert out_eval["pred_mask"].shape == (4, 1, 128, 128)
        assert out_eval["class_logits"].shape == (4, 2)
        assert torch.all(out_eval["pred_label"] <= 1)

        # End-to-end Trainer integration with mixed casia + imd2020 dataset
        cfg = load_config("configs/experiments/focal/focal_hrnet_domain_split.yaml")
        cfg.project.output_dir = str(tmp_path / "run_sup_dom")
        cfg.project.device = str(device)
        cfg.data.image_size = [128, 128]
        cfg.data.batch_size = 4
        cfg.training.epochs = 2
        cfg.training.freeze_backbone_epochs = 1
        cfg.training.auto_batch_size = False
        cfg.training.early_stopping_patience = 10

        ds = _SyntheticJointDomainDataset(n_samples=4, size=128)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        trainer = Trainer(config=cfg, train_loader=loader, val_loader=loader)
        res = trainer.train()
        assert len(res["history"]) == 2

    @pytest.mark.timeout(180)
    def test_contrastive_domain_split_4clusters_and_2class_inference(self, tmp_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FOCAL(
            backbone="hrnet",
            pretrained=True,
            finetune_mode="contrastive",
            domain_split_mode=True,
            contrastive_sample_size=256,
            peft_enabled=True,
            hrnet_trainable_stages=["stage4"],
            freeze_backbone_bn=True,
        ).to(device)
        assert model.domain_prototypes is not None
        assert model.domain_prototypes.shape == (4, 32)
        assert not model.domain_prototypes_initialized.any()

        ds = _SyntheticJointDomainDataset(n_samples=4, size=128)
        batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)
        dom_ids = encode_domain_sources(batch["dataset"], device=device)

        model.train()
        opt = torch.optim.AdamW(model.get_param_groups(lr=5e-4))
        loss_fn = SIDTotalLoss(aux_classifier=False, artifact_weight=1.0).to(device)

        for _ in range(3):
            opt.zero_grad()
            out = model(imgs, return_dict=True, gt_masks=masks, domain_ids=dom_ids)
            assert out["skip_mask_loss"] is True
            loss, _ = loss_fn(out, masks)
            assert torch.isfinite(loss) and loss.item() > 0.0
            loss.backward()
            missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
            assert missing == []
            opt.step()

        # All 4 domain prototypes initialized from CASIA/IMD pristine/tampered batches
        assert model.domain_prototypes_initialized.all()

        # Edge cases: domain_ids=None and pure-pristine batch must not produce NaN/Inf
        pure_pristine = torch.zeros_like(masks)
        loss_no_dom = model.compute_domain_contrastive_loss(
            torch.randn(4, 32, 32, 32, device=device, requires_grad=True), pure_pristine, None
        )
        assert torch.isfinite(loss_no_dom)

        # Inference WITHOUT domain_ids -> strictly 2 classes (pristine vs tampered)
        model.eval()
        with torch.no_grad():
            out_eval = model(imgs, return_dict=True)
        assert out_eval["mask_logits"].shape == (4, 1, 128, 128)
        assert out_eval["raw_logits"].shape == (4, 2, 128, 128)
        assert out_eval["pred_mask"].shape == (4, 1, 128, 128)
        assert "cluster_separability" in out_eval

        # Checkpoint round-trip preserves learned domain_prototypes and initialized mask
        ckpt_path = tmp_path / "focal_domain_contrastive.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "model": {
                        "name": "focal",
                        "backbone": "hrnet",
                        "pretrained": True,
                        "finetune_mode": "contrastive",
                        "domain_split_mode": True,
                        "peft_enabled": True,
                        "hrnet_trainable_stages": ["stage4"],
                        "freeze_backbone_bn": True,
                    }
                },
            },
            str(ckpt_path),
        )
        reloaded = UNet.from_checkpoint(str(ckpt_path), device=device)
        assert reloaded.domain_prototypes_initialized.all()
        assert torch.allclose(reloaded.domain_prototypes, model.domain_prototypes)

    @pytest.mark.parametrize(
        "cfg_path,expected_backbone,expected_mode,expected_clustering",
        [
            ("configs/experiments/focal/focal_hrnet_domain_split.yaml", "hrnet", "supervised", False),
            (
                "configs/experiments/focal/focal_hrnet_contrastive_domain_split.yaml",
                "hrnet",
                "contrastive",
                True,
            ),
            ("configs/experiments/focal/focal_vit_domain_split.yaml", "vit", "supervised", False),
            (
                "configs/experiments/focal/focal_vit_contrastive_domain_split.yaml",
                "vit",
                "contrastive",
                True,
            ),
        ],
    )
    def test_task1_domain_split_yaml_configs(
        self, cfg_path, expected_backbone, expected_mode, expected_clustering
    ):
        cfg = load_config(cfg_path)
        assert cfg.data.dataset_name == "KhangTruong/IMD2020-CASIAv2-Joint"
        assert cfg.data.normalize is False
        assert list(cfg.data.image_size) == [512, 512]
        assert cfg.model.backbone == expected_backbone
        assert cfg.model.finetune_mode == expected_mode
        assert cfg.model.use_clustering is expected_clustering
        assert cfg.model.domain_split_mode is True
        assert cfg.model.num_classes == 2


class TestFOCALDualBackboneFusion:
    """Task 2: Independent dual-backbone fusion alongside frozen primary FOCAL backbone."""

    def test_zero_gated_fusion_is_exact_identity_at_step0(self):
        fusion = DualBackboneFusionModule(primary_dim=32, secondary_dim=256, fusion_mode="zero_gated")
        f1 = torch.randn(2, 32, 32, 32)
        f2 = torch.randn(2, 256, 32, 32)
        assert torch.equal(fusion(f1, f2), f1)

    @pytest.mark.timeout(240)
    def test_frozen_hrnet_primary_plus_peft_vit_secondary_zero_gated(self, tmp_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FOCAL(
            backbone="hrnet",
            secondary_backbone="vit",
            freeze_primary_backbone=True,
            secondary_trainable=True,
            secondary_peft_enabled=True,
            fusion_mode="zero_gated",
            fusion_spatial_align="primary",
            pretrained=True,
            finetune_mode="supervised",
            head_type="decoder",
            force_vit_1024=False,
            lora_r=4,
            lora_alpha=8.0,
            lora_block_scope=[23],
            lora_target_modules=["qkv", "proj"],
        ).to(device)

        assert all(not p.requires_grad for p in model.net.parameters())
        assert any(p.requires_grad for p in model.secondary_net.parameters())
        assert model.in_dim == 32

        # Freeze/unfreeze warmup transition MUST NOT unfreeze frozen primary backbone
        model.set_backbone_trainable(False)
        assert all(not p.requires_grad for p in model.net.parameters())
        assert all(not p.requires_grad for p in model.secondary_net.parameters())
        model.set_backbone_trainable(True)
        assert all(not p.requires_grad for p in model.net.parameters())
        assert any(p.requires_grad for p in model.secondary_net.parameters())
        assert not model.net.training

        conv1_snap = model.net.conv1.weight.detach().clone()
        bn1_mean_snap = model.net.bn1.running_mean.detach().clone()

        model.train()
        opt = torch.optim.AdamW(model.get_param_groups(lr=1e-4))
        loss_fn = SIDTotalLoss(aux_classifier=False).to(device)

        x = torch.rand(1, 3, 256, 256, device=device)
        gt = torch.zeros(1, 1, 256, 256, device=device)
        gt[:, :, 64:192, 64:192] = 1.0

        opt.zero_grad()
        out = model(x, return_dict=True)
        assert out["features"].shape == (1, 32, 64, 64)
        loss, _ = loss_fn(out, gt)
        assert torch.isfinite(loss)
        loss.backward()

        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert missing == []
        opt.step()

        # Primary frozen HRNet weights and BN running stats stay bit-identical
        assert torch.equal(model.net.conv1.weight, conv1_snap)
        assert torch.equal(model.net.bn1.running_mean, bn1_mean_snap)

        # Checkpoint round-trip via UNet.from_checkpoint
        ckpt_path = tmp_path / "focal_dual_hrnet_vit.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "model": {
                        "name": "focal",
                        "backbone": "hrnet",
                        "secondary_backbone": "vit",
                        "freeze_primary_backbone": True,
                        "secondary_trainable": True,
                        "secondary_peft_enabled": True,
                        "fusion_mode": "zero_gated",
                        "fusion_spatial_align": "primary",
                        "pretrained": True,
                        "finetune_mode": "supervised",
                        "head_type": "decoder",
                        "force_vit_1024": False,
                        "lora_r": 4,
                        "lora_alpha": 8.0,
                        "lora_block_scope": [23],
                        "lora_target_modules": ["qkv", "proj"],
                    }
                },
            },
            str(ckpt_path),
        )
        reloaded = UNet.from_checkpoint(str(ckpt_path), device=device)
        reloaded.eval()
        model.eval()
        with torch.no_grad():
            assert torch.allclose(
                reloaded(x, return_dict=True)["mask_logits"],
                model(x, return_dict=True)["mask_logits"],
                atol=1e-4,
            )

    @pytest.mark.timeout(240)
    def test_frozen_vit_primary_plus_hrnet_secondary_concat_contrastive(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FOCAL(
            backbone="vit",
            secondary_backbone="hrnet",
            freeze_primary_backbone=True,
            secondary_trainable=True,
            secondary_trainable_stages=["stage4"],
            freeze_backbone_bn=True,
            fusion_mode="concat",
            fusion_spatial_align="max",
            pretrained=True,
            finetune_mode="contrastive",
            force_vit_1024=False,
            contrastive_sample_size=256,
        ).to(device)

        assert model.in_dim == 288
        assert all(not p.requires_grad for p in model.net.parameters())
        assert any(p.requires_grad for p in model.secondary_net.stage4.parameters())
        assert all(not p.requires_grad for p in model.secondary_net.stage3.parameters())

        vit_snap = model.net.image_encoder.neck[0].weight.detach().clone()
        model.train()
        opt = torch.optim.AdamW(model.get_param_groups(lr=1e-4))
        loss_fn = SIDTotalLoss(aux_classifier=False, artifact_weight=1.0).to(device)

        x = torch.rand(1, 3, 256, 256, device=device)
        gt = torch.zeros(1, 1, 256, 256, device=device)
        gt[:, :, 64:192, 64:192] = 1.0

        opt.zero_grad()
        out = model(x, return_dict=True, gt_masks=gt)
        assert out["features"].shape == (1, 288, 64, 64)
        loss, _ = loss_fn(out, gt)
        assert torch.isfinite(loss)
        loss.backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert missing == []
        opt.step()

        assert torch.equal(model.net.image_encoder.neck[0].weight, vit_snap)

    @pytest.mark.timeout(180)
    def test_frozen_hrnet_primary_plus_timm_secondary_backbone(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FOCAL(
            backbone="hrnet",
            secondary_backbone="timm:resnet18",
            freeze_primary_backbone=True,
            secondary_out_dim=64,
            fusion_mode="proj_concat",
            fusion_out_dim=64,
            pretrained=True,
            finetune_mode="supervised",
        ).to(device)

        assert all(not p.requires_grad for p in model.net.parameters())
        assert model.in_dim == 64

        model.train()
        x = torch.rand(2, 3, 128, 128, device=device)
        gt = torch.zeros(2, 1, 128, 128, device=device)
        gt[:, :, 32:96, 32:96] = 1.0
        out = model(x, return_dict=True)
        assert out["features"].shape == (2, 64, 32, 32)
        loss, _ = SIDTotalLoss(aux_classifier=False).to(device)(out, gt)
        loss.backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert missing == []

    @pytest.mark.parametrize(
        "cfg_path,primary_bb,sec_bb,mode",
        [
            ("configs/experiments/focal/focal_hrnet_vit_fusion.yaml", "hrnet", "vit", "supervised"),
            (
                "configs/experiments/focal/focal_vit_hrnet_fusion_contrastive.yaml",
                "vit",
                "hrnet",
                "contrastive",
            ),
        ],
    )
    def test_task2_fusion_yaml_configs(self, cfg_path, primary_bb, sec_bb, mode):
        cfg = load_config(cfg_path)
        assert cfg.data.normalize is False
        assert list(cfg.data.image_size) == [512, 512]
        assert cfg.model.backbone == primary_bb
        assert cfg.model.secondary_backbone == sec_bb
        assert cfg.model.freeze_primary_backbone is True
        assert cfg.model.finetune_mode == mode
