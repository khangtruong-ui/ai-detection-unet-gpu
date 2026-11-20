"""
Tests for FOCAL clustering-anchored hybrid residual calibration (`finetune_mode="hybrid"`),
learnable dual-backbone concatenation scaling, and dual-backbone prior anchoring.
"""

import os
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from sid_unet.losses import build_loss
from sid_unet.models.focal import ClusterResidualRefiner, DualBackboneFusionModule, FOCAL
from sid_unet.models.unet import UNet, build_model
from sid_unet.training.trainer import Trainer
from sid_unet.utils.config import ConfigDict, DEFAULT_CONFIG, deep_merge


class _SyntheticHybridDataset(Dataset):
    """Synthetic tampered dataset for hybrid FOCAL Trainer integration tests."""

    def __init__(self, n_samples: int = 4, size: int = 128):
        self.n_samples = n_samples
        self.size = size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        img = torch.full((3, self.size, self.size), 0.25, dtype=torch.float32)
        mask = torch.zeros((1, self.size, self.size), dtype=torch.float32)
        q = self.size // 4
        img[:, q : 3 * q, q : 3 * q] = 0.85
        mask[:, q : 3 * q, q : 3 * q] = 1.0
        return {
            "image": img,
            "mask": mask,
            "label": torch.tensor(2, dtype=torch.long),
            "dataset": "casia",
        }


class TestDualBackboneFusionScaleAndAnchor:
    """Verify learnable concat scale step-0 parity and dual-backbone prior anchoring."""

    def test_learnable_fusion_scale_exact_identity_at_step0(self):
        f1 = torch.randn(2, 64, 16, 16)
        f2 = torch.randn(2, 32, 16, 16)
        mod_fixed = DualBackboneFusionModule(64, 32, fusion_mode="concat", learnable_fusion_scale=False)
        mod_learn = DualBackboneFusionModule(64, 32, fusion_mode="concat", learnable_fusion_scale=True)
        out_fixed = mod_fixed(f1, f2)
        out_learn = mod_learn(f1, f2)
        assert torch.allclose(out_fixed, out_learn, atol=1e-6)

        loss = out_learn.pow(2).mean()
        loss.backward()
        assert mod_learn.fusion_scale_logits is not None
        assert mod_learn.fusion_scale_logits.grad is not None

    def test_dual_backbone_prior_anchor_active_for_secondary_and_adapter(self):
        model = FOCAL(
            backbone="hrnet",
            secondary_backbone="hrnet",
            freeze_primary_backbone=True,
            secondary_trainable=True,
            secondary_trainable_stages=["stage4"],
            fusion_mode="concat",
            fusion_spatial_align="max",
            use_feature_adapter=True,
            finetune_mode="hybrid",
            prior_anchor_weight=0.25,
            cluster_refiner_l2_weight=0.05,
        )
        # Perturb trainable stage4 and adapter so drift > 0
        with torch.no_grad():
            for p in model.secondary_net.stage4.parameters():
                p.add_(0.05)
            model.feature_adapter.proj_out.weight.fill_(0.05)
            model.cluster_refiner.out_conv.weight.fill_(0.02)

        model.train()
        x = torch.rand(2, 3, 128, 128)
        gt = (torch.rand(2, 1, 128, 128) > 0.7).float()
        out = model(x, gt_masks=gt, return_dict=True)

        assert "prior_anchor_loss" in out
        assert "prior_anchor_penalty" in out
        assert float(out["prior_anchor_loss"].detach()) > 0.0
        assert float(out["prior_anchor_penalty"].detach()) > 0.0
        assert torch.isfinite(out["prior_anchor_penalty"])

        # Anchor modules must not leak into trainable param groups or surgery param counts
        groups = model.get_param_groups(lr=1e-4)
        group_param_ids = {id(p) for g in groups for p in g["params"]}
        for p in model._anchor_secondary_hrnet_stage4.parameters():
            assert id(p) not in group_param_ids


class TestFOCALHybridMode:
    """Verify clustering-anchored hybrid mode parity, gradient decoupling, and round-trip."""

    def test_cluster_residual_refiner_zero_init_identity(self):
        refiner = ClusterResidualRefiner(in_dim=32, num_upsample_stages=2, mid_dim=32)
        feats = torch.randn(2, 32, 32, 32)
        c_logits = torch.randn(2, 1, 128, 128)
        delta = refiner(feats, c_logits, orig_h=128, orig_w=128)
        assert delta.shape == (2, 1, 128, 128)
        assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-7)

    def test_hybrid_step0_matches_contrastive_clustering_and_decouples_gradients(self):
        m_contrast = FOCAL(
            backbone="hrnet",
            finetune_mode="contrastive",
            use_feature_adapter=False,
        ).eval()
        m_hybrid = FOCAL(
            backbone="hrnet",
            finetune_mode="hybrid",
            use_feature_adapter=True,
            cluster_residual_max_scale=2.5,
            detach_refiner_feats=True,
        ).eval()

        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            out_c = m_contrast(x, return_dict=True)
            out_h = m_hybrid(x, return_dict=True)
        assert torch.allclose(out_c["mask_logits"], out_h["mask_logits"], atol=1e-6)

        # Training pass: mask_logits gradients only touch cluster_refiner when detach_refiner_feats=True
        m_hybrid.train()
        assert m_hybrid.requires_gt_masks is True
        gt = torch.zeros(2, 1, 128, 128)
        gt[:, :, 32:64, 32:64] = 1.0
        out_t = m_hybrid(x, gt_masks=gt, return_dict=True)
        assert out_t["skip_mask_loss"] is False
        assert "align_loss" in out_t

        m_hybrid.zero_grad()
        out_t["mask_logits"].mean().backward(retain_graph=True)
        assert m_hybrid.cluster_refiner.out_conv.weight.grad is not None
        assert m_hybrid.cluster_refiner.out_conv.weight.grad.abs().sum().item() > 0.0
        # Backbone/adapter must not receive supervised mask_logits gradients
        assert m_hybrid.feature_adapter.proj_out.weight.grad is None

        # Contrastive align_loss updates feature_adapter
        out_t["align_loss"].backward()
        assert m_hybrid.feature_adapter.proj_out.weight.grad is not None

    def test_hybrid_checkpoint_save_and_reload_roundtrip(self):
        cfg_dict = deep_merge(
            DEFAULT_CONFIG,
            {
                "project": {"name": "test_hybrid_roundtrip", "device": "cpu"},
                "data": {"image_size": [128, 128], "normalize": False, "batch_size": 2},
                "model": {
                    "name": "focal",
                    "backbone": "hrnet",
                    "pretrained": True,
                    "finetune_mode": "hybrid",
                    "use_clustering": True,
                    "use_feature_adapter": True,
                    "hrnet_trainable_stages": ["stage4"],
                    "freeze_backbone_bn": True,
                    "cluster_residual_max_scale": 2.0,
                    "cluster_refiner_dim": 48,
                    "aux_classifier": False,
                    "num_classes": 2,
                },
            },
        )
        cfg = ConfigDict(cfg_dict)
        model = build_model(cfg).eval()
        with torch.no_grad():
            model.cluster_refiner.out_conv.weight.normal_(0.0, 0.05)

        x = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            ref_out = model(x)["mask_logits"]

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "checkpoint_best.pt")
            torch.save(
                {
                    "epoch": 1,
                    "step": 10,
                    "model_state_dict": model.state_dict(),
                    "config": cfg.to_dict(),
                    "best_score": 0.5,
                },
                ckpt_path,
            )
            reloaded, _ = UNet.from_checkpoint(ckpt_path, device="cpu", return_config=True)
            reloaded.eval()
            with torch.no_grad():
                rel_out = reloaded(x)["mask_logits"]
            assert torch.allclose(ref_out, rel_out, atol=1e-5)

    def test_hybrid_trainer_end_to_end_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ConfigDict(
                deep_merge(
                    DEFAULT_CONFIG,
                    {
                        "project": {"name": "test_hybrid_trainer", "device": "cpu", "output_dir": tmpdir},
                        "data": {"image_size": [128, 128], "normalize": False, "batch_size": 2},
                        "model": {
                            "name": "focal",
                            "backbone": "hrnet",
                            "pretrained": True,
                            "finetune_mode": "hybrid",
                            "use_clustering": True,
                            "use_feature_adapter": True,
                            "hrnet_trainable_stages": ["stage4"],
                            "freeze_backbone_bn": True,
                            "prior_anchor_weight": 0.1,
                            "cluster_refiner_l2_weight": 0.02,
                            "aux_classifier": False,
                            "num_classes": 2,
                        },
                        "loss": {
                            "mask_loss_type": "combined",
                            "bce_weight": 0.5,
                            "dice_weight": 0.5,
                            "edge_weight": 0.5,
                            "align_weight": 1.0,
                        },
                        "training": {
                            "epochs": 1,
                            "learning_rate": 1e-4,
                            "auto_batch_size": False,
                            "amp": False,
                            "mixed_precision": "fp32",
                            "precision": "fp32",
                            "save_best": True,
                            "save_latest": False,
                        },
                    },
                )
            )
            ds = _SyntheticHybridDataset(n_samples=4, size=128)
            loader = DataLoader(ds, batch_size=2, shuffle=False)
            model = build_model(cfg)
            loss_fn = build_loss(cfg)
            trainer = Trainer(
                config=cfg,
                model=model,
                loss_fn=loss_fn,
                train_loader=loader,
                val_loader=loader,
            )
            res = trainer.train()
            assert "final_metrics" in res
            assert os.path.isfile(os.path.join(tmpdir, "checkpoints", "checkpoint_best.pt"))
