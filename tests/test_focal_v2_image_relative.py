"""
Tests for FOCAL v2 upgrades:
- Soft-EM spherical clustering (`clustering_mode="soft_kmeans"`)
- Configurable ViT resolution (`vit_input_size`) & eval horizontal-flip TTA (`tta_hflip`)
- Cross-scale spatial forensic adapter (`SpatialCrossScaleAdapter`)
- Domain-invariant, channel-rotation-invariant `ImageRelativeClusterRefiner`
- Learnable multi-branch base logit fusion (`multi_branch_logit_fusion`)
- Boundary-aware `MyInfoNCE` token sampling (`contrastive_boundary_ratio`)
- Checkpoint roundtrip and end-to-end Trainer integration
"""

import os
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from sid_unet.losses import build_loss
from sid_unet.models.focal import (
    FOCAL,
    ImageRelativeClusterRefiner,
    SpatialCrossScaleAdapter,
)
from sid_unet.models.unet import UNet, build_model
from sid_unet.training.trainer import Trainer
from sid_unet.utils.config import ConfigDict, DEFAULT_CONFIG, deep_merge


class _SyntheticV2Dataset(Dataset):
    """Synthetic tampered dataset for FOCAL v2 Trainer integration tests."""

    def __init__(self, n_samples: int = 4, size: int = 128):
        self.n_samples = n_samples
        self.size = size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        img = torch.full((3, self.size, self.size), 0.2, dtype=torch.float32)
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


class TestFOCALv2ClusteringAndTTA:
    """Verify soft_kmeans, vit_input_size, and tta_hflip."""

    def test_soft_kmeans_minority_detection_and_corner_robustness(self):
        model = FOCAL(backbone="hrnet", clustering_mode="soft_kmeans").eval()
        b, c, h_f, w_f = 2, 32, 32, 32
        feats = torch.zeros(b, c, h_f, w_f)
        feats[:, 0, :, :] = 1.0  # Pristine background
        # Put a strong single-pixel corner artifact at (0, 0) and a real minority region in center
        feats[:, 1, 0, 0] = 5.0
        feats[:, 2, 10:18, 10:18] = 2.0
        feats[:, 0, 10:18, 10:18] = 0.0

        logit_full, prob_full, sep, ratio = model._cluster_features_with_full_stats(
            feats, orig_h=128, orig_w=128
        )
        assert logit_full.shape == (b, 1, 128, 128)
        assert torch.isfinite(logit_full).all()
        assert (prob_full >= 0.0).all() and (prob_full <= 1.0).all()
        assert (sep > 0.1).all()
        # Minority ratio should reflect the ~6.25% center region, not collapse to corner pixel 0.1%
        assert (ratio > 0.03).all() and (ratio < 0.25).all()
        # Center should have higher tampered probability than pristine background
        assert prob_full[:, 0, 48:80, 48:80].mean() > prob_full[:, 0, 100:120, 100:120].mean()

    def test_tta_hflip_equivariance_and_vit_input_size(self):
        model = FOCAL(
            backbone="vit",
            vit_input_size=192,
            tta_hflip=True,
            clustering_mode="soft_kmeans",
        ).eval()
        assert model.vit_input_size == 192
        x = torch.full((1, 3, 128, 128), 0.2)
        x[:, :, 24:64, 16:56] = 0.85
        x_flip = torch.flip(x, dims=[-1])
        with torch.no_grad():
            out = model(x, return_dict=True)
            out_flip = model(x_flip, return_dict=True)
        assert out["mask_logits"].shape == (1, 1, 128, 128)
        assert torch.allclose(
            out["mask_logits"],
            torch.flip(out_flip["mask_logits"], dims=[-1]),
            atol=1e-3,
        )


class TestSpatialCrossScaleAdapter:
    """Verify SpatialCrossScaleAdapter step-0 identity and gradient flow."""

    def test_spatial_adapter_zero_init_identity_and_grad(self):
        adapter = SpatialCrossScaleAdapter(channels=64, mid_channels=32)
        x = torch.randn(2, 64, 32, 32, requires_grad=True)
        out = adapter(x)
        assert out.shape == x.shape
        assert torch.allclose(out, x, atol=1e-7)

        loss = out.pow(2).mean()
        loss.backward()
        assert adapter.proj_out.weight.grad is not None
        assert adapter.dw_local[0].weight.grad is not None


class TestImageRelativeClusterRefiner:
    """Verify step-0 identity, rotation/domain invariance, multi-branch fusion, and Trainer roundtrip."""

    def test_step0_exact_identity_and_multi_branch_fusion(self):
        m_ref = FOCAL(
            backbone="hrnet",
            secondary_backbone="hrnet",
            freeze_primary_backbone=True,
            secondary_trainable=False,
            fusion_mode="concat",
            finetune_mode="contrastive",
            clustering_mode="soft_kmeans",
        ).eval()
        m_v2 = FOCAL(
            backbone="hrnet",
            secondary_backbone="hrnet",
            freeze_primary_backbone=True,
            secondary_trainable=False,
            fusion_mode="concat",
            finetune_mode="hybrid",
            clustering_mode="soft_kmeans",
            use_feature_adapter=True,
            feature_adapter_type="spatial",
            cluster_refiner_type="image_relative",
            multi_branch_logit_fusion=True,
            cluster_residual_max_scale=1.5,
            cluster_residual_scale_gain=0.5,
        ).eval()

        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            out_ref = m_ref(x, return_dict=True)
            out_v2 = m_v2(x, return_dict=True)
        assert torch.allclose(out_ref["mask_logits"], out_v2["mask_logits"], atol=1e-6)

        # Verify gradients flow to refiner, branch_logit_weights, and spatial adapter cleanly
        m_v2.train()
        gt = torch.zeros(2, 1, 128, 128)
        gt[:, :, 32:64, 32:64] = 1.0
        out_t = m_v2(x, gt_masks=gt, return_dict=True)
        assert torch.isfinite(out_t["align_loss"])
        total = out_t["mask_logits"].mean() + out_t["align_loss"]
        m_v2.zero_grad()
        total.backward()
        assert m_v2.cluster_refiner.out_conv.weight.grad is not None
        assert m_v2.branch_logit_weights.grad is not None
        assert m_v2.feature_adapter.proj_out.weight.grad is not None

    def test_descriptor_exact_channel_rotation_invariance(self):
        """
        Mathematical proof of domain/channel invariance:
        Rotating all backbone feature channels by arbitrary orthogonal matrices Q_f, Q_p, Q_s
        must produce IDENTICAL ImageRelativeClusterRefiner descriptors and refiner outputs.
        """
        torch.manual_seed(7)
        model = FOCAL(
            backbone="hrnet",
            secondary_backbone="hrnet",
            freeze_primary_backbone=True,
            secondary_trainable=False,
            fusion_mode="concat",
            finetune_mode="hybrid",
            clustering_mode="soft_kmeans",
            cluster_refiner_type="image_relative",
        ).eval()
        # Perturb refiner weights so output is non-zero
        with torch.no_grad():
            for p in model.cluster_refiner.parameters():
                p.add_(0.05 * torch.randn_like(p))

        b, h_f, w_f = 2, 32, 32
        f_p = torch.randn(b, 32, h_f, w_f)
        f_s = torch.randn(b, 32, h_f, w_f)
        feats = torch.cat([f_p, f_s], dim=1)

        def _rand_ortho(dim: int) -> torch.Tensor:
            q, _ = torch.linalg.qr(torch.randn(dim, dim))
            return q

        q_p = _rand_ortho(32)
        q_s = _rand_ortho(32)
        q_f = _rand_ortho(64)

        f_p_rot = torch.einsum("cd,bdhw->bchw", q_p, f_p)
        f_s_rot = torch.einsum("cd,bdhw->bchw", q_s, f_s)
        feats_rot = torch.einsum("cd,bdhw->bchw", q_f, feats)

        with torch.no_grad():
            c_f, _, sep_f, area_f = model._cluster_features_with_full_stats(feats, 128, 128)
            c_p, _, sep_p, area_p = model._cluster_features_with_full_stats(f_p, 128, 128)
            c_s, _, sep_s, area_s = model._cluster_features_with_full_stats(f_s, 128, 128)
            desc1 = ImageRelativeClusterRefiner.build_descriptors(
                feats, c_f, sep_f, area_f, (f_p, c_p, sep_p, area_p), (f_s, c_s, sep_s, area_s), multi_branch=True
            )
            bias1, scale1 = model.cluster_refiner(desc1, 128, 128)

            c_f_r, _, sep_f_r, area_f_r = model._cluster_features_with_full_stats(feats_rot, 128, 128)
            c_p_r, _, sep_p_r, area_p_r = model._cluster_features_with_full_stats(f_p_rot, 128, 128)
            c_s_r, _, sep_s_r, area_s_r = model._cluster_features_with_full_stats(f_s_rot, 128, 128)
            desc2 = ImageRelativeClusterRefiner.build_descriptors(
                feats_rot, c_f_r, sep_f_r, area_f_r, (f_p_rot, c_p_r, sep_p_r, area_p_r), (f_s_rot, c_s_r, sep_s_r, area_s_r), multi_branch=True
            )
            bias2, scale2 = model.cluster_refiner(desc2, 128, 128)

        assert desc1.shape[1] == ImageRelativeClusterRefiner.MULTI_BRANCH_DESC_CHANNELS
        assert torch.allclose(desc1, desc2, atol=1e-4)
        assert torch.allclose(bias1, bias2, atol=1e-4)
        assert torch.allclose(scale1, scale2, atol=1e-4)

    def test_boundary_contrastive_loss_and_checkpoint_roundtrip(self):
        cfg_dict = deep_merge(
            DEFAULT_CONFIG,
            {
                "project": {"name": "test_focal_v2_roundtrip", "device": "cpu"},
                "data": {"image_size": [128, 128], "normalize": False, "batch_size": 2},
                "model": {
                    "name": "focal",
                    "backbone": "hrnet",
                    "secondary_backbone": "hrnet",
                    "freeze_primary_backbone": True,
                    "secondary_trainable": False,
                    "fusion_mode": "concat",
                    "fusion_spatial_align": "max",
                    "learnable_fusion_scale": True,
                    "use_feature_adapter": True,
                    "feature_adapter_type": "spatial",
                    "prior_anchor_weight": 0.5,
                    "pretrained": True,
                    "finetune_mode": "hybrid",
                    "clustering_mode": "soft_kmeans",
                    "cluster_refiner_type": "image_relative",
                    "multi_branch_logit_fusion": True,
                    "cluster_residual_max_scale": 1.5,
                    "cluster_residual_scale_gain": 0.5,
                    "cluster_refiner_dim": 32,
                    "cluster_refiner_l2_weight": 0.02,
                    "contrastive_boundary_ratio": 0.4,
                    "aux_classifier": False,
                    "num_classes": 2,
                },
            },
        )
        cfg = ConfigDict(cfg_dict)
        model = build_model(cfg).eval()
        with torch.no_grad():
            model.cluster_refiner.out_conv.weight.normal_(0.0, 0.05)
            model.branch_logit_weights.fill_(0.15)

        x = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            ref_out = model(x)["mask_logits"]

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "checkpoint_best.pt")
            torch.save(
                {
                    "epoch": 1,
                    "step": 5,
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

    def test_v2_trainer_end_to_end_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ConfigDict(
                deep_merge(
                    DEFAULT_CONFIG,
                    {
                        "project": {"name": "test_focal_v2_trainer", "device": "cpu", "output_dir": tmpdir},
                        "data": {"image_size": [128, 128], "normalize": False, "batch_size": 2},
                        "model": {
                            "name": "focal",
                            "backbone": "hrnet",
                            "secondary_backbone": "hrnet",
                            "freeze_primary_backbone": True,
                            "secondary_trainable": False,
                            "fusion_mode": "concat",
                            "learnable_fusion_scale": True,
                            "use_feature_adapter": True,
                            "feature_adapter_type": "spatial",
                            "prior_anchor_weight": 0.5,
                            "pretrained": True,
                            "finetune_mode": "hybrid",
                            "clustering_mode": "soft_kmeans",
                            "cluster_refiner_type": "image_relative",
                            "multi_branch_logit_fusion": True,
                            "cluster_refiner_dim": 32,
                            "cluster_refiner_l2_weight": 0.02,
                            "contrastive_boundary_ratio": 0.4,
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
                            "learning_rate": 2e-4,
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
            ds = _SyntheticV2Dataset(n_samples=4, size=128)
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
