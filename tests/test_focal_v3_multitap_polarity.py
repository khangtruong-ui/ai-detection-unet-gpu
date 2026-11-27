"""
Tests for FOCAL v3 upgrades:
- Multi-tap ViT extraction (`ImageEncoderViT.forward_with_taps`, `FOCALViTNet.forward_with_taps`)
- Flip-equivariant robust k-means (`clustering_mode="kmeans_robust"`)
- Domain-invariant border-aware & learned antisymmetric cluster polarity (`ClusterPolarityHead`)
- Branch-balanced spatial adapter (`BranchBalancedSpatialAdapter`) & equal per-branch drift anchoring
- Separability-gated, DC-bias-regularized, and orthogonality-regularized `ImageRelativeClusterRefiner`
- End-to-end Trainer integration and checkpoint roundtrip
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from sid_unet.losses import build_loss
from sid_unet.models.focal import (
    BranchBalancedSpatialAdapter,
    ClusterPolarityHead,
    FOCAL,
    FOCALViTNet,
)
from sid_unet.models.unet import UNet, build_model
from sid_unet.training.trainer import Trainer
from sid_unet.utils.config import ConfigDict, DEFAULT_CONFIG, deep_merge


class _SyntheticV3Dataset(Dataset):
    """Synthetic tampered dataset with interior tampered blocks (including >50% area)."""

    def __init__(self, n_samples: int = 4, size: int = 128):
        self.n_samples = n_samples
        self.size = size

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        img = torch.full((3, self.size, self.size), 0.2, dtype=torch.float32)
        mask = torch.zeros((1, self.size, self.size), dtype=torch.float32)
        if idx % 2 == 0:
            q = self.size // 4
            img[:, q : 3 * q, q : 3 * q] = 0.85
            mask[:, q : 3 * q, q : 3 * q] = 1.0
        else:
            # >50% interior tampered region (border remains pristine)
            m = self.size // 8
            img[:, m : self.size - m, m : self.size - m] = 0.85
            mask[:, m : self.size - m, m : self.size - m] = 1.0
        return {
            "image": img,
            "mask": mask,
            "label": torch.tensor(2, dtype=torch.long),
            "dataset": "casia",
        }


def test_vit_forward_with_taps_preserves_forward_and_validates_indices():
    torch.manual_seed(0)
    vit = FOCALViTNet(img_size=128, depth=4, embed_dim=64, num_heads=4, out_chans=32).eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        y_std = vit(x)
        y_tap, taps = vit.forward_with_taps(x, tap_indices=[1, -1])

    assert torch.allclose(y_std, y_tap, atol=1e-6)
    assert set(taps.keys()) == {1, -1}
    assert taps[1].shape == (2, 64, 8, 8)
    assert taps[-1].shape == (2, 64, 8, 8)

    with pytest.raises(ValueError):
        vit.forward_with_taps(x, tap_indices=[10])


def test_kmeans_robust_hflip_equivariance_and_corner_immunity():
    model = FOCAL(
        backbone="hrnet",
        clustering_mode="kmeans_robust",
        cluster_polarity_mode="border_aware",
    ).eval()
    b, c, h_f, w_f = 2, 32, 32, 32
    feats = torch.zeros(b, c, h_f, w_f)
    feats[:, 0, :, :] = 1.0
    # Corner outlier should not hijack initialization
    feats[:, 1, 0, 0] = 5.0
    # Genuine interior patch off-center so HFlip is non-trivial
    feats[:, 2, 10:18, 6:14] = 2.0
    feats[:, 0, 10:18, 6:14] = 0.0

    logit_orig, prob_orig, sep, _ = model._cluster_features_with_full_stats(
        feats, orig_h=64, orig_w=64
    )
    logit_flip, _, _, _ = model._cluster_features_with_full_stats(
        torch.flip(feats, dims=[-1]), orig_h=64, orig_w=64
    )
    assert (sep > 0.1).all()
    assert torch.allclose(logit_orig, torch.flip(logit_flip, dims=[-1]), atol=1e-4)
    assert prob_orig[:, 0, 22:34, 14:26].mean() > prob_orig[:, 0, 48:60, 48:60].mean()


def test_border_aware_and_learned_polarity_antisymmetry():
    head = ClusterPolarityHead(hidden_dim=16)
    # Perturb final layer away from zero to test antisymmetry cleanly
    with torch.no_grad():
        head.mlp[-1].weight.normal_(0.0, 0.2)
        head.mlp[-1].bias.normal_(0.0, 0.1)

    g0 = torch.tensor([[0.65, 0.02, 0.0, 0.10, 0.30, 0.05]])
    g1 = torch.tensor([[0.35, 0.98, 1.0, 0.75, 0.60, 0.08]])
    z = torch.tensor([[0.42]])
    out_01 = head(g0, g1, z, margin_scale=5.0)
    out_10 = head(g1, g0, -z, margin_scale=5.0)
    assert torch.allclose(out_01, -out_10, atol=1e-6)

    # Test >50% interior tampered region (56.25% area, 0% border)
    feats = torch.zeros(1, 32, 32, 32)
    feats[:, 0, :, :] = 1.0
    feats[:, 0, 4:28, 4:28] = 0.0
    feats[:, 1, 4:28, 4:28] = 1.0

    m_minority = FOCAL(backbone="hrnet", clustering_mode="kmeans_robust", cluster_polarity_mode="minority").eval()
    m_border = FOCAL(
        backbone="hrnet",
        clustering_mode="kmeans_robust",
        cluster_polarity_mode="border_aware",
        polarity_border_weight=0.75,
        polarity_area_gate=0.25,
    ).eval()
    _, p_min, _, _ = m_minority._cluster_features_with_full_stats(feats, 64, 64)
    _, p_brd, _, _ = m_border._cluster_features_with_full_stats(feats, 64, 64)

    # Minority inverts the >50% interior mask; border-aware recovers it cleanly
    assert p_min[:, 0, 24:40, 24:40].mean() < 0.2
    assert p_brd[:, 0, 24:40, 24:40].mean() > 0.8


def test_branch_balanced_adapter_and_gated_ortho_refiner():
    adapter = BranchBalancedSpatialAdapter(primary_dim=32, secondary_dim=16)
    x = torch.randn(2, 48, 16, 16, requires_grad=True)
    y = adapter(x)
    assert torch.allclose(y, x, atol=1e-7)
    y.pow(2).mean().backward()
    assert adapter.primary_adapter.proj_out.weight.grad is not None
    assert adapter.secondary_adapter.proj_out.weight.grad is not None

    model = FOCAL(
        backbone="vit",
        secondary_backbone="hrnet",
        freeze_primary_backbone=True,
        secondary_trainable=False,
        fusion_mode="concat",
        fusion_spatial_align="max",
        use_feature_adapter=True,
        feature_adapter_type="branch_balanced",
        prior_anchor_weight=1.0,
        finetune_mode="hybrid",
        clustering_mode="kmeans_robust",
        cluster_polarity_mode="border_aware",
        vit_input_size=128,
        vit_feature_taps=[22, 23],
        vit_tap_fusion_mode="dual_logit",
        vit_tap_cluster_size=16,
        learnable_tap_weights=True,
        cluster_refiner_type="image_relative",
        multi_branch_logit_fusion=True,
        cluster_residual_max_scale=0.40,
        cluster_residual_scale_gain=0.20,
        cluster_refiner_dim=32,
        refiner_sep_gate_tau=0.50,
        refiner_zero_mean_bias_weight=0.5,
        refiner_ortho_weight=0.5,
    )

    # Inject collinear out_conv weights & non-zero bias to verify ortho, dc_bias, and sep_gate
    with torch.no_grad():
        row = torch.ones_like(model.cluster_refiner.out_conv.weight[0]) * 0.1
        model.cluster_refiner.out_conv.weight[0].copy_(row)
        model.cluster_refiner.out_conv.weight[1].copy_(-row)
        model.cluster_refiner.out_conv.bias.copy_(torch.tensor([-2.0, 0.0]))

    model.train()
    img = torch.rand(1, 3, 128, 128)
    gt = torch.zeros(1, 1, 128, 128)
    gt[:, :, 32:96, 32:96] = 1.0
    out = model(img, return_dict=True, gt_masks=gt)
    assert "prior_anchor_penalty" in out
    # With collinear out_conv rows + non-zero DC bias, penalty must be strictly positive
    assert float(out["prior_anchor_penalty"].item()) > 0.1

    # Verify low-separability gating attenuates refiner bias smoothly
    model.eval()
    uniform_img = torch.full((1, 3, 128, 128), 0.5)
    with torch.no_grad():
        out_unif = model(uniform_img, return_dict=True)
    assert torch.isfinite(out_unif["mask_logits"]).all()
    assert float(out_unif["cluster_separability"].item()) < 0.5


def test_v3_trainer_end_to_end_step_and_checkpoint_roundtrip():
    torch.manual_seed(7)
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_dict = deep_merge(
            DEFAULT_CONFIG,
            {
                "project": {"name": "test_focal_v3", "device": "cpu", "output_dir": tmpdir},
                "data": {"dataset_name": "mock", "image_size": [128, 128], "normalize": False, "batch_size": 2, "num_workers": 0},
                "model": {
                    "name": "focal",
                    "backbone": "vit",
                    "secondary_backbone": "hrnet",
                    "freeze_primary_backbone": True,
                    "secondary_trainable": False,
                    "freeze_backbone_bn": True,
                    "fusion_mode": "concat",
                    "fusion_spatial_align": "max",
                    "learnable_fusion_scale": True,
                    "use_feature_adapter": True,
                    "feature_adapter_type": "branch_balanced",
                    "prior_anchor_weight": 1.0,
                    "finetune_mode": "hybrid",
                    "clustering_mode": "kmeans_robust",
                    "cluster_polarity_mode": "learned",
                    "polarity_border_weight": 0.75,
                    "polarity_area_gate": 0.25,
                    "polarity_loss_weight": 0.5,
                    "vit_input_size": 128,
                    "vit_feature_taps": [22, 23],
                    "vit_tap_fusion_mode": "dual_logit",
                    "vit_tap_cluster_size": 16,
                    "learnable_tap_weights": True,
                    "tta_hflip": True,
                    "cluster_refiner_type": "image_relative",
                    "multi_branch_logit_fusion": True,
                    "cluster_residual_max_scale": 0.40,
                    "cluster_residual_scale_gain": 0.20,
                    "cluster_refiner_dim": 32,
                    "cluster_refiner_l2_weight": 0.05,
                    "refiner_sep_gate_tau": 0.35,
                    "refiner_zero_mean_bias_weight": 0.2,
                    "refiner_ortho_weight": 0.1,
                    "aux_classifier": False,
                },
                "loss": {"mask_loss_type": "combined", "bce_weight": 0.5, "dice_weight": 0.5, "pos_weight": 1.5, "align_weight": 1.0},
                "training": {"epochs": 1, "learning_rate": 1e-3, "auto_batch_size": False, "amp": False, "mixed_precision": "fp32", "precision": "fp32"},
            },
        )
        cfg = ConfigDict(cfg_dict)
        model = build_model(cfg)
        loss_fn = build_loss(cfg)
        ds = _SyntheticV3Dataset(n_samples=4, size=128)
        loader = DataLoader(ds, batch_size=2, shuffle=False)

        trainer = Trainer(
            config=cfg,
            model=model,
            loss_fn=loss_fn,
            train_loader=loader,
            val_loader=loader,
        )
        res = trainer.train()
        assert "final_metrics" in res

        ckpt_path = os.path.join(tmpdir, "checkpoints", "checkpoint_best.pt")
        assert os.path.isfile(ckpt_path)

        reloaded, _ = UNet.from_checkpoint(ckpt_path, return_config=True)
        reloaded.eval()
        model.eval()
        x = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            o1 = model(x, return_dict=True)
            o2 = reloaded(x, return_dict=True)
        assert torch.allclose(o1["mask_logits"], o2["mask_logits"], atol=1e-5)
