"""
Tests for FOCAL v4 upgrades:
- Per-image dead-channel-guarded HRNet AdaBN-Mix (`hrnet_adabn_lambda`, `running_var <= 1e-5` guard)
- Strictly per-image batch-size independence (`bs=1` == `bs=4`) for AdaBN-Mix, affinity diffusion, and Otsu calibration
- Vectorized affinity graph diffusion (`_diffuse_logits_on_graph`), spherical feature smoothing, & Otsu calibration HFlip equivariance
- Orthogonal zero-bias `PreNeckForensicProjector` + spectral orthogonality & contrastive loss
- Non-local affinity-propagating `ImageRelativeClusterRefiner` v4 exact channel-rotation invariance & zero-init step-0 parity
- `adapter_ortho_weight` decorrelation penalty on `BranchBalancedSpatialAdapter`
- Opt-in `JointMaskZoomCrop` and `JointSoftBoundaryFeather` augmentations
- End-to-end `Trainer` integration & checkpoint roundtrip
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from sid_unet.dataset.transforms import (
    JointMaskZoomCrop,
    JointSoftBoundaryFeather,
    get_transforms,
)
from sid_unet.losses import build_loss
from sid_unet.models.focal import (
    FOCAL,
    ImageRelativeClusterRefiner,
    PreNeckForensicProjector,
)
from sid_unet.models.unet import UNet, build_model
from sid_unet.training.trainer import Trainer
from sid_unet.utils.config import ConfigDict, DEFAULT_CONFIG, deep_merge


class _SyntheticV4Dataset(Dataset):
    def __init__(self, n_samples: int = 4, size: int = 128):
        self.n_samples = n_samples
        self.size = size

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        img = torch.full((3, self.size, self.size), 0.25, dtype=torch.float32)
        mask = torch.zeros((1, self.size, self.size), dtype=torch.float32)
        q = self.size // 4
        if idx % 2 == 0:
            img[:, q : 3 * q, q : 3 * q] = 0.80
            mask[:, q : 3 * q, q : 3 * q] = 1.0
        else:
            m = self.size // 8
            img[:, m : self.size - m, m : self.size - m] = 0.80
            mask[:, m : self.size - m, m : self.size - m] = 1.0
        return {
            "image": img,
            "mask": mask,
            "label": torch.tensor(2, dtype=torch.long),
            "dataset": "casia",
        }


def test_hrnet_adabn_mix_dead_channel_guard_and_batch_independence():
    model = FOCAL(
        backbone="hrnet",
        hrnet_adabn_lambda=0.25,
        hrnet_adabn_stages=("stage4",),
    ).eval()

    # Check dead subnormal channels in stage4.2.branches.0.1.bn2 exist in pretrained HRNet
    dead_bn = model.net.stage4[2].branches[0][1].bn2
    orig_rm = dead_bn.running_mean.clone()
    orig_rv = dead_bn.running_var.clone()
    assert bool((orig_rv <= 1e-5).any())

    torch.manual_seed(42)
    x = torch.rand(4, 3, 128, 128)
    with torch.no_grad():
        f_batch = model._forward_hrnet_maybe_adabn(model.net, x)
        f_single = torch.cat(
            [model._forward_hrnet_maybe_adabn(model.net, x[i : i + 1]) for i in range(4)],
            dim=0,
        )

    assert torch.isfinite(f_batch).all()
    assert torch.allclose(f_batch, f_single, atol=1e-5)
    # Must never mutate running stats
    assert torch.equal(dead_bn.running_mean, orig_rm)
    assert torch.equal(dead_bn.running_var, orig_rv)


def test_hrnet_shift_gated_adabn_suppresses_low_shift_and_preserves_batch_independence():
    base_model = FOCAL(backbone="hrnet", hrnet_adabn_lambda=0.0, hrnet_adabn_stages=("stage4",)).eval()
    high_floor_model = FOCAL(
        backbone="hrnet",
        hrnet_adabn_lambda=0.25,
        hrnet_adabn_stages=("stage4",),
        hrnet_adabn_shift_floor=50.0,
        hrnet_adabn_shift_scale=0.35,
    ).eval()
    active_gate_model = FOCAL(
        backbone="hrnet",
        hrnet_adabn_lambda=0.25,
        hrnet_adabn_stages=("stage4",),
        hrnet_adabn_shift_floor=0.05,
        hrnet_adabn_shift_scale=0.35,
    ).eval()

    torch.manual_seed(7)
    x = torch.rand(4, 3, 128, 128)
    with torch.no_grad():
        f_base = base_model._forward_hrnet_maybe_adabn(base_model.net, x)
        f_suppressed = high_floor_model._forward_hrnet_maybe_adabn(high_floor_model.net, x)
        f_active_b = active_gate_model._forward_hrnet_maybe_adabn(active_gate_model.net, x)
        f_active_s = torch.cat(
            [active_gate_model._forward_hrnet_maybe_adabn(active_gate_model.net, x[i : i + 1]) for i in range(4)],
            dim=0,
        )

    assert torch.allclose(f_suppressed, f_base, atol=1e-5)
    assert not torch.allclose(f_active_b, f_base, atol=1e-3)
    assert torch.allclose(f_active_b, f_active_s, atol=1e-5)


def test_affinity_diffusion_and_otsu_hflip_equivariance_and_batch_independence():
    torch.manual_seed(0)
    b, c, h_f, w_f = 3, 32, 24, 24
    feats = torch.randn(b, c, h_f, w_f)
    logits = torch.randn(b, 1, 64, 64)

    P = FOCAL._build_affinity_matrix(feats, tau=0.08, grid_size=24)
    P_flip = FOCAL._build_affinity_matrix(torch.flip(feats, dims=[-1]), tau=0.08, grid_size=24)
    assert P.shape == (b, 24 * 24, 24 * 24)
    assert torch.allclose(P.sum(dim=-1), torch.ones(b, 24 * 24), atol=1e-5)

    # Diffusion equivariance
    l_diff = FOCAL._diffuse_logits_on_graph(logits, P, beta=0.4, steps=2)
    l_diff_flip = FOCAL._diffuse_logits_on_graph(torch.flip(logits, dims=[-1]), P_flip, beta=0.4, steps=2)
    assert torch.allclose(l_diff, torch.flip(l_diff_flip, dims=[-1]), atol=1e-5)

    # Otsu calibration equivariance & batch independence
    l_otsu = FOCAL._apply_otsu_calibration(l_diff, alpha=0.5)
    l_otsu_flip = FOCAL._apply_otsu_calibration(torch.flip(l_diff, dims=[-1]), alpha=0.5)
    assert torch.allclose(l_otsu, torch.flip(l_otsu_flip, dims=[-1]), atol=1e-5)

    l_otsu_single = torch.cat(
        [FOCAL._apply_otsu_calibration(l_diff[i : i + 1], alpha=0.5) for i in range(b)],
        dim=0,
    )
    assert torch.allclose(l_otsu, l_otsu_single, atol=1e-5)


def test_v4_refiner_rotation_invariance_zero_init_and_orthogonality_penalties():
    torch.manual_seed(11)
    b, c, h_f, w_f = 2, 64, 16, 16
    feats = torch.randn(b, c, h_f, w_f)
    c_logits = torch.randn(b, 1, 64, 64)
    sep = torch.tensor([0.35, 0.15])
    area = torch.tensor([0.22, 0.45])
    raw_rgb = torch.rand(b, 3, 64, 64)

    refiner = ImageRelativeClusterRefiner(
        num_upsample_stages=2,
        mid_dim=32,
        multi_branch=False,
        use_affinity_prop=True,
        use_srm_desc=True,
    ).eval()

    P1 = FOCAL._build_affinity_matrix(feats, tau=0.08, grid_size=16)
    desc1 = ImageRelativeClusterRefiner.build_descriptors(
        feats=feats,
        c_fused_logits=c_logits,
        sep_fused=sep,
        area_fused=area,
        multi_branch=False,
        srm_kernels=refiner._srm_kernels,
        raw_rgb=raw_rgb,
        use_srm_desc=True,
    )
    assert desc1.shape == (b, ImageRelativeClusterRefiner.BASE_DESC_CHANNELS + 2, h_f, w_f)

    # Step-0 zero-init check
    b0, s0 = refiner(desc1, 64, 64, affinity_P=P1)
    assert torch.allclose(b0, torch.zeros_like(b0), atol=1e-7)
    assert torch.allclose(s0, torch.zeros_like(s0), atol=1e-7)

    # Perturb weights away from zero and verify exact invariance under orthogonal channel rotation Q
    with torch.no_grad():
        refiner.aff_proj.weight.normal_(0.0, 0.1)
        refiner.out_conv.weight.normal_(0.0, 0.1)
        q_mat, _ = torch.linalg.qr(torch.randn(c, c))
    feats_rot = torch.einsum("cd,bdhw->bchw", q_mat, feats)
    P2 = FOCAL._build_affinity_matrix(feats_rot, tau=0.08, grid_size=16)
    desc2 = ImageRelativeClusterRefiner.build_descriptors(
        feats=feats_rot,
        c_fused_logits=c_logits,
        sep_fused=sep,
        area_fused=area,
        multi_branch=False,
        srm_kernels=refiner._srm_kernels,
        raw_rgb=raw_rgb,
        use_srm_desc=True,
    )
    with torch.no_grad():
        bias1, scale1 = refiner(desc1, 64, 64, affinity_P=P1)
        bias2, scale2 = refiner(desc2, 64, 64, affinity_P=P2)
    assert torch.allclose(desc1, desc2, atol=1e-5)
    assert torch.allclose(bias1, bias2, atol=1e-5)
    assert torch.allclose(scale1, scale2, atol=1e-5)


def test_border_knn_descriptors_invariance_equivariance_and_batch_independence():
    torch.manual_seed(11)
    b, c, h_f, w_f = 4, 24, 16, 16
    feats = torch.randn(b, c, h_f, w_f)
    sec_feats = torch.randn(b, 16, h_f, w_f)
    c_logits = torch.randn(b, 1, 64, 64)
    sep = torch.tensor([0.2, 0.5, 0.7, 0.9])
    area = torch.tensor([0.1, 0.2, 0.3, 0.15])
    raw_rgb = torch.rand(b, 3, 64, 64)

    refiner = ImageRelativeClusterRefiner(
        num_upsample_stages=2,
        mid_dim=32,
        multi_branch=True,
        use_affinity_prop=True,
        use_srm_desc=True,
        use_border_knn_desc=True,
    )
    prim = (feats[:, :12], c_logits, sep, area)
    sec = (sec_feats, c_logits, sep, area)
    P = FOCAL._build_affinity_matrix(feats, tau=0.08, grid_size=16)

    desc = ImageRelativeClusterRefiner.build_descriptors(
        feats=feats,
        c_fused_logits=c_logits,
        sep_fused=sep,
        area_fused=area,
        primary_branch=prim,
        secondary_branch=sec,
        multi_branch=True,
        srm_kernels=refiner._srm_kernels,
        raw_rgb=raw_rgb,
        use_srm_desc=True,
        use_border_knn_desc=True,
    )
    assert desc.shape == (
        b,
        ImageRelativeClusterRefiner.MULTI_BRANCH_DESC_CHANNELS
        + ImageRelativeClusterRefiner.SRM_DESC_CHANNELS
        + ImageRelativeClusterRefiner.BORDER_KNN_DESC_CHANNELS,
        h_f,
        w_f,
    )

    # Step-0 zero-init check
    b0, s0 = refiner(desc, 64, 64, affinity_P=P)
    assert torch.allclose(b0, torch.zeros_like(b0), atol=1e-7)
    assert torch.allclose(s0, torch.zeros_like(s0), atol=1e-7)

    # Orthogonal channel rotation Q invariance
    with torch.no_grad():
        q_mat, _ = torch.linalg.qr(torch.randn(c, c))
        q_sec, _ = torch.linalg.qr(torch.randn(16, 16))
    b_loc_1, b_brd_1 = ImageRelativeClusterRefiner._border_knn_descriptors(feats)
    b_loc_rot, b_brd_rot = ImageRelativeClusterRefiner._border_knn_descriptors(torch.einsum("cd,bdhw->bchw", q_mat, feats))
    assert torch.allclose(b_loc_1, b_loc_rot, atol=1e-5)
    assert torch.allclose(b_brd_1, b_brd_rot, atol=1e-5)

    # HFlip + VFlip equivariance & per-image batch independence
    b_loc_h, b_brd_h = ImageRelativeClusterRefiner._border_knn_descriptors(torch.flip(feats, dims=[-1]))
    assert torch.allclose(b_loc_h, torch.flip(b_loc_1, dims=[-1]), atol=1e-5)
    assert torch.allclose(b_brd_h, torch.flip(b_brd_1, dims=[-1]), atol=1e-5)

    desc_rot = ImageRelativeClusterRefiner.build_descriptors(
        feats=torch.einsum("cd,bdhw->bchw", q_mat, feats),
        c_fused_logits=c_logits,
        sep_fused=sep,
        area_fused=area,
        primary_branch=(torch.einsum("cd,bdhw->bchw", q_mat[:12, :12], feats[:, :12]), c_logits, sep, area),
        secondary_branch=(torch.einsum("cd,bdhw->bchw", q_sec, sec_feats), c_logits, sep, area),
        multi_branch=True,
        srm_kernels=refiner._srm_kernels,
        raw_rgb=raw_rgb,
        use_srm_desc=True,
        use_border_knn_desc=True,
    )
    assert torch.allclose(desc[:, -2:], desc_rot[:, -2:], atol=1e-5)

    b_loc_single = torch.cat(
        [ImageRelativeClusterRefiner._border_knn_descriptors(feats[i : i + 1])[0] for i in range(b)],
        dim=0,
    )
    assert torch.allclose(b_loc_1, b_loc_single, atol=1e-5)


def test_preneck_projector_and_adapter_ortho_losses():
    pn = PreNeckForensicProjector(in_dim=128, proj_dim=32)
    assert pn.proj.bias is None
    x = torch.randn(2, 128, 12, 12)
    y = pn(x)
    assert y.shape == (2, 32, 12, 12)
    assert torch.allclose(y.norm(dim=1), torch.ones(2, 12, 12), atol=1e-5)

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
        adapter_ortho_weight=0.5,
        finetune_mode="hybrid",
        clustering_mode="kmeans_robust",
        cluster_polarity_mode="border_aware",
        vit_input_size=128,
        vit_feature_taps=[22, 23],
        vit_tap_cluster_size=16,
        learnable_tap_weights=True,
        use_preneck_projector=True,
        preneck_proj_dim=64,
        preneck_ortho_weight=0.5,
        preneck_contrastive_weight=0.5,
        cluster_refiner_type="image_relative",
        multi_branch_logit_fusion=True,
        refiner_use_affinity_prop=True,
        refiner_use_srm_desc=True,
        hrnet_adabn_lambda=0.10,
        affinity_diff_beta=0.35,
        otsu_cal_alpha=0.50,
    ).train()

    # Perturb secondary_adapter.proj_out with correlated channels to verify adapter_ortho_weight penalty
    with torch.no_grad():
        w = model.feature_adapter.secondary_adapter.proj_out.weight
        w[0].fill_(0.05)
        w[1].fill_(0.05)

    img = torch.rand(2, 3, 128, 128)
    gt = torch.zeros(2, 1, 128, 128)
    gt[:, :, 32:96, 32:96] = 1.0
    out = model(img, return_dict=True, gt_masks=gt)
    assert float(out["prior_anchor_penalty"].item()) > 0.0
    assert float(out["align_loss"].item()) >= 0.0
    loss = out["mask_logits"].mean() + out["prior_anchor_penalty"] + out["align_loss"]
    loss.backward()
    assert model.preneck_projector.proj.weight.grad is not None
    assert model.feature_adapter.secondary_adapter.proj_out.weight.grad is not None


def test_curriculum_augmentations_mask_alignment():
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    mask = np.zeros((128, 128), dtype=np.float32)
    mask[48:80, 48:80] = 1.0

    crop_tf = JointMaskZoomCrop(p=1.0, zoom_prob=1.0, min_crop_scale=0.4, max_crop_scale=0.6)
    img_c, mask_c = crop_tf(img, mask)
    assert img_c.size == (mask_c.shape[1], mask_c.shape[0])
    assert mask_c.max() == 1.0
    assert mask_c.mean() > mask.mean()

    feather_tf = JointSoftBoundaryFeather(p=1.0)
    img_f, mask_f = feather_tf(img, mask)
    assert img_f.size == (128, 128)
    assert np.array_equal(mask_f, mask)
    assert not np.array_equal(np.asarray(img_f), arr)

    tf = get_transforms(
        image_size=(128, 128),
        is_train=True,
        augment_config={
            "mask_zoom_crop_prob": 1.0,
            "soft_boundary_feather_prob": 1.0,
        },
        normalize=False,
    )
    img_t, mask_t = tf(img, mask)
    assert img_t.shape == (3, 128, 128)
    assert mask_t.shape == (1, 128, 128)


def test_v4_trainer_end_to_end_and_checkpoint_roundtrip():
    torch.manual_seed(13)
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_dict = deep_merge(
            DEFAULT_CONFIG,
            {
                "project": {"name": "test_focal_v4", "device": "cpu", "output_dir": tmpdir},
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
                    "prior_anchor_weight": 1.5,
                    "adapter_ortho_weight": 0.05,
                    "finetune_mode": "hybrid",
                    "clustering_mode": "kmeans_robust",
                    "cluster_polarity_mode": "border_aware",
                    "vit_input_size": 128,
                    "vit_feature_taps": [22, 23],
                    "vit_tap_fusion_mode": "dual_logit",
                    "vit_tap_cluster_size": 16,
                    "learnable_tap_weights": True,
                    "tta_hflip": True,
                    "hrnet_adabn_lambda": 0.10,
                    "hrnet_adabn_stages": ["stage4"],
                    "affinity_diff_beta": 0.35,
                    "affinity_diff_tau": 0.08,
                    "affinity_diff_steps": 2,
                    "otsu_cal_alpha": 0.50,
                    "use_preneck_projector": True,
                    "preneck_proj_dim": 64,
                    "preneck_ortho_weight": 0.1,
                    "preneck_contrastive_weight": 0.5,
                    "cluster_refiner_type": "image_relative",
                    "multi_branch_logit_fusion": True,
                    "refiner_use_affinity_prop": True,
                    "refiner_use_srm_desc": True,
                    "refiner_use_border_knn_desc": True,
                    "cluster_residual_max_scale": 0.45,
                    "cluster_residual_scale_gain": 0.25,
                    "cluster_refiner_dim": 32,
                    "cluster_refiner_l2_weight": 0.03,
                    "refiner_sep_gate_tau": 0.30,
                    "refiner_zero_mean_bias_weight": 0.1,
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
        ds = _SyntheticV4Dataset(n_samples=4, size=128)
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
