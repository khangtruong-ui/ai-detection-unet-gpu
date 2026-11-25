"""Regression tests for forensic audit findings across evaluation metrics and FOCAL."""

from __future__ import annotations

import torch
import torch.nn as nn

from sid_unet.cross_eval import eval_single_batch as cross_eval_single_batch
from sid_unet.evaluate import eval_single_batch as evaluate_single_batch_fn
from sid_unet.metrics.segmentation import (
    SegmentationMetricTracker,
    compute_binary_auroc_batch,
    compute_binary_metrics_batch,
)
from sid_unet.models.focal import FOCAL
from sid_unet.postprocessing import MaskPostProcessor


def test_binary_auroc_fastpath_matches_continuous_on_single_class_masks():
    h, w = 16, 16
    # Sample 0: all-zero GT + 0% FP
    # Sample 1: all-zero GT + 10% FP (mean_p=0.10 < 0.5 -> continuous helper returns 1.0)
    # Sample 2: all-zero GT + 75% FP (mean_p=0.75 >= 0.5 -> 1 - 0.75 = 0.25)
    # Sample 3: all-one GT  + 10% FN (mean_p=0.90 >= 0.5 -> 1.0)
    # Sample 4: all-one GT  + 75% FN (mean_p=0.25 < 0.5 -> 0.25)
    # Sample 5: 50% GT      + mixed binary pred (rec=0.75, spec=0.75 -> 0.75)
    targets = torch.zeros(6, h, w)
    targets[3] = 1.0
    targets[4] = 1.0
    targets[5, : h // 2, :] = 1.0

    preds = torch.zeros(6, h, w)
    preds.view(6, -1)[1, :26] = 1.0
    preds.view(6, -1)[2, :192] = 1.0
    preds.view(6, -1)[3, :230] = 1.0
    preds.view(6, -1)[4, :64] = 1.0
    preds[5, :6, :] = 1.0
    preds[5, 8:10, :] = 1.0

    res = compute_binary_metrics_batch(preds, targets, threshold=0.5, is_logit=False)
    ref = compute_binary_auroc_batch(preds, targets, is_logit=False)

    assert torch.allclose(res["auroc"], ref, atol=1e-6)
    assert torch.allclose(res["binary_auroc"], ref, atol=1e-6)
    assert res["auroc"][1].item() == 1.0
    assert abs(res["auroc"][2].item() - 0.25) < 1e-4
    assert res["auroc"][3].item() == 1.0
    assert abs(res["auroc"][4].item() - 0.25) < 1e-4
    assert abs(res["auroc"][5].item() - 0.75) < 1e-4


def test_segmentation_tracker_preserves_continuous_auroc_with_score_masks():
    # Continuous sub-threshold logits (all sigmoid < 0.5 so post_binary is all zeros -> binary_auroc=0.5),
    # while true continuous pixel probabilities strictly rank foreground above background -> AUROC = 1.0.
    bg = torch.linspace(-5.0, -2.5, 16).view(1, 1, 1, 16).expand(2, 1, 16, 16).clone()
    fg = torch.linspace(-1.5, -0.2, 8).view(1, 1, 8, 1).expand(2, 1, 8, 16)
    raw_logits = bg
    raw_logits[:, :, :8, :] = fg  # max prob sigmoid(-0.2) ~= 0.450 < 0.5, many distinct values
    targets = torch.zeros(2, 1, 16, 16)
    targets[:, 0, :8, :] = 1.0
    post_binary = (torch.sigmoid(raw_logits) >= 0.5).float()

    raw_tracker = SegmentationMetricTracker(threshold=0.5)
    raw_tracker.update(raw_logits, targets, is_logit=True, datasets=["DS_A", "DS_B"])
    raw_m, _ = raw_tracker.compute()

    post_tracker = SegmentationMetricTracker(threshold=0.5)
    post_tracker.update(
        post_binary,
        targets,
        is_logit=False,
        datasets=["DS_A", "DS_B"],
        score_masks=raw_logits,
        score_is_logit=True,
    )
    post_m, _, post_pds = post_tracker.compute(return_per_dataset=True)

    assert raw_m["auroc"] == 1.0
    assert post_m["auroc"] == 1.0
    assert post_m["pixel_auroc"] == 1.0
    assert abs(post_m["binary_auroc"] - 0.5) < 1e-5
    assert post_m["iou"] == 0.0
    assert post_pds["DS_A"]["auroc"] == 1.0
    assert post_pds["DS_B"]["auroc"] == 1.0


def test_evaluate_and_cross_eval_preserve_continuous_auroc_under_postprocessing():
    class _FixedLogitModel(nn.Module):
        def __init__(self, logits: torch.Tensor):
            super().__init__()
            self.register_buffer("logits", logits)
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return self.logits + 0.0 * self.dummy

    logits = torch.full((2, 1, 32, 32), -4.0)
    # Soft tampered signal with some pixels < 0 logit (prob < 0.5) and small false positive speckle
    logits[:, 0, 8:24, 8:24] = torch.linspace(-0.8, 2.0, 16).unsqueeze(1).expand(16, 16)
    logits[:, 0, 0, 0] = 3.0  # tiny 1-px noise removed by min_area=16

    targets = torch.zeros(2, 1, 32, 32)
    targets[:, 0, 8:24, 8:24] = 1.0
    images = torch.rand(2, 3, 32, 32)
    labels = torch.tensor([2, 2])

    model = _FixedLogitModel(logits).eval()
    postproc = MaskPostProcessor(enabled=True, min_area=16, fill_holes=True, morphology="open_close")
    loss_fn = lambda out, m, l: (torch.tensor(0.0), {})

    for fn in (evaluate_single_batch_fn, cross_eval_single_batch):
        raw_t = SegmentationMetricTracker(threshold=0.5)
        post_t = SegmentationMetricTracker(threshold=0.5)
        cls_t = type("_Cls", (), {"update": lambda *a, **k: None})()
        batch = {"image": images, "mask": targets, "label": labels, "dataset": ["TestDS", "TestDS"]}

        fn(
            batch=batch,
            model=model,
            loss_fn=loss_fn,
            device=torch.device("cpu"),
            raw_seg_tracker=raw_t,
            cls_tracker=cls_t,
            threshold=0.5,
            post_seg_tracker=post_t,
            postprocessor=postproc,
        )
        raw_m, _ = raw_t.compute()
        post_m, _ = post_t.compute()

        assert abs(post_m["auroc"] - raw_m["auroc"]) < 1e-6
        assert abs(post_m["pixel_auroc"] - raw_m["pixel_auroc"]) < 1e-6
        assert post_m["auroc"] > post_m["binary_auroc"] + 0.05


def test_cluster_refiner_l2_weight_anchors_groupnorm_to_one_not_zero():
    model = FOCAL(
        backbone="hrnet",
        secondary_backbone="hrnet",
        freeze_primary_backbone=True,
        secondary_trainable=False,
        fusion_mode="concat",
        finetune_mode="hybrid",
        cluster_refiner_type="image_relative",
        multi_branch_logit_fusion=True,
        cluster_refiner_l2_weight=0.06,
        prior_anchor_weight=0.0,
        neck_l2_anchor_weight=0.0,
        pretrained=True,
    )

    gn_params = [
        m.weight
        for m in model.cluster_refiner.modules()
        if isinstance(m, nn.GroupNorm) and m.weight is not None
    ]
    assert len(gn_params) > 0
    assert all(torch.allclose(p.detach(), torch.ones_like(p)) for p in gn_params)

    # Zero non-GroupNorm params to isolate GroupNorm L2 regularization gradient directly
    with torch.no_grad():
        for m in model.cluster_refiner.modules():
            if not isinstance(m, nn.GroupNorm):
                for p in m.parameters(recurse=False):
                    p.zero_()

    x_dummy = torch.zeros(1, 3, 16, 16)
    feats = torch.randn(1, 64, 16, 16)
    model.zero_grad()
    _, pen_at_one = model._compute_prior_anchor_terms(x_dummy, feats, feats)
    assert pen_at_one is not None
    assert pen_at_one.item() == 0.0
    pen_at_one.backward()
    for p in gn_params:
        assert p.grad is not None
        assert torch.allclose(p.grad, torch.zeros_like(p.grad))

    # Verify conv weights (not normalization scale) are still pulled toward 0
    with torch.no_grad():
        model.cluster_refiner.out_conv.weight.fill_(0.5)
    model.zero_grad()
    _, pen_with_conv = model._compute_prior_anchor_terms(x_dummy, feats, feats)
    assert pen_with_conv is not None and pen_with_conv.item() > 0.0
    pen_with_conv.backward()
    assert model.cluster_refiner.out_conv.weight.grad is not None
    assert (model.cluster_refiner.out_conv.weight.grad > 0.0).all()


def test_ensure_raw_rgb_is_batch_independent():
    model = FOCAL(backbone="hrnet", pretrained=True).eval()
    torch.manual_seed(11)
    raw_img = torch.rand(1, 3, 16, 16) * 0.8 + 0.1  # already in [0.1, 0.9]
    norm_img = (torch.rand(1, 3, 16, 16) - 0.485) / 0.229  # has values < -0.5 or > 1.5

    ref_raw = model._ensure_raw_rgb(raw_img)
    ref_norm = model._ensure_raw_rgb(norm_img)

    mixed_batch = torch.cat([raw_img, norm_img], dim=0)
    mixed_out = model._ensure_raw_rgb(mixed_batch)

    assert torch.allclose(mixed_out[0], ref_raw[0], atol=1e-6)
    assert torch.allclose(mixed_out[1], ref_norm[0], atol=1e-6)
