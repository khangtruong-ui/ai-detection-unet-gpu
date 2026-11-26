import numpy as np
import pytest
import torch

from sid_unet.postprocessing import (
    MaskPostProcessor,
    apply_morphology,
    fill_mask_holes,
    get_postprocessor_from_config,
    remove_small_components,
)


def test_remove_small_components():
    # 64x64 mask with one large component (10x10 = 100 px) and one small speckle (2x2 = 4 px)
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[5:15, 5:15] = 1.0  # 100 pixels
    mask[40:42, 40:42] = 1.0  # 4 pixels

    filtered, num_removed = remove_small_components(mask, min_area=20)
    assert num_removed == 1
    assert np.sum(filtered[40:42, 40:42]) == 0.0
    assert np.sum(filtered[5:15, 5:15]) == 100.0


def test_fill_mask_holes():
    # 64x64 mask with a square containing a 2x2 hole inside
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[10:30, 10:30] = 1.0
    mask[18:20, 18:20] = 0.0  # 4-pixel hole

    filled, filled_pixels = fill_mask_holes(mask, max_hole_size=50)
    assert filled_pixels == 4
    assert np.all(filled[18:20, 18:20] == 1.0)
    assert np.sum(filled) == 400.0


def test_apply_morphology():
    # Test open, close, open_close
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[5:15, 5:15] = 1.0
    mask[25, 25] = 1.0  # single pixel speckle

    opened = apply_morphology(mask, operation="open", kernel_size=3)
    assert opened[25, 25] == 0.0
    assert opened[10, 10] == 1.0

    closed = apply_morphology(mask, operation="close", kernel_size=3)
    assert closed.shape == (32, 32)

    open_close = apply_morphology(mask, operation="open_close", kernel_size=3)
    assert open_close.shape == (32, 32)

    none_op = apply_morphology(mask, operation="none")
    assert np.array_equal(none_op, mask)


def test_postprocessor_process_single():
    proc = MaskPostProcessor(enabled=True, min_area=16, fill_holes=True, morphology="open_close")

    # Empty mask
    empty = np.zeros((32, 32), dtype=np.float32)
    res, stats = proc.process_single(empty)
    assert np.sum(res) == 0.0
    assert stats["pixels_changed"] == 0

    # Mask with small noise
    noisy = np.zeros((32, 32), dtype=np.float32)
    noisy[2:10, 2:10] = 1.0  # 64 px
    noisy[20, 20] = 1.0  # 1 px noise
    res, stats = proc.process_single(noisy)
    assert stats["enabled"] is True
    assert stats["pixels_changed"] > 0
    assert res[20, 20] == 0.0


def test_postprocessor_process_batch():
    proc = MaskPostProcessor(enabled=True, min_area=10, fill_holes=True)

    # Batch of shape [2, 1, 32, 32] with negative logits for background
    batch = torch.full((2, 1, 32, 32), -5.0)
    batch[0, 0, 5:15, 5:15] = 5.0  # Positive logits (100 px)
    batch[1, 0, 20, 20] = 5.0      # Isolated positive logit (1 px)

    out, stats = proc.process_batch(batch)
    assert out.shape == (2, 1, 32, 32)
    assert len(stats) == 2
    # Sample 0 has large component preserved
    assert torch.sum(out[0]) > 0
    # Sample 1 had only 1 pixel, removed by min_area=10
    assert torch.sum(out[1]) == 0


def test_get_postprocessor_from_config():
    config = {
        "post_processing": {
            "enabled": True,
            "min_area": 128,
            "fill_holes": True,
            "max_hole_size": 512,
            "morphology": "close",
            "morph_kernel_size": 5,
        }
    }
    proc = get_postprocessor_from_config(config)
    assert proc.enabled is True
    assert proc.min_area == 128
    assert proc.morphology == "close"
    assert proc.morph_kernel_size == 5

    # Override enabled
    proc_disabled = get_postprocessor_from_config(config, override_enabled=False)
    assert proc_disabled.enabled is False


def test_process_batch_binary_np_and_parity_with_process_single():
    proc = MaskPostProcessor(enabled=True, min_area=12, fill_holes=True, max_hole_size=32, morphology="open_close", morph_kernel_size=3)

    # Batch of 4: [empty, large block with small hole, isolated specks, mixed]
    batch = torch.full((4, 1, 32, 32), -4.0, dtype=torch.float32)
    batch[1, 0, 4:24, 4:24] = 4.0
    batch[1, 0, 12:14, 12:14] = -4.0  # 4px enclosed hole -> should fill
    batch[2, 0, 10, 10] = 4.0         # 1px speck -> should remove
    batch[2, 0, 20:22, 20:22] = 4.0   # 4px speck (<12) -> should remove
    batch[3, 0, 6:18, 6:18] = 4.0     # 144px block -> keep
    batch[3, 0, 28, 28] = 4.0         # 1px speck -> remove

    batch_out, batch_stats = proc.process_batch(batch, is_logit=True, output_device=torch.device("cpu"))
    assert batch_out.shape == (4, 1, 32, 32)
    assert len(batch_stats) == 4

    for i in range(4):
        single_out, single_stats = proc.process_single(torch.sigmoid(batch[i, 0]))
        np.testing.assert_array_equal(batch_out[i, 0].numpy(), single_out)
        assert batch_stats[i]["original_area"] == single_stats["original_area"]
        assert batch_stats[i]["final_area"] == single_stats["final_area"]
        assert batch_stats[i]["pixels_changed"] == single_stats["pixels_changed"]

    # Direct binary uint8 numpy path + empty batch edge case
    bin_u8 = (torch.sigmoid(batch[:, 0]) >= 0.5).to(torch.uint8).numpy()
    np_out, np_stats = proc.process_batch_binary_np(bin_u8, collect_stats=False)
    assert np_out.dtype == np.uint8
    assert np_out.shape == (4, 1, 32, 32)
    np.testing.assert_array_equal(np_out[:, 0].astype(np.float32), batch_out[:, 0].numpy())
    assert np_stats == [{}, {}, {}, {}]

    empty_out, empty_stats = proc.process_batch(torch.empty((0, 1, 32, 32)), is_logit=True)
    assert empty_out.shape == (0, 1, 32, 32)
    assert empty_stats == []


def test_async_post_process_pipeline_parity_and_error_propagation():
    from sid_unet.postprocessing import AsyncPostProcessPipeline
    from sid_unet.metrics.segmentation import SegmentationMetricTracker

    proc = MaskPostProcessor(enabled=True, min_area=8, fill_holes=True, morphology="open_close", morph_kernel_size=3)
    sync_raw = SegmentationMetricTracker(threshold=0.5)
    sync_post = SegmentationMetricTracker(threshold=0.5)
    async_raw = SegmentationMetricTracker(threshold=0.5)
    async_post = SegmentationMetricTracker(threshold=0.5)
    async_collected = []

    logits = torch.full((3, 1, 24, 24), -3.0)
    logits[0, 0, 4:14, 4:14] = 3.0
    logits[1, 0, 1:3, 1:3] = 3.0
    targets = torch.zeros((3, 1, 24, 24))
    targets[0, 0, 4:14, 4:14] = 1.0
    labels = torch.tensor([2, 0, 0])
    images = torch.zeros((3, 3, 24, 24))
    datasets = ["DS_A", "DS_A", "DS_B"]

    # Sync reference
    raw_auroc = sync_raw.update(logits, targets, labels, is_logit=True, datasets=datasets)
    post_masks, _ = proc.process_batch(logits, is_logit=True)
    sync_post.update(post_masks, targets, labels, is_logit=False, datasets=datasets, precomputed_auroc=raw_auroc)

    # Async pipeline
    with AsyncPostProcessPipeline(proc, async_post, collect_samples=async_collected, max_collect=2) as pipe:
        auroc_t = async_raw.update(logits, targets, labels, is_logit=True, datasets=datasets)
        pipe.submit(logits, targets, labels=labels, datasets=datasets, raw_auroc=auroc_t, is_logit=True, images=images, img_ids=["id0", "id1", "id2"])

    sync_m, _ = sync_post.compute()
    async_m, _ = async_post.compute()
    for key in ("iou", "dice", "f1", "auroc", "binary_auroc", "pixel_acc"):
        assert async_m[key] == pytest.approx(sync_m[key], abs=1e-6)
    assert len(async_collected) == 2
    assert [s["img_id"] for s in async_collected] == ["id0", "id1"]
    assert async_collected[0]["post_mask"] is not None

    # Worker exception propagation test
    class FailingTracker:
        threshold = 0.5

        def update(self, *args, **kwargs):
            raise ValueError("boom in worker")

    failing_pipe = AsyncPostProcessPipeline(proc, FailingTracker())
    failing_pipe.submit(logits, targets, labels=labels, is_logit=True)
    with pytest.raises(ValueError, match="boom in worker"):
        failing_pipe.finish()
