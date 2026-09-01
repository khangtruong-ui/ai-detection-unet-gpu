import os
import sys
import pytest
import numpy as np
import torch
from PIL import Image
from sid_unet.models.sam3_refiner import (
    denormalize_image_to_pil,
    extract_mask_bounding_boxes,
    SAMRefiner,
    get_sam_refiner,
)



def test_denormalize_image_to_pil():
    # Standard normalized tensor [3, 64, 64]
    t = torch.randn(3, 64, 64)
    pil_img = denormalize_image_to_pil(t)
    assert isinstance(pil_img, Image.Image)
    assert pil_img.size == (64, 64)
    assert pil_img.mode == "RGB"


def test_extract_mask_bounding_boxes():
    # Empty mask
    empty_mask = np.zeros((100, 100), dtype=np.float32)
    boxes = extract_mask_bounding_boxes(empty_mask, min_pixels=10)
    assert boxes == []

    # Single rectangle component: [20:50, 30:70] -> area 30x40 = 1200
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[20:50, 30:70] = 1.0
    boxes = extract_mask_bounding_boxes(mask, min_pixels=16, margin=2)
    assert len(boxes) == 1
    # Bounding box is [x_min, y_min, x_max, y_max]
    x_min, y_min, x_max, y_max = boxes[0]
    assert x_min <= 30 and x_max >= 70
    assert y_min <= 20 and y_max >= 50

    # Multiple components
    mask[70:90, 10:30] = 1.0
    boxes_multi = extract_mask_bounding_boxes(mask, min_pixels=16, margin=2)
    assert len(boxes_multi) == 2


def test_sam_refiner_empty_mask_passthrough():
    # When UNet predicts clean/authentic image (no positive pixels), refiner should skip SAM inference
    class DummyRefiner(SAMRefiner):
        def _load_model(self):
            self.processor = None
            self.model = None

    refiner = DummyRefiner(model_name="dummy", device="cpu", threshold=0.5, min_pixels=16)
    img = Image.new("RGB", (64, 64), (128, 128, 128))
    clean_mask = np.zeros((64, 64), dtype=np.float32)

    refined, metrics = refiner.refine_single_sample(img, clean_mask)
    assert np.array_equal(refined, clean_mask)
    assert metrics["pixels_changed"] == 0
    assert metrics["pixel_change_ratio"] == 0.0
    assert metrics["num_components"] == 0


def test_sam_refiner_join_contrast_mock():
    # Test join and contrast logic with mock SAM outputs
    class MockSAMRefiner(SAMRefiner):
        def _load_model(self):
            self.processor = object()
            self.model = object()

        def refine_single_sample(self, image, unet_mask):
            return super().refine_single_sample(image, unet_mask)

    refiner = MockSAMRefiner(model_name="dummy", device="cpu", threshold=0.5, min_pixels=10)

    # Imbue with mock processor and model
    class MockOutputs:
        pass

    class MockProcessor:
        def __call__(self, images, input_boxes, return_tensors):
            return {"mock": torch.tensor(1)}

        def post_process_instance_segmentation(self, outputs, threshold, target_sizes):
            # Return a precise SAM mask covering [10:60, 10:60]
            sam_mask = np.zeros((64, 64), dtype=np.float32)
            sam_mask[10:60, 10:60] = 1.0
            return [{
                "masks": torch.from_numpy(sam_mask).unsqueeze(0),
                "scores": torch.tensor([0.95]),
            }]

    class MockModel:
        def to(self, dev):
            return self
        def eval(self):
            return self
        def __call__(self, **kwargs):
            return MockOutputs()

    refiner.processor = MockProcessor()
    refiner.model = MockModel()

    # Imperfect UNet mask covering [20:45, 20:45]
    unet_mask = np.zeros((64, 64), dtype=np.float32)
    unet_mask[20:45, 20:45] = 1.0

    img = Image.new("RGB", (64, 64), (200, 200, 200))
    refined_mask, metrics = refiner.refine_single_sample(img, unet_mask)

    # Refined mask should expand to the full SAM object segment [10:60, 10:60]
    assert np.sum(refined_mask) == 50 * 50
    assert metrics["pixels_changed"] > 0
    assert metrics["pixel_change_ratio"] > 0.0
    assert metrics["num_joined_segments"] == 1


def test_sam_refiner_batch_processing():
    class MockSAMRefiner(SAMRefiner):
        def _load_model(self):
            pass

    refiner = MockSAMRefiner(model_name="dummy", device="cpu")
    # Mock refine_single_sample returning 2D array [H, W]
    refiner.refine_single_sample = lambda img, m: (
        (m.squeeze().detach().cpu().numpy() >= 0.5).astype(np.float32),
        {"pixel_change_ratio": 0.0, "pixels_changed": 0},
    )

    batch_imgs = torch.zeros(2, 3, 32, 32)
    batch_logits = torch.randn(2, 1, 32, 32)

    refined_batch, metrics_list = refiner.refine_batch(batch_imgs, batch_logits)
    assert refined_batch.shape == (2, 1, 32, 32)
    assert len(metrics_list) == 2



def test_evaluate_and_cross_eval_cli_args_parsing():
    import argparse
    from sid_unet.evaluate import parse_args as eval_parse_args
    from sid_unet.cross_eval import parse_args as cross_parse_args
    from sid_unet.predict import parse_args as pred_parse_args

    import sys

    # Test evaluate parser
    orig_argv = sys.argv
    try:
        sys.argv = ["sid-eval", "--checkpoint", "dummy.pt", "--segment", "facebook/sam3"]
        args = eval_parse_args()
        assert args.segment == "facebook/sam3"

        sys.argv = ["sid-cross-eval", "--cross-configs", "c.yaml", "--checkpoints", "p.pt", "--segment", "facebook/sam3"]
        args_cross = cross_parse_args()
        assert args_cross.segment == "facebook/sam3"

        sys.argv = ["sid-predict", "--checkpoint", "dummy.pt", "--image", "test.jpg", "--segment", "facebook/sam3"]
        args_pred = pred_parse_args()
        assert args_pred.segment == "facebook/sam3"
    finally:
        sys.argv = orig_argv


def test_predict_and_eval_pipeline_with_segment(monkeypatch, tmp_path):
    import tempfile
    from sid_unet.train import main as train_main
    from sid_unet.evaluate import main as eval_main
    from sid_unet.predict import main as predict_main

    class MockHFDataset:
        def __init__(self, count=4):
            self.samples = [
                {
                    "image": Image.new("RGB", (32, 32), color=(i * 30, 100, 100)),
                    "label": i % 3,
                    "mask": Image.new("L", (32, 32), color=255 if i % 3 == 2 else 0),
                    "img_id": f"mock_{i}",
                }
                for i in range(count)
            ]
        def __iter__(self):
            return iter(self.samples)
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]
        def shuffle(self, seed=None, buffer_size=None):
            return self
        def select(self, indices):
            return [self.samples[i] for i in indices]

    monkeypatch.setattr("sid_unet.dataset.loader.hf_load_dataset", lambda *a, **kw: MockHFDataset(4))

    # Mock get_sam_refiner to return a mock refiner
    class DummyRefiner:
        model_name = "facebook/sam3"
        def refine_single_sample(self, img, u_mask):
            return np.zeros((32, 32), dtype=np.float32), {
                "pixel_change_ratio": 0.05,
                "pixels_changed": 10,
                "unet_mask_ratio": 0.1,
                "refined_mask_ratio": 0.12,
                "num_components": 1,
                "num_joined_segments": 1,
            }
        def refine_batch(self, imgs, logits):
            b_sz = logits.size(0)
            return torch.zeros((b_sz, 1, 32, 32)), [
                {"pixel_change_ratio": 0.05, "pixels_changed": 10, "num_joined_segments": 1}
                for _ in range(b_sz)
            ]

    monkeypatch.setattr("sid_unet.models.sam3_refiner.get_sam_refiner", lambda *a, **kw: DummyRefiner())
    monkeypatch.setattr("sid_unet.evaluate.get_sam_refiner", lambda *a, **kw: DummyRefiner())
    monkeypatch.setattr("sid_unet.cross_eval.get_sam_refiner", lambda *a, **kw: DummyRefiner())

    output_dir = str(tmp_path / "outputs")
    pred_dir = str(tmp_path / "predictions")

    # 1. Train quick model
    monkeypatch.setattr(sys, "argv", [
        "sid-train",
        "--config", "configs/default.yaml",
        "--override",
        f"project.output_dir={output_dir}",
        "project.device=cpu",
        "training.epochs=1",
        "data.num_workers=0",
        "data.train_samples_per_epoch=2",
        "data.val_samples=2",
        "data.batch_size=2",
        "model.features=[8, 16]",
        "data.image_size=[32, 32]",
        "logging.save_sample_images=false",
        "training.amp=false",
    ])
    train_main()
    best_ckpt = os.path.join(output_dir, "checkpoints", "checkpoint_best.pt")

    # 2. Evaluate with --segment facebook/sam3
    monkeypatch.setattr(sys, "argv", [
        "sid-eval",
        "--checkpoint", best_ckpt,
        "--split", "test",
        "--samples", "2",
        "--segment", "facebook/sam3",
        "--override",
        "data.num_workers=0",
        "data.batch_size=2",
        "project.device=cpu",
        "--output_dir", os.path.join(output_dir, "eval_reports"),
    ])
    eval_res = eval_main()
    assert os.path.exists(os.path.join(output_dir, "eval_reports", "evaluation_report.md"))
    with open(os.path.join(output_dir, "eval_reports", "evaluation_report.md")) as f:
        md_content = f.read()
        assert "SAM Mask Refinement Analysis" in md_content
        assert "facebook/sam3" in md_content

    # 3. Predict with --segment facebook/sam3
    sample_img = str(tmp_path / "test_img.png")
    Image.new("RGB", (32, 32), color=(100, 150, 200)).save(sample_img)
    monkeypatch.setattr(sys, "argv", [
        "sid-predict",
        "--checkpoint", best_ckpt,
        "--image", sample_img,
        "--segment", "facebook/sam3",
        "--output_dir", pred_dir,
        "--image_size", "32", "32",
        "--device", "cpu",
    ])
    predict_main()
    assert os.path.exists(os.path.join(pred_dir, "test_img_mask.png"))

