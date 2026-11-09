import numpy as np
import pytest
from PIL import Image
import torch
from sid_unet.dataset.mask_utils import (
    process_sample_mask,
    ensure_rgb_image,
    check_image_mask_mismatch,
    is_mask_provided,
    decode_mask_bytes,
)
from sid_unet.dataset.loader import process_raw_sample


def test_label_0_full_black_mask():
    # Label 0: Real image -> all zeros
    mask = process_sample_mask(mask_input=None, label=0, image_size=(256, 256))
    assert mask.shape == (256, 256)
    assert np.all(mask == 0.0)
    assert mask.dtype == np.float32


def test_label_1_full_white_mask():
    # Label 1: Fully AI generated -> all ones
    mask = process_sample_mask(mask_input=None, label=1, image_size=(256, 256))
    assert mask.shape == (256, 256)
    assert np.all(mask == 1.0)
    assert mask.dtype == np.float32


def test_label_2_pil_mask():
    # Label 2: Inpainting/tampered -> ground truth mask
    raw_mask = Image.new("L", (100, 100), color=0)
    # Paint top half white
    for x in range(100):
        for y in range(50):
            raw_mask.putpixel((x, y), 255)

    processed = process_sample_mask(mask_input=raw_mask, label=2, image_size=(256, 256))
    assert processed.shape == (256, 256)
    # Top half should be 1.0, bottom half 0.0
    assert np.all(processed[:128, :] == 1.0)
    assert np.all(processed[128:, :] == 0.0)


def test_ensure_rgb_image():
    # Grayscale image
    gray_img = Image.new("L", (64, 64), color=128)
    rgb_img = ensure_rgb_image(gray_img)
    assert rgb_img.mode == "RGB"

    # RGBA image
    rgba_img = Image.new("RGBA", (64, 64), color=(100, 150, 200, 255))
    rgb_img = ensure_rgb_image(rgba_img)
    assert rgb_img.mode == "RGB"

    # Numpy array
    np_img = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb_img = ensure_rgb_image(np_img)
    assert rgb_img.mode == "RGB"


def test_two_column_dataset_mask_processing():
    # 2-column format (image, mask) where label is not present / None (e.g. KhangTruong/IMD2020)
    raw_mask = Image.new("L", (100, 100), color=0)
    # Paint top half white
    for x in range(100):
        for y in range(50):
            raw_mask.putpixel((x, y), 255)

    processed = process_sample_mask(mask_input=raw_mask, label=None, image_size=(256, 256))
    assert processed.shape == (256, 256)
    assert np.all(processed[:128, :] == 1.0)
    assert np.all(processed[128:, :] == 0.0)

    # With numpy array input
    np_mask = np.zeros((100, 100), dtype=np.uint8)
    np_mask[:50, :] = 255
    processed_np = process_sample_mask(mask_input=np_mask, label=None, image_size=(256, 256))
    assert processed_np.shape == (256, 256)
    assert np.all(processed_np[:128, :] == 1.0)
    assert np.all(processed_np[128:, :] == 0.0)

    # With torch tensor input
    t_mask = torch.zeros((100, 100))
    t_mask[:50, :] = 1.0
    processed_t = process_sample_mask(mask_input=t_mask, label=None, image_size=(256, 256))
    assert processed_t.shape == (256, 256)
    assert np.all(processed_t[:128, :] == 1.0)
    assert np.all(processed_t[128:, :] == 0.0)

    # With None mask input
    processed_none = process_sample_mask(mask_input=None, label=None, image_size=(256, 256))
    assert processed_none.shape == (256, 256)
    assert np.all(processed_none == 0.0)


def test_check_image_mask_mismatch():
    # 1. Matching PIL image and mask
    img = Image.new("RGB", (128, 128))
    mask = Image.new("L", (128, 128))
    res = check_image_mask_mismatch(img, mask)
    assert res["mismatch"] is False
    assert len(res["issues"]) == 0
    assert res["image_size"] == (128, 128)
    assert res["mask_size"] == (128, 128)

    # 2. Mismatched dimensions (e.g. image 256x128 vs mask 128x128)
    img_mismatch = Image.new("RGB", (256, 128))
    res_mismatch = check_image_mask_mismatch(img_mismatch, mask)
    assert res_mismatch["mismatch"] is True
    assert any("Spatial size mismatch" in issue for issue in res_mismatch["issues"])

    # 3. Missing image
    res_no_img = check_image_mask_mismatch(None, mask)
    assert res_no_img["mismatch"] is True
    assert "Image is None or missing" in res_no_img["issues"]

    # 4. With raise_on_mismatch=True
    with pytest.raises(ValueError):
        check_image_mask_mismatch(img_mismatch, mask, raise_on_mismatch=True)

    # 5. Numpy array matching & mismatch
    np_img = np.zeros((64, 64, 3), dtype=np.uint8)
    np_mask = np.zeros((64, 64), dtype=np.uint8)
    assert check_image_mask_mismatch(np_img, np_mask)["mismatch"] is False

    np_mask_diff = np.zeros((32, 32), dtype=np.uint8)
    assert check_image_mask_mismatch(np_img, np_mask_diff)["mismatch"] is True

    # 6. PyTorch tensor matching & mismatch
    t_img = torch.zeros(3, 64, 64)
    t_mask = torch.zeros(1, 64, 64)
    assert check_image_mask_mismatch(t_img, t_mask)["mismatch"] is False

    t_mask_diff = torch.zeros(1, 32, 32)
    assert check_image_mask_mismatch(t_img, t_mask_diff)["mismatch"] is True


def test_mask_utils_with_dict_and_bytes():
    import io
    # Test ensure_rgb_image with bytes
    im = Image.new("RGB", (64, 64), color="green")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    raw_b = buf.getvalue()

    rgb_from_b = ensure_rgb_image(raw_b)
    assert rgb_from_b.size == (64, 64)
    assert rgb_from_b.mode == "RGB"

    rgb_from_d = ensure_rgb_image({"bytes": raw_b, "path": None})
    assert rgb_from_d.size == (64, 64)
    assert rgb_from_d.mode == "RGB"

    # Test process_sample_mask with bytes dict
    m = Image.new("L", (64, 64), color=255)
    buf_m = io.BytesIO()
    m.save(buf_m, format="PNG")
    mask_b = buf_m.getvalue()

    mask_out = process_sample_mask({"bytes": mask_b}, label=2, image_size=(64, 64))
    assert mask_out.shape == (64, 64)
    assert np.all(mask_out == 1.0)


def test_is_mask_provided():
    assert is_mask_provided(None) is False
    assert is_mask_provided({"bytes": None, "path": None}) is False
    assert is_mask_provided(b"") is False
    assert is_mask_provided(Image.new("L", (10, 10))) is True
    assert is_mask_provided({"bytes": b"fake_bytes"}) is True
    assert is_mask_provided(np.zeros((10, 10))) is True
    assert is_mask_provided(torch.zeros((10, 10))) is True


def test_label_1_with_provided_tampering_mask():
    # When label is 1 but an explicit ground truth mask is provided (e.g. KhangTruong/Tampered-Images-Combined-Dataset-Fake)
    raw_mask = Image.new("L", (100, 100), color=0)
    for x in range(50):
        for y in range(50):
            raw_mask.putpixel((x, y), 255)

    processed = process_sample_mask(mask_input=raw_mask, label=1, image_size=(256, 256))
    assert processed.shape == (256, 256)
    # Top-left quadrant should be 1.0, rest should be 0.0
    assert np.all(processed[:128, :128] == 1.0)
    assert np.all(processed[128:, :] == 0.0)
    assert np.all(processed[:, 128:] == 0.0)

    # When label is 1 and mask_input is None (saberzl/SID_Set fully synthetic), returns all ones
    synthetic_mask = process_sample_mask(mask_input=None, label=1, image_size=(256, 256))
    assert np.all(synthetic_mask == 1.0)


def test_process_raw_sample_tampered_dataset_fake():
    # Emulates a sample from KhangTruong/Tampered-Images-Combined-Dataset-Fake:
    # label is 1 (fake), mask is a localized tampering mask (not all ones)
    raw_img = Image.new("RGB", (64, 64), color="blue")
    raw_mask = Image.new("L", (64, 64), color=0)
    for x in range(32):
        for y in range(32):
            raw_mask.putpixel((x, y), 255)

    sample = {
        "id": "sample_001",
        "image": raw_img,
        "mask": raw_mask,
        "label": 1,
        "label_text": "fake",
        "dataset": "CASIA_V2",
        "method": "Splicing",
    }
    processed = process_raw_sample(sample, transform=None, target_image_size=(64, 64))
    assert processed["label"].item() == 2  # Partially Synthetic / Tampered
    assert processed["mask"].shape == (1, 64, 64)
    assert processed["dataset"] == "CASIA_V2"
    assert processed["method"] == "Splicing"
    # Quadrant has 1s, other parts have 0s
    assert float(processed["mask"].sum()) > 0
    assert float(processed["mask"].sum()) < float(processed["mask"].numel())


def test_decode_mask_bytes_png():
    import io
    # Create known binary mask
    m = Image.new("L", (64, 64), color=0)
    for y in range(32):
        for x in range(64):
            m.putpixel((x, y), 255)

    buf = io.BytesIO()
    m.save(buf, format="PNG")
    raw_b = buf.getvalue()

    decoded = decode_mask_bytes(raw_b)
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (64, 64)
    assert decoded.dtype == np.float32
    assert np.all(decoded[:32, :] == 1.0)
    assert np.all(decoded[32:, :] == 0.0)


def test_decode_mask_bytes_resize():
    import io
    m = Image.new("L", (64, 64), color=255)
    buf = io.BytesIO()
    m.save(buf, format="PNG")
    raw_b = buf.getvalue()

    decoded_resized = decode_mask_bytes(raw_b, image_size=(128, 128))
    assert decoded_resized.shape == (128, 128)
    assert np.all(decoded_resized == 1.0)

    decoded_rect = decode_mask_bytes(raw_b, image_size=(32, 64))
    assert decoded_rect.shape == (64, 32) or decoded_rect.shape == (32, 64)


def test_decode_mask_bytes_continuous_vs_threshold():
    import io
    # Grayscale image with gradient values
    grad_arr = np.linspace(0, 255, 100, dtype=np.uint8).reshape(10, 10)
    pil_grad = Image.fromarray(grad_arr, mode="L")
    buf = io.BytesIO()
    pil_grad.save(buf, format="PNG")
    raw_b = buf.getvalue()

    # Continuous decoding (threshold=None)
    cont = decode_mask_bytes(raw_b, threshold=None)
    assert cont.min() == 0.0
    assert cont.max() == 1.0
    # Must contain continuous values between 0 and 1
    assert np.any((cont > 0.0) & (cont < 1.0))

    # Binarized decoding (threshold=0.5)
    binary = decode_mask_bytes(raw_b, threshold=0.5)
    unique_vals = set(np.unique(binary))
    assert unique_vals.issubset({0.0, 1.0})


def test_decode_mask_bytes_edge_cases():
    import io
    # 1. None returns zeros
    zeros_none = decode_mask_bytes(None, image_size=(64, 64))
    assert zeros_none.shape == (64, 64)
    assert np.all(zeros_none == 0.0)

    # 2. Empty bytes returns zeros
    zeros_empty = decode_mask_bytes(b"", image_size=(64, 64))
    assert zeros_empty.shape == (64, 64)
    assert np.all(zeros_empty == 0.0)

    # 3. Bytearray and BytesIO
    m = Image.new("L", (32, 32), color=255)
    buf = io.BytesIO()
    m.save(buf, format="PNG")
    raw_b = buf.getvalue()

    assert decode_mask_bytes(bytearray(raw_b)).shape == (32, 32)
    assert decode_mask_bytes(io.BytesIO(raw_b)).shape == (32, 32)

    # 4. Dict with bytes
    assert decode_mask_bytes({"bytes": raw_b}).shape == (32, 32)

    # 5. Numpy array & Torch Tensor
    arr = np.ones((40, 40), dtype=np.float32)
    assert decode_mask_bytes(arr).shape == (40, 40)
    t = torch.ones(40, 40)
    assert decode_mask_bytes(t).shape == (40, 40)

    # 6. Invalid type
    with pytest.raises(TypeError, match="Unsupported mask input type"):
        decode_mask_bytes(12345)




