"""
CLI inference and mask prediction entrypoint for SID-UNet.
Supports single image or folder inference with mask exports and visual overlays.

Usage:
    python -m sid_unet.predict --checkpoint outputs/checkpoints/checkpoint_best.pt --image test.jpg --output_dir predictions
    python -m sid_unet.predict --checkpoint outputs/checkpoints/checkpoint_best.pt --input_dir /path/to/images --save_overlay
"""

from __future__ import annotations

import argparse
import os
import glob
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from tabulate import tabulate

from sid_unet.models.unet import build_model
from sid_unet.utils.config import load_config, DEFAULT_CONFIG, deep_merge, ConfigDict


def parse_args():
    parser = argparse.ArgumentParser(description="Predict AI-generated image mask using trained UNet")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--image", type=str, default=None, help="Path to single input image")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory of input images")
    parser.add_argument("--output_dir", type=str, default="predictions", help="Directory to save output masks")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256], help="Inference resolution [H, W]")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization probability threshold")
    parser.add_argument("--save_overlay", action="store_true", help="Save color overlay visualization")
    parser.add_argument("--device", type=str, default="auto", help="Device ('auto', 'cuda', 'cpu')")
    return parser.parse_args()


def load_model_for_inference(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    saved_cfg = ckpt.get("config", {})
    merged = deep_merge(DEFAULT_CONFIG, saved_cfg)
    config = ConfigDict(merged)

    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config


def preprocess_image(image_path: str, target_size: Tuple[int, int]) -> Tuple[torch.Tensor, Image.Image]:
    orig_pil = Image.open(image_path).convert("RGB")
    h, w = target_size
    resized_pil = orig_pil.resize((w, h), resample=Image.BILINEAR)

    # Tensor normalization (ImageNet stats)
    img_t = TF.to_tensor(resized_pil)
    norm_t = TF.normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return norm_t.unsqueeze(0), orig_pil


def create_overlay(orig_pil: Image.Image, mask_2d: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """Create RGB overlay highlighting masked synthetic regions in red."""
    w, h = orig_pil.size
    mask_pil = Image.fromarray((mask_2d * 255).astype(np.uint8)).resize((w, h), resample=Image.NEAREST)
    mask_arr = np.array(mask_pil) > 128

    orig_arr = np.array(orig_pil, dtype=np.float32)
    overlay_arr = orig_arr.copy()

    # Red highlight: [255, 50, 50]
    overlay_arr[mask_arr] = (1 - alpha) * orig_arr[mask_arr] + alpha * np.array([255, 50, 50], dtype=np.float32)
    return Image.fromarray(np.clip(overlay_arr, 0, 255).astype(np.uint8))


def main():
    args = parse_args()

    if not args.image and not args.input_dir:
        raise ValueError("Must provide either --image or --input_dir.")

    dev_str = args.device
    if dev_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_str)

    model, config = load_model_for_inference(args.checkpoint, device)
    target_size = (args.image_size[0], args.image_size[1])

    image_paths: List[str] = []
    if args.image:
        image_paths.append(args.image)
    if args.input_dir:
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
            image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))

    os.makedirs(args.output_dir, exist_ok=True)
    class_names = {0: "Real", 1: "Fully AI", 2: "Partially AI (Tampered)"}

    results_table = []

    print(f"Running inference on {len(image_paths)} image(s) using device: {device}...")
    with torch.no_grad():
        for img_path in image_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            inp_tensor, orig_pil = preprocess_image(img_path, target_size)
            inp_tensor = inp_tensor.to(device)

            outputs = model(inp_tensor)
            if isinstance(outputs, tuple):
                mask_logits, class_logits = outputs
            else:
                mask_logits, class_logits = outputs, None

            probs = torch.sigmoid(mask_logits).squeeze().cpu().numpy()
            bin_mask = (probs >= args.threshold).astype(np.uint8)

            # Resize predicted mask back to original image dimensions
            orig_w, orig_h = orig_pil.size
            mask_pil = Image.fromarray(bin_mask * 255).resize((orig_w, orig_h), resample=Image.NEAREST)

            mask_save_path = os.path.join(args.output_dir, f"{base_name}_mask.png")
            mask_pil.save(mask_save_path)

            if args.save_overlay:
                overlay_img = create_overlay(orig_pil, np.array(mask_pil) > 128)
                overlay_save_path = os.path.join(args.output_dir, f"{base_name}_overlay.png")
                overlay_img.save(overlay_save_path)

            mask_ratio = float(np.mean(np.array(mask_pil) > 128))

            pred_class_str = "N/A"
            if class_logits is not None:
                class_probs = torch.softmax(class_logits, dim=1).squeeze().cpu().numpy()
                pred_cls = int(np.argmax(class_probs))
                conf = float(class_probs[pred_cls])
                pred_class_str = f"{class_names.get(pred_cls, str(pred_cls))} ({conf:.1%})"

            results_table.append([base_name, f"{mask_ratio:.2%}", pred_class_str, mask_save_path])

    headers = ["Image", "Synthetic Area Ratio", "Pred Label (Confidence)", "Saved Mask"]
    print("\n" + tabulate(results_table, headers=headers, tablefmt="github"))
    print(f"\nAll outputs saved to '{args.output_dir}'")


if __name__ == "__main__":
    main()
