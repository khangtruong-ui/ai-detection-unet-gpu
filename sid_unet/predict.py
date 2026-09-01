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

from sid_unet.models.unet import UNet, build_model
from sid_unet.utils.config import load_config, DEFAULT_CONFIG, deep_merge, ConfigDict, apply_overrides
from sid_unet.utils.memory import clear_memory_cache, is_oom_error


def parse_args():
    parser = argparse.ArgumentParser(description="Predict AI-generated image mask using trained UNet")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--config", type=str, default=None, help="Path to optional YAML config (defaults to embedded checkpoint config)")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides in key.nested=value format")
    parser.add_argument("--image", type=str, default=None, help="Path to single input image")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory of input images")
    parser.add_argument("--output_dir", type=str, default="predictions", help="Directory to save output masks")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=16, help="Batch size for batch inference")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256], help="Inference resolution [H, W]")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization probability threshold")
    parser.add_argument("--save_overlay", action="store_true", help="Save color overlay visualization")
    parser.add_argument("--device", type=str, default="auto", help="Device ('auto', 'cuda', 'cpu')")
    parser.add_argument(
        "--segment",
        type=str,
        default=None,
        help="Optional segment model for mask refinement (e.g. 'facebook/sam3'). Contrasts UNet mask areas with SAM segments via joins.",
    )
    return parser.parse_args()



def load_model_for_inference(
    checkpoint_path: str,
    device: torch.device,
    config_path: Optional[str] = None,
    overrides: Optional[List[str]] = None,
):
    override_cfg = None
    if config_path:
        override_cfg = load_config(config_path, overrides=overrides)
    elif overrides:
        override_cfg = ConfigDict(apply_overrides({}, overrides))

    return UNet.from_checkpoint(
        checkpoint_path,
        device=device,
        override_config=override_cfg,
        return_config=True,
    )


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


def _predict_chunk(model, chunk_tensors, device):
    """Run model prediction on a batch of tensors with OOM handling."""
    chunk_inp = torch.cat(chunk_tensors, dim=0).to(device)
    outputs = model(chunk_inp)
    if isinstance(outputs, tuple):
        mask_logits, class_logits = outputs
    else:
        mask_logits, class_logits = outputs, None
    return mask_logits, class_logits


def main():
    args = parse_args()

    if not args.image and not args.input_dir:
        raise ValueError("Must provide either --image or --input_dir.")

    dev_str = args.device
    if dev_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_str)

    model, config = load_model_for_inference(
        args.checkpoint,
        device,
        config_path=args.config,
        overrides=args.override,
    )
    target_size = (args.image_size[0], args.image_size[1])

    image_paths: List[str] = []
    if args.image:
        image_paths.append(args.image)
    if args.input_dir:
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
            image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))

    os.makedirs(args.output_dir, exist_ok=True)
    class_names = {0: "Real", 1: "Fully AI", 2: "Partially AI (Tampered)"}

    refiner = None
    if args.segment:
        from sid_unet.models.sam3_refiner import get_sam_refiner
        print(f"Loading Segment Mask Refiner '{args.segment}' on device: {device}...")
        refiner = get_sam_refiner(args.segment, device=device, threshold=args.threshold)
        print(f"Segment Mask Refinement active ({args.segment}).")

    results_table = []
    batch_size = max(1, args.batch_size)

    print(f"Running inference on {len(image_paths)} image(s) using device: {device} (batch_size={batch_size})...")
    clear_memory_cache(device)

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            chunk_paths = image_paths[i : i + batch_size]
            preprocessed = [preprocess_image(p, target_size) for p in chunk_paths]
            chunk_tensors = [t for t, _ in preprocessed]
            orig_pils = [p for _, p in preprocessed]

            try:
                mask_logits, class_logits = _predict_chunk(model, chunk_tensors, device)
            except Exception as exc:
                if is_oom_error(exc):
                    print(f"⚠️ OOM during batch inference of {len(chunk_paths)} images. Falling back to single-image mode...")
                    clear_memory_cache(device)
                    # Process individually
                    mask_logits_list = []
                    class_logits_list = []
                    for t in chunk_tensors:
                        m_l, c_l = _predict_chunk(model, [t], device)
                        mask_logits_list.append(m_l)
                        if c_l is not None:
                            class_logits_list.append(c_l)
                    mask_logits = torch.cat(mask_logits_list, dim=0)
                    class_logits = torch.cat(class_logits_list, dim=0) if class_logits_list else None
                else:
                    raise exc

            probs = torch.sigmoid(mask_logits).cpu().numpy()
            if class_logits is not None:
                class_probs = torch.softmax(class_logits, dim=1).cpu().numpy()
            else:
                class_probs = None

            for idx, img_path in enumerate(chunk_paths):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                orig_pil = orig_pils[idx]
                sample_prob = probs[idx, 0] if probs.ndim == 4 else probs[idx]
                bin_mask = (sample_prob >= args.threshold).astype(np.uint8)
                raw_mask_ratio = float(np.mean(bin_mask > 0))

                if refiner is not None:
                    refined_mask_np, change_met = refiner.refine_single_sample(orig_pil, bin_mask)
                    final_mask_np = (refined_mask_np >= 0.5).astype(np.uint8)
                    change_ratio_str = f"{change_met['pixel_change_ratio']:.2%}"
                else:
                    final_mask_np = bin_mask
                    change_ratio_str = "N/A"

                # Resize predicted mask back to original image dimensions
                orig_w, orig_h = orig_pil.size
                mask_pil = Image.fromarray(final_mask_np * 255).resize((orig_w, orig_h), resample=Image.NEAREST)

                mask_save_path = os.path.join(args.output_dir, f"{base_name}_mask.png")
                mask_pil.save(mask_save_path)

                if args.save_overlay:
                    overlay_img = create_overlay(orig_pil, np.array(mask_pil) > 128)
                    overlay_save_path = os.path.join(args.output_dir, f"{base_name}_overlay.png")
                    overlay_img.save(overlay_save_path)

                mask_ratio = float(np.mean(np.array(mask_pil) > 128))

                pred_class_str = "N/A"
                if class_probs is not None:
                    sample_cls_probs = class_probs[idx]
                    pred_cls = int(np.argmax(sample_cls_probs))
                    conf = float(sample_cls_probs[pred_cls])
                    pred_class_str = f"{class_names.get(pred_cls, str(pred_cls))} ({conf:.1%})"

                if refiner is not None:
                    results_table.append([base_name, f"{raw_mask_ratio:.2%}", f"{mask_ratio:.2%}", change_ratio_str, pred_class_str, mask_save_path])
                else:
                    results_table.append([base_name, f"{mask_ratio:.2%}", pred_class_str, mask_save_path])

    clear_memory_cache(device)

    if refiner is not None:
        headers = ["Image", "UNet Area", "Refined Area", "Pixel Change Ratio", "Pred Label (Confidence)", "Saved Mask"]
    else:
        headers = ["Image", "Synthetic Area Ratio", "Pred Label (Confidence)", "Saved Mask"]

    print("\n" + tabulate(results_table, headers=headers, tablefmt="github"))
    print(f"\nAll outputs saved to '{args.output_dir}'")
    return results_table



def cli_main():
    import sys
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    cli_main()
