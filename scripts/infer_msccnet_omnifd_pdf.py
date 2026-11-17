"""
Run inference with MSCCNet and OmniFD on real dataset images and generate
a detailed 8-column, multi-page PDF inspection report.

Usage:
    python scripts/infer_msccnet_omnifd_pdf.py \
        --output-pdf docs/msccnet_omnifd_inference_report.pdf \
        --num-samples 4
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from sid_unet.metrics.segmentation import compute_binary_metrics
from sid_unet.models.msccnet import MSCCNet
from sid_unet.models.omnifd import OMNIFD_GDRIVE_FOLDER_URL, OmniFD
from sid_unet.postprocessing import MaskPostProcessor
from sid_unet.utils.plotting import plot_detailed_inference_pdf


def _fetch_dataset_samples(num_imd: int = 2, num_autosplice: int = 2, num_casia_auth: int = 1) -> List[Dict[str, Any]]:
    """Stream representative tampered and authentic images with masks from Hugging Face datasets."""
    from datasets import load_dataset

    samples: List[Dict[str, Any]] = []

    def _collect(
        ds_name: str,
        split: str,
        limit: int,
        tag: str,
        min_area: float = 0.005,
        max_area: float = 1.0,
        step_skip: int = 1,
    ) -> None:
        if limit <= 0:
            return
        ds_short = ds_name.split("/")[-1]
        ds = load_dataset(ds_name, split=split, streaming=True)
        count = 0
        for idx, ex in enumerate(ds):
            if idx % step_skip != 0:
                continue
            img = ex.get("image")
            mask = ex.get("mask")
            if img is None or mask is None or not isinstance(img, Image.Image):
                continue
            img_rgb = img.convert("RGB")
            mask_arr = (np.array(mask.convert("L"), dtype=np.float32) >= 128.0).astype(np.float32)
            area = float(mask_arr.mean())
            if area < min_area or area > max_area:
                continue
            samples.append(
                {
                    "pil_image": img_rgb,
                    "gt_mask": mask_arr,
                    "label": ex.get("label", 2 if area > 0 else 0),
                    "image_id": f"{tag}_{idx}",
                    "dataset": ds_short,
                }
            )
            count += 1
            if count >= limit:
                break

    _collect("KhangTruong/IMD2020", "test", num_imd, "imd", min_area=0.005)
    _collect("KhangTruong/AutoSplice", "cross_test", num_autosplice, "autosplice", min_area=0.05, step_skip=3)
    _collect("KhangTruong/Casia-v2.0", "cross_test", num_casia_auth, "casia_au", min_area=0.0, max_area=0.0)
    return samples


def _preprocess_for_model(pil_img: Image.Image, size: Tuple[int, int]) -> torch.Tensor:
    h, w = size
    resized = pil_img.resize((w, h), resample=Image.BILINEAR)
    t = TF.to_tensor(resized)
    return TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0)


def run_inference_and_pdf(
    output_pdf: str = "docs/msccnet_omnifd_inference_report.pdf",
    num_samples_per_dataset: int = 2,
    threshold: float = 0.5,
    device_str: str = "auto",
) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_str == "auto" else torch.device(device_str)
    print(f"Running MSCCNet & OmniFD inference on device={device}")

    raw_samples = _fetch_dataset_samples(
        num_imd=num_samples_per_dataset,
        num_autosplice=num_samples_per_dataset,
        num_casia_auth=1,
    )
    if not raw_samples:
        raise RuntimeError("No samples fetched for inference.")
    print(f"Fetched {len(raw_samples)} evaluation images.")

    models_spec: List[Tuple[str, torch.nn.Module, Tuple[int, int], str]] = [
        (
            "OmniFD (Swin-T ForgeryNet-50k)",
            OmniFD(variant="swin-t", checkpoint_name="omnifd_swin-t_50k_forgerynet", pretrained=True).to(device).eval(),
            (256, 256),
            "omnifd",
        ),
        (
            "OmniFD (Swin-S ForgeryNet-50k)",
            OmniFD(variant="swin-s", checkpoint_name="omnifd_swin-s_50k_forgerynet", pretrained=True).to(device).eval(),
            (256, 256),
            "omnifd",
        ),
        (
            "OmniFD (Swin-S FF++-Spatial-20k)",
            OmniFD(variant="swin-s", checkpoint_name="omnifd_swin-s_20k_ff_spatial", pretrained=True).to(device).eval(),
            (299, 299),
            "omnifd",
        ),
        (
            "MSCCNet (ResNet-50 ImageNet-BB)",
            MSCCNet(backbone_depth=50, pretrained=True).to(device).eval(),
            (512, 512),
            "msccnet",
        ),
    ]

    postprocessor = MaskPostProcessor(enabled=True, min_area=64, fill_holes=True, max_hole_size=256, morphology="open_close")

    samples_data: List[Dict[str, Any]] = []
    for s in raw_samples:
        pil_img = s["pil_image"]
        gt_mask = s["gt_mask"]
        disp_size = (256, 256)
        disp_img = np.array(pil_img.resize(disp_size, resample=Image.BILINEAR), dtype=np.uint8)
        gt_disp = np.array(
            Image.fromarray((gt_mask * 255).astype(np.uint8)).resize(disp_size, resample=Image.NEAREST),
            dtype=np.float32,
        ) / 255.0

        entry: Dict[str, Any] = {
            "image": disp_img,
            "gt_mask": gt_disp,
            "label": s["label"],
            "image_id": s["image_id"],
            "dataset": s["dataset"],
            "models": {},
        }

        for m_name, model, inp_hw, m_kind in models_spec:
            inp_t = _preprocess_for_model(pil_img, inp_hw).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(inp_t, return_dict=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - t0) * 1000.0

            prob_hw = out["pred_mask"].float().squeeze().cpu().numpy()
            prob_disp = np.array(
                Image.fromarray((np.clip(prob_hw, 0.0, 1.0) * 255).astype(np.uint8)).resize(disp_size, resample=Image.BILINEAR),
                dtype=np.float32,
            ) / 255.0
            bin_disp = (prob_disp >= threshold).astype(np.float32)
            met = compute_binary_metrics(bin_disp, gt_disp, threshold=threshold)

            cls_probs = None
            if "pred_label_probs" in out and out["pred_label_probs"] is not None:
                cls_probs = out["pred_label_probs"].float().squeeze(0).cpu().tolist()

            if m_kind == "msccnet" and "fres_map_prob" in out:
                fres_hw = out["fres_map_prob"].float().squeeze().cpu().numpy()
                aux_map = np.array(
                    Image.fromarray((np.clip(fres_hw, 0.0, 1.0) * 255).astype(np.uint8)).resize(disp_size, resample=Image.BILINEAR),
                    dtype=np.float32,
                ) / 255.0
                aux_label = "MSCC Spectral Map"
                aux_cmap = "viridis"
                aux_extra = f"μ:{aux_map.mean():.2f} max:{aux_map.max():.2f}"
            else:
                post_m, _ = postprocessor.process_single(bin_disp)
                aux_map = post_m
                aux_label = "Post-Processed Mask"
                aux_cmap = "gray"
                post_met = compute_binary_metrics(post_m, gt_disp, threshold=threshold)
                aux_extra = f"Post IoU:{post_met['iou']:.3f}"

            entry["models"][m_name] = {
                "pred_mask": bin_disp,
                "prob_map": prob_disp,
                "aux_map": aux_map,
                "aux_label": aux_label,
                "aux_cmap": aux_cmap,
                "aux_metric_str": aux_extra,
                "cls_probs": cls_probs,
                "inference_ms": infer_ms,
                "iou": met["iou"],
                "dice": met["dice"],
                "pixel_acc": met["pixel_acc"],
                "precision": met["precision"],
                "recall": met["recall"],
            }
            print(
                f"  [{s['image_id']}] {m_name}: IoU={met['iou']:.4f}, F1={met['dice']:.4f}, "
                f"PAcc={met['pixel_acc']:.4f}, P(Fake)={cls_probs[1] if cls_probs else 'N/A'}, {infer_ms:.1f}ms"
            )

        samples_data.append(entry)

    model_summaries = [
        {
            "model": "OmniFD (Swin-T ForgeryNet-50k)",
            "paper": "arXiv:2512.01128 / haotianll/OmniFD",
            "backbone": "Omnivore Swin-T 3D",
            "detection": "ImageClsHead (2-cls)",
            "localization": "SpatialHead (Interaction)",
            "weights": "GDrive: omnifd_swin-t_50k_forgerynet.pth",
        },
        {
            "model": "OmniFD (Swin-S ForgeryNet-50k)",
            "paper": "arXiv:2512.01128 / haotianll/OmniFD",
            "backbone": "Omnivore Swin-S 3D",
            "detection": "ImageClsHead (2-cls)",
            "localization": "SpatialHead (Interaction)",
            "weights": "GDrive: omnifd_swin-s_50k_forgerynet.pth",
        },
        {
            "model": "OmniFD (Swin-S FF++-Spatial-20k)",
            "paper": "arXiv:2512.01128 / haotianll/OmniFD",
            "backbone": "Omnivore Swin-S 3D",
            "detection": "N/A (spatial-only ckpt)",
            "localization": "SpatialHead (FF++)",
            "weights": "GDrive: omnifd_swin-s_20k_ff_spatial.pth",
        },
        {
            "model": "MSCCNet (ResNet-50 ImageNet-BB)",
            "paper": "ACM TOMM 2025 / miaoct/MSCCNet",
            "backbone": "Dilated ResNet-50 v1c (os=8)",
            "detection": "MSCCClassifier (2-cls)",
            "localization": "MSCC (2D DCT + GCN) + Decoder",
            "weights": "ImageNet resnet50_v1c BB (no upstream task ckpt)",
        },
    ]

    model_names = [m[0] for m in models_spec]
    out_paths = [output_pdf]
    if os.path.abspath(output_pdf) != os.path.abspath("outputs/illustrations/msccnet_omnifd_inference_report.pdf"):
        out_paths.append("outputs/illustrations/msccnet_omnifd_inference_report.pdf")

    for target_p in out_paths:
        plot_detailed_inference_pdf(
            samples_data=samples_data,
            model_names=model_names,
            output_path=target_p,
            title="MSCCNet & OmniFD — 8-Column Detailed Inference & Localization Report",
            model_summaries=model_summaries,
            rows_per_page=3,
            dpi=180,
        )
        print(f"Generated detailed 8-column PDF: {target_p}")

    return output_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MSCCNet and OmniFD inference and generate detailed PDF.")
    parser.add_argument("--output-pdf", type=str, default="docs/msccnet_omnifd_inference_report.pdf")
    parser.add_argument("--num-samples-per-dataset", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    run_inference_and_pdf(
        output_pdf=args.output_pdf,
        num_samples_per_dataset=args.num_samples_per_dataset,
        threshold=args.threshold,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
