"""
Run zero-random-init inference on KhangTruong/face-forgery-domain-eval and produce:
  1. Full 3-split quantitative metrics JSON (model1_in_domain, model2_in_domain, out_of_domain)
  2. Multi-page, 8-column detailed visual + quantitative PDF report

Strict zero-random-init policy:
  - All evaluated OmniFD models load verified Google Drive checkpoints with allow_random_init=False.
  - MSCCNet raises RuntimeError unless --msccnet-ckpt is supplied (never falls back to random init).

Usage:
    python scripts/infer_msccnet_omnifd_pdf.py \
        --dataset-repo KhangTruong/face-forgery-domain-eval \
        --output-pdf docs/msccnet_omnifd_inference_report.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from sid_unet.metrics.segmentation import compute_binary_metrics
from sid_unet.models.msccnet import MSCCNet
from sid_unet.models.omnifd import OmniFD
from sid_unet.postprocessing import MaskPostProcessor
from sid_unet.utils.plotting import plot_detailed_inference_pdf


SPLIT_LABELS = {
    "model1_in_domain": "model1_in_domain (MSCCNet FF++ C23)",
    "model2_in_domain": "model2_in_domain (OmniFD FF++ C23)",
    "out_of_domain": "out_of_domain (CelebA + Celeb-DF-v2)",
}


def _fetch_domain_samples(
    dataset_repo: str,
    local_dir: Optional[str] = None,
    splits: Tuple[str, ...] = ("model1_in_domain", "model2_in_domain", "out_of_domain"),
) -> List[Dict[str, Any]]:
    """Load all splits from the domain-aligned human face dataset."""
    from datasets import load_dataset, load_from_disk

    samples: List[Dict[str, Any]] = []
    disk_ds = load_from_disk(local_dir) if local_dir and os.path.isdir(local_dir) else None

    for sp in splits:
        ds = disk_ds[sp] if disk_ds is not None else load_dataset(dataset_repo, split=sp)
        for idx, ex in enumerate(ds):
            img = ex["image"]
            mask = ex["mask"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.asarray(img))
            if not isinstance(mask, Image.Image):
                mask = Image.fromarray(np.asarray(mask))
            img_rgb = img.convert("RGB")
            mask_arr = (np.asarray(mask.convert("L"), dtype=np.float32) >= 128.0).astype(np.float32)
            raw_lbl = int(ex.get("label", 1 if mask_arr.mean() > 0 else 0))
            # plot_detailed_inference_pdf uses {0: Real, 2: Tampered}
            vis_label = 2 if raw_lbl > 0 else 0
            samples.append(
                {
                    "pil_image": img_rgb,
                    "gt_mask": mask_arr,
                    "gt_binary_label": raw_lbl,
                    "label": vis_label,
                    "image_id": str(ex.get("image_id", f"{sp}_{idx}")),
                    "dataset": str(ex.get("source_dataset", sp)),
                    "manipulation_type": str(ex.get("manipulation_type", "")),
                    "domain_split": SPLIT_LABELS.get(sp, sp),
                    "split_key": sp,
                }
            )
    return samples


def _preprocess_for_model(pil_img: Image.Image, size: Tuple[int, int]) -> torch.Tensor:
    h, w = size
    resized = pil_img.resize((w, h), resample=Image.BILINEAR)
    t = TF.to_tensor(resized)
    return TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0)


def _resolve_local_omnifd(ckpt_name: str, filename: str) -> Optional[str]:
    for base in ("/tmp/omnifd_weights", os.path.expanduser("~/.cache/omnifd")):
        cand = os.path.join(base, filename)
        if os.path.isfile(cand):
            return cand
    return None


def _select_balanced_pdf_subset(samples: List[Dict[str, Any]], per_split: int = 4) -> List[Dict[str, Any]]:
    """Select balanced real + manipulated faces from each domain split for visual PDF pages."""
    if per_split <= 0:
        return samples
    selected: List[Dict[str, Any]] = []
    by_split: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        by_split.setdefault(s["split_key"], []).append(s)
    for sp, items in by_split.items():
        reals = [x for x in items if x["gt_binary_label"] == 0]
        fakes = [x for x in items if x["gt_binary_label"] == 1]
        n_real = max(1, per_split // 2)
        n_fake = max(1, per_split - n_real)
        selected.extend(reals[:n_real])
        selected.extend(fakes[:n_fake])
    return selected


def run_inference_and_pdf(
    dataset_repo: str = "KhangTruong/face-forgery-domain-eval",
    local_dataset_dir: Optional[str] = "outputs/face_domain_dataset",
    output_pdf: str = "docs/msccnet_omnifd_inference_report.pdf",
    output_json: str = "docs/msccnet_omnifd_domain_metrics.json",
    msccnet_ckpt: Optional[str] = None,
    pdf_samples_per_split: int = 4,
    threshold: float = 0.5,
    device_str: str = "auto",
) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_str == "auto" else torch.device(device_str)
    print(f"Running strict zero-random-init human-face inference on device={device}")

    raw_samples = _fetch_domain_samples(dataset_repo=dataset_repo, local_dir=local_dataset_dir)
    if not raw_samples:
        raise RuntimeError(f"No samples fetched from {dataset_repo}.")
    print(f"Loaded {len(raw_samples)} domain-evaluation human face samples.")

    # Verify strict zero-random-init enforcement on MSCCNet
    msccnet_blocked_reason: Optional[str] = None
    if msccnet_ckpt is None:
        try:
            _ = MSCCNet(backbone_depth=50, pretrained=True)
        except RuntimeError as err:
            msccnet_blocked_reason = str(err)
            print(f"[Verified Strict Guard] MSCCNet blocked without random init: {msccnet_blocked_reason}")

    omnifd_specs = [
        (
            "OmniFD (Swin-S FF++-Spatial-20k)",
            "swin-s",
            "omnifd_swin-s_20k_ff_spatial",
            "omnifd_swin-s_20k_ff_spatial.pth",
            False,
            True,
            (299, 299),
            "SpatialHead (FF++)",
            "N/A (spatial-only)",
        ),
        (
            "OmniFD (Swin-S FF++-Image-20k)",
            "swin-s",
            "omnifd_swin-s_20k_ff_image",
            "omnifd_swin-s_20k_ff_image.pth",
            True,
            False,
            (224, 224),
            "N/A (image-only)",
            "ImageClsHead (FF++)",
        ),
        (
            "OmniFD (Swin-S FMLD-Spatial-20k)",
            "swin-s",
            "omnifd_swin-s_20k_fmld_spatial",
            "omnifd_swin-s_20k_fmld_spatial.pth",
            False,
            True,
            (299, 299),
            "SpatialHead (FMLD)",
            "N/A (spatial-only)",
        ),
        (
            "OmniFD (Swin-T ForgeryNet-50k)",
            "swin-t",
            "omnifd_swin-t_50k_forgerynet",
            "omnifd_swin-t_50k_forgerynet.pth",
            True,
            True,
            (224, 224),
            "SpatialHead (ForgeryNet)",
            "ImageClsHead (ForgeryNet)",
        ),
        (
            "OmniFD (Swin-S ForgeryNet-50k)",
            "swin-s",
            "omnifd_swin-s_50k_forgerynet",
            "omnifd_swin-s_50k_forgerynet.pth",
            True,
            True,
            (224, 224),
            "SpatialHead (ForgeryNet)",
            "ImageClsHead (ForgeryNet)",
        ),
    ]

    loaded_models: List[Tuple[str, torch.nn.Module, Tuple[int, int], str, bool]] = []
    model_summaries: List[Dict[str, Any]] = []

    for m_name, variant, ckpt_name, ckpt_fn, has_img, has_spa, inp_hw, loc_desc, det_desc in omnifd_specs:
        local_ckpt = _resolve_local_omnifd(ckpt_name, ckpt_fn)
        m = OmniFD(
            variant=variant,
            checkpoint_name=ckpt_name,
            checkpoint_path=local_ckpt,
            pretrained=True,
            allow_random_init=False,
            has_image_head=has_img,
            has_spatial_head=has_spa,
        ).to(device).eval()
        loaded_models.append((m_name, m, inp_hw, "omnifd", has_spa))
        model_summaries.append(
            {
                "model": m_name,
                "paper": "arXiv:2512.01128 / haotianll/OmniFD",
                "backbone": f"Omnivore {variant.upper()} 3D",
                "detection": det_desc,
                "localization": loc_desc,
                "weights": f"GDrive (zero-rand): {ckpt_fn}",
            }
        )

    if msccnet_ckpt is not None:
        m_mscc = MSCCNet(
            backbone_depth=50,
            pretrained=True,
            checkpoint_path=msccnet_ckpt,
            allow_random_init=False,
        ).to(device).eval()
        loaded_models.append(("MSCCNet (ResNet-50)", m_mscc, (512, 512), "msccnet", True))
        model_summaries.append(
            {
                "model": "MSCCNet (ResNet-50)",
                "paper": "ACM TOMM 2025 / miaoct/MSCCNet",
                "backbone": "Dilated ResNet-50 v1c (os=8)",
                "detection": "MSCCClassifier (2-cls)",
                "localization": "MSCC (2D DCT + GCN)",
                "weights": f"User ckpt: {os.path.basename(msccnet_ckpt)}",
            }
        )
    else:
        model_summaries.append(
            {
                "model": "MSCCNet (ResNet-50 / Model 1)",
                "paper": "ACM TOMM 2025 / miaoct/MSCCNet",
                "backbone": "Dilated ResNet-50 v1c (os=8)",
                "detection": "MSCCClassifier (2-cls)",
                "localization": "MSCC (2D DCT + GCN)",
                "weights": "BLOCKED (zero-random-init; no upstream task ckpt released)",
            }
        )

    postprocessor = MaskPostProcessor(enabled=True, min_area=64, fill_holes=True, max_hole_size=256, morphology="open_close")
    samples_data: List[Dict[str, Any]] = []
    disp_size = (256, 256)

    for s in raw_samples:
        pil_img = s["pil_image"]
        gt_mask = s["gt_mask"]
        disp_img = np.array(pil_img.resize(disp_size, resample=Image.BILINEAR), dtype=np.uint8)
        gt_disp = (
            np.array(
                Image.fromarray((gt_mask * 255).astype(np.uint8)).resize(disp_size, resample=Image.NEAREST),
                dtype=np.float32,
            )
            / 255.0
        )

        entry: Dict[str, Any] = {
            "image": disp_img,
            "gt_mask": gt_disp,
            "label": s["label"],
            "gt_binary_label": s["gt_binary_label"],
            "image_id": s["image_id"],
            "dataset": f"{s['dataset']} ({s['manipulation_type']})",
            "domain_split": s["domain_split"],
            "split_key": s["split_key"],
            "models": {},
        }

        for m_name, model, inp_hw, m_kind, has_spa in loaded_models:
            inp_t = _preprocess_for_model(pil_img, inp_hw).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(inp_t, return_dict=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_ms = (time.perf_counter() - t0) * 1000.0

            if has_spa:
                prob_hw = out["pred_mask"].float().squeeze().cpu().numpy()
                prob_disp = (
                    np.array(
                        Image.fromarray((np.clip(prob_hw, 0.0, 1.0) * 255).astype(np.uint8)).resize(disp_size, resample=Image.BILINEAR),
                        dtype=np.float32,
                    )
                    / 255.0
                )
                bin_disp = (prob_disp >= threshold).astype(np.float32)
            else:
                prob_disp = np.zeros(disp_size, dtype=np.float32)
                bin_disp = np.zeros(disp_size, dtype=np.float32)

            met = compute_binary_metrics(bin_disp, gt_disp, threshold=threshold)

            cls_probs = None
            if "pred_label_probs" in out and out["pred_label_probs"] is not None:
                cls_probs = out["pred_label_probs"].float().squeeze(0).cpu().tolist()

            if m_kind == "msccnet" and "fres_map_prob" in out:
                fres_hw = out["fres_map_prob"].float().squeeze().cpu().numpy()
                aux_map = (
                    np.array(
                        Image.fromarray((np.clip(fres_hw, 0.0, 1.0) * 255).astype(np.uint8)).resize(disp_size, resample=Image.BILINEAR),
                        dtype=np.float32,
                    )
                    / 255.0
                )
                aux_label = "MSCC Spectral Map"
                aux_cmap = "viridis"
                aux_extra = f"μ:{aux_map.mean():.2f} max:{aux_map.max():.2f}"
            elif has_spa:
                post_m, _ = postprocessor.process_single(bin_disp)
                aux_map = post_m
                aux_label = "Post-Processed Mask"
                aux_cmap = "gray"
                post_met = compute_binary_metrics(post_m, gt_disp, threshold=threshold)
                aux_extra = f"Post IoU:{post_met['iou']:.3f}"
            else:
                aux_map = np.zeros(disp_size, dtype=np.float32)
                aux_label = "N/A (Image-Only Ckpt)"
                aux_cmap = "gray"
                aux_extra = "Image-cls head only"

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
                "has_spatial_head": has_spa,
            }

        samples_data.append(entry)

    # Aggregate full metrics across all splits and models -> save JSON
    model_names = [m[0] for m in loaded_models]
    summary_json: Dict[str, Any] = {
        "dataset_repo": dataset_repo,
        "total_samples": len(samples_data),
        "msccnet_status": "loaded" if msccnet_ckpt else "blocked_strict_zero_random_init",
        "msccnet_blocked_reason": msccnet_blocked_reason,
        "splits": {},
        "overall": {},
    }

    for sp_key in ("model1_in_domain", "model2_in_domain", "out_of_domain"):
        sp_samples = [x for x in samples_data if x["split_key"] == sp_key]
        sp_dict: Dict[str, Any] = {"n_samples": len(sp_samples), "models": {}}
        for m_name in model_names:
            m_items = [(x, x["models"][m_name]) for x in sp_samples]
            cls_hits = [
                int((e["cls_probs"][1] >= 0.5) == (x["gt_binary_label"] == 1))
                for x, e in m_items
                if e["cls_probs"] is not None
            ]
            # Tampered-only localization IoU / F1 (avoids inflated 0/0=1 on real faces if needed)
            fake_items = [(x, e) for x, e in m_items if x["gt_binary_label"] == 1]
            sp_dict["models"][m_name] = {
                "mean_iou": float(np.mean([e["iou"] for _, e in m_items])),
                "mean_dice_f1": float(np.mean([e["dice"] for _, e in m_items])),
                "mean_pixel_acc": float(np.mean([e["pixel_acc"] for _, e in m_items])),
                "tampered_only_iou": float(np.mean([e["iou"] for _, e in fake_items])) if fake_items else None,
                "tampered_only_f1": float(np.mean([e["dice"] for _, e in fake_items])) if fake_items else None,
                "image_cls_acc": float(np.mean(cls_hits)) if cls_hits else None,
                "mean_p_fake_on_real": (
                    float(np.mean([e["cls_probs"][1] for x, e in m_items if x["gt_binary_label"] == 0 and e["cls_probs"]]))
                    if any(x["gt_binary_label"] == 0 and e["cls_probs"] for x, e in m_items)
                    else None
                ),
                "mean_p_fake_on_tampered": (
                    float(np.mean([e["cls_probs"][1] for x, e in m_items if x["gt_binary_label"] == 1 and e["cls_probs"]]))
                    if any(x["gt_binary_label"] == 1 and e["cls_probs"] for x, e in m_items)
                    else None
                ),
                "mean_latency_ms": float(np.mean([e["inference_ms"] for _, e in m_items])),
            }
        summary_json["splits"][sp_key] = sp_dict

    for m_name in model_names:
        all_items = [(x, x["models"][m_name]) for x in samples_data]
        cls_hits = [
            int((e["cls_probs"][1] >= 0.5) == (x["gt_binary_label"] == 1))
            for x, e in all_items
            if e["cls_probs"] is not None
        ]
        fake_items = [(x, e) for x, e in all_items if x["gt_binary_label"] == 1]
        summary_json["overall"][m_name] = {
            "mean_iou": float(np.mean([e["iou"] for _, e in all_items])),
            "mean_dice_f1": float(np.mean([e["dice"] for _, e in all_items])),
            "mean_pixel_acc": float(np.mean([e["pixel_acc"] for _, e in all_items])),
            "tampered_only_iou": float(np.mean([e["iou"] for _, e in fake_items])) if fake_items else None,
            "tampered_only_f1": float(np.mean([e["dice"] for _, e in fake_items])) if fake_items else None,
            "image_cls_acc": float(np.mean(cls_hits)) if cls_hits else None,
            "mean_latency_ms": float(np.mean([e["inference_ms"] for _, e in all_items])),
        }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as fp:
        json.dump(summary_json, fp, indent=2)
    print(f"Saved domain inference metrics JSON: {output_json}")
    print(json.dumps(summary_json, indent=2))

    # Build multi-page PDF on balanced domain subset (or all if pdf_samples_per_split <= 0)
    pdf_samples = _select_balanced_pdf_subset(samples_data, per_split=pdf_samples_per_split)
    out_paths = [output_pdf]
    alt_pdf = "outputs/illustrations/msccnet_omnifd_inference_report.pdf"
    if os.path.abspath(output_pdf) != os.path.abspath(alt_pdf):
        out_paths.append(alt_pdf)

    for target_p in out_paths:
        plot_detailed_inference_pdf(
            samples_data=pdf_samples,
            model_names=model_names,
            output_path=target_p,
            title="MSCCNet & OmniFD — Zero-Random-Init Human-Face Domain Evaluation (8-Column Report)",
            model_summaries=model_summaries,
            rows_per_page=3,
            dpi=180,
        )
        print(f"Generated detailed 8-column PDF ({len(pdf_samples)} faces x {len(model_names)} models): {target_p}")

    return output_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-random-init face-domain inference and generate PDF.")
    parser.add_argument("--dataset-repo", type=str, default="KhangTruong/face-forgery-domain-eval")
    parser.add_argument("--local-dataset-dir", type=str, default="outputs/face_domain_dataset")
    parser.add_argument("--output-pdf", type=str, default="docs/msccnet_omnifd_inference_report.pdf")
    parser.add_argument("--output-json", type=str, default="docs/msccnet_omnifd_domain_metrics.json")
    parser.add_argument("--msccnet-ckpt", type=str, default=None)
    parser.add_argument("--pdf-samples-per-split", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    run_inference_and_pdf(
        dataset_repo=args.dataset_repo,
        local_dataset_dir=args.local_dataset_dir,
        output_pdf=args.output_pdf,
        output_json=args.output_json,
        msccnet_ckpt=args.msccnet_ckpt,
        pdf_samples_per_split=args.pdf_samples_per_split,
        threshold=args.threshold,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
