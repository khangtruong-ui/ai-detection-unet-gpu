"""
SAM3-based mask refiner for SID-UNet.
Contrasts UNet predicted mask regions with segment instances from SAM3 (e.g. facebook/sam3)
via spatial joins between detected mask areas and segmented objects, generating refined masks.
Only used during evaluate, cross-eval, and predict phases (never during training).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from scipy.ndimage import label, find_objects

logger = logging.getLogger("sid_unet.sam3_refiner")

_GLOBAL_REFINER_CACHE: Dict[str, Any] = {}


def denormalize_image_to_pil(
    tensor_img: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Image.Image:
    """Convert a normalized [3, H, W] PyTorch tensor back into a PIL Image [H, W, 3]."""
    t = tensor_img.detach().cpu().float().clone()
    if t.ndim == 4 and t.size(0) == 1:
        t = t.squeeze(0)

    for c in range(3):
        t[c] = t[c] * std[c] + mean[c]

    t = torch.clamp(t, 0.0, 1.0)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def extract_mask_bounding_boxes(
    binary_mask: np.ndarray,
    min_pixels: int = 16,
    margin: int = 6,
) -> List[List[float]]:
    """
    Extract bounding boxes [x_min, y_min, x_max, y_max] for all connected
    components in a 2D binary mask larger than min_pixels.
    """
    h, w = binary_mask.shape
    labeled, num_features = label(binary_mask > 0.5)
    if num_features == 0:
        return []

    slices = find_objects(labeled)
    boxes: List[List[float]] = []

    for idx, s in enumerate(slices, 1):
        if s is None:
            continue
        comp_area = int(np.sum(labeled[s] == idx))
        if comp_area < min_pixels:
            continue

        y_min = max(0.0, float(s[0].start - margin))
        y_max = min(float(h), float(s[0].stop + margin))
        x_min = max(0.0, float(s[1].start - margin))
        x_max = min(float(w), float(s[1].stop + margin))

        # Ensure valid box dimensions
        if (x_max - x_min) >= 2.0 and (y_max - y_min) >= 2.0:
            boxes.append([x_min, y_min, x_max, y_max])

    return boxes


class SAMRefiner:
    """
    Refines UNet predicted masks using Meta's Segment Anything Model 3 (facebook/sam3).
    Extracts UNet mask areas, prompts SAM3 to discover precise object boundaries,
    contrasts and joins matching segments, and produces an enhanced output mask.
    """

    def __init__(
        self,
        model_name: str = "facebook/sam3",
        device: Optional[Union[str, torch.device]] = None,
        threshold: float = 0.5,
        score_threshold: float = 0.15,
        min_overlap: float = 0.1,
        min_pixels: int = 16,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.score_threshold = score_threshold
        self.min_overlap = min_overlap
        self.min_pixels = min_pixels

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) or device == "cuda" else "cpu")
        else:
            self.device = device

        logger.info(f"Initializing SAMRefiner with model '{self.model_name}' on device '{self.device}'")
        self._load_model()

    def _load_model(self):
        """Load SAM3 processor and model."""
        try:
            from transformers import Sam3Model, Sam3Processor
            self.processor = Sam3Processor.from_pretrained(self.model_name)
            self.model = Sam3Model.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Successfully loaded {self.model_name} onto {self.device}")
        except Exception as exc:
            logger.error(f"Failed to load SAM3 model '{self.model_name}': {exc}")
            raise exc

    def refine_single_sample(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        unet_mask: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Refine a single UNet predicted mask for an image.

        Args:
            image: PIL Image, RGB numpy array, or normalized [3, H, W] PyTorch tensor.
            unet_mask: 2D numpy array or [1, H, W] / [H, W] PyTorch tensor of probabilities or binary values.

        Returns:
            Tuple of (refined_binary_mask [H, W], change_metrics_dict).
        """
        # Convert image to PIL RGB
        if isinstance(image, torch.Tensor):
            pil_img = denormalize_image_to_pil(image)
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] == 3:  # (3, H, W)
                image = np.transpose(image, (1, 2, 0))
            if image.max() <= 1.0:
                image = (image * 255.0).astype(np.uint8)
            pil_img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Convert unet_mask to 2D binary numpy array
        if isinstance(unet_mask, torch.Tensor):
            u_arr = unet_mask.detach().cpu().float().numpy()
            if u_arr.ndim == 3:
                u_arr = u_arr[0]
        else:
            u_arr = np.asarray(unet_mask, dtype=np.float32)
            if u_arr.ndim == 3:
                u_arr = u_arr[0]

        h, w = u_arr.shape
        orig_binary_mask = (u_arr >= self.threshold).astype(np.float32)
        u_area = float(np.sum(orig_binary_mask))

        # If UNet predicted clean / authentic image (no positive mask areas), retain clean mask
        if u_area < self.min_pixels:
            return orig_binary_mask, {
                "pixels_changed": 0,
                "pixel_change_ratio": 0.0,
                "unet_mask_ratio": float(u_area / (h * w)),
                "refined_mask_ratio": float(u_area / (h * w)),
                "num_components": 0,
                "num_joined_segments": 0,
            }

        # Extract bounding boxes of connected components in UNet mask
        boxes = extract_mask_bounding_boxes(orig_binary_mask, min_pixels=self.min_pixels)
        if not boxes:
            return orig_binary_mask, {
                "pixels_changed": 0,
                "pixel_change_ratio": 0.0,
                "unet_mask_ratio": float(u_area / (h * w)),
                "refined_mask_ratio": float(u_area / (h * w)),
                "num_components": 0,
                "num_joined_segments": 0,
            }

        # Run SAM3 with box prompts
        try:
            inputs = self.processor(
                images=pil_img,
                input_boxes=[boxes],
                return_tensors="pt",
            )
            if hasattr(inputs, "to"):
                inputs = inputs.to(self.device)
            elif isinstance(inputs, dict):
                inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=self.score_threshold,
                target_sizes=[(h, w)],
            )

            sam_masks = results[0]["masks"].detach().cpu().numpy()
            scores = results[0]["scores"].detach().cpu().numpy()
        except Exception as exc:
            logger.warning(f"SAM3 inference encountered error ({exc}), falling back to UNet mask.")
            return orig_binary_mask, {
                "pixels_changed": 0,
                "pixel_change_ratio": 0.0,
                "unet_mask_ratio": float(u_area / (h * w)),
                "refined_mask_ratio": float(u_area / (h * w)),
                "num_components": len(boxes),
                "num_joined_segments": 0,
                "error": str(exc),
            }


        # Join and contrast: find SAM3 segments that overlap/join with UNet mask areas
        refined_mask = np.zeros((h, w), dtype=np.float32)
        joined_count = 0
        covered_unet_mask = np.zeros((h, w), dtype=np.float32)

        if len(sam_masks) > 0:
            for s_idx, s_mask in enumerate(sam_masks):
                s_bin = (s_mask > 0.5).astype(np.float32)
                s_area = float(np.sum(s_bin))
                if s_area == 0:
                    continue

                intersection = float(np.sum(s_bin * orig_binary_mask))
                # Check join criteria: either overlap with SAM segment or with UNet mask
                if (intersection / s_area >= self.min_overlap) or (intersection / max(1.0, u_area) >= self.min_overlap):
                    refined_mask = np.maximum(refined_mask, s_bin)
                    covered_unet_mask = np.maximum(covered_unet_mask, s_bin * orig_binary_mask)
                    joined_count += 1

        # For any UNet mask area not matched by SAM3 segments, preserve original UNet prediction
        unmatched_unet = np.clip(orig_binary_mask - covered_unet_mask, 0.0, 1.0)
        refined_mask = np.maximum(refined_mask, unmatched_unet)

        # Compute change statistics
        diff = np.abs(refined_mask - orig_binary_mask)
        pixels_changed = int(np.sum(diff > 0.5))
        change_ratio = float(pixels_changed / (h * w))
        refined_ratio = float(np.sum(refined_mask) / (h * w))
        unet_ratio = float(u_area / (h * w))

        metrics = {
            "pixels_changed": pixels_changed,
            "pixel_change_ratio": change_ratio,
            "unet_mask_ratio": unet_ratio,
            "refined_mask_ratio": refined_ratio,
            "num_components": len(boxes),
            "num_joined_segments": joined_count,
        }

        return refined_mask, metrics

    def refine_batch(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        mask_logits_or_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Refine a batch of predicted masks from UNet.

        Args:
            images: Batch of PyTorch image tensors [B, 3, H, W] or list of PIL Images.
            mask_logits_or_probs: PyTorch tensor of shape [B, 1, H, W] or [B, H, W].

        Returns:
            Tuple of (refined_mask_logits_or_probs [B, 1, H, W] on same device, list of change_metrics).
        """
        device = mask_logits_or_probs.device
        b_sz = mask_logits_or_probs.size(0)

        # Determine if inputs are logits or probabilities
        probs = torch.sigmoid(mask_logits_or_probs) if (mask_logits_or_probs.min() < 0.0 or mask_logits_or_probs.max() > 1.0) else mask_logits_or_probs

        refined_tensors = []
        metrics_list = []

        for i in range(b_sz):
            img_item = images[i] if isinstance(images, (list, tuple)) else images[i]
            m_item = probs[i]

            ref_mask_np, met = self.refine_single_sample(img_item, m_item)
            metrics_list.append(met)

            # Convert 0/1 refined mask back to tensor with shape [1, H, W]
            ref_t = torch.from_numpy(ref_mask_np).to(device=device, dtype=mask_logits_or_probs.dtype)
            if ref_t.ndim == 2:
                ref_t = ref_t.unsqueeze(0)
            elif ref_t.ndim == 3 and ref_t.size(0) != 1:
                ref_t = ref_t[0:1]
            refined_tensors.append(ref_t)

        refined_batch = torch.stack(refined_tensors, dim=0)
        return refined_batch, metrics_list



def get_sam_refiner(
    model_name: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    threshold: float = 0.5,
) -> Optional[SAMRefiner]:
    """Factory helper to obtain or cache a SAMRefiner instance."""
    if not model_name:
        return None

    cache_key = f"{model_name}_{device}"
    if cache_key in _GLOBAL_REFINER_CACHE:
        return _GLOBAL_REFINER_CACHE[cache_key]

    refiner = SAMRefiner(model_name=model_name, device=device, threshold=threshold)
    _GLOBAL_REFINER_CACHE[cache_key] = refiner
    return refiner
