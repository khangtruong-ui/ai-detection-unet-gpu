"""
SAM2 (Segment Anything Model 2) + QLoRA model for AI-generated and tampered synthetic image segmentation.
Applies 4-bit NormalFloat (NF4) / 8-bit quantization via bitsandbytes and Low-Rank Adaptation (LoRA)
via PEFT to Meta's SAM2 foundation model (Hiera backbone), supporting both small and large model variants.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from sid_unet.models.blocks import get_submodule_device_dtype

logger = logging.getLogger("sid_unet.models.sam2")

DEFAULT_SAM2_SMALL_CHECKPOINT = "facebook/sam2-hiera-small"
DEFAULT_SAM2_LARGE_CHECKPOINT = "facebook/sam2-hiera-large"
FALLBACK_SAM2_CHECKPOINT = "facebook/sam2-hiera-tiny"

SAM2_CHECKPOINTS = {
    "small": DEFAULT_SAM2_SMALL_CHECKPOINT,
    "large": DEFAULT_SAM2_LARGE_CHECKPOINT,
    "tiny": FALLBACK_SAM2_CHECKPOINT,
}


def resolve_sam2_checkpoint(
    checkpoint_or_variant: Optional[str] = None,
    variant: Optional[str] = None,
) -> str:
    """
    Resolve checkpoint model path or Hugging Face repository ID.

    Args:
        checkpoint_or_variant: Checkpoint path, HF repo ID, or variant name ('small', 'large', 'tiny').
        variant: Optional explicit variant specifier.

    Returns:
        Validated Hugging Face repository ID or local filesystem path.
    """
    spec = str(variant or checkpoint_or_variant or "small").strip().lower()
    if spec in SAM2_CHECKPOINTS:
        return SAM2_CHECKPOINTS[spec]
    if "large" in spec:
        return DEFAULT_SAM2_LARGE_CHECKPOINT
    if "small" in spec:
        return DEFAULT_SAM2_SMALL_CHECKPOINT
    if "tiny" in spec:
        return FALLBACK_SAM2_CHECKPOINT
    # Explicit custom path or repo ID
    return checkpoint_or_variant or DEFAULT_SAM2_SMALL_CHECKPOINT


def pull_sam2_weights(
    variant: str = "small",
    cache_dir: Optional[str] = None,
    force_download: bool = False,
    token: Optional[str] = None,
) -> str:
    """
    Pull down and cache the weights for a SAM2 model variant from Hugging Face Hub.

    Args:
        variant: 'small', 'large', 'tiny', or explicit Hugging Face repo ID.
        cache_dir: Optional custom Hugging Face cache directory.
        force_download: Whether to force re-download even if already cached.
        token: Optional Hugging Face auth token.

    Returns:
        The resolved model repository ID or cached directory path.
    """
    from transformers import Sam2Model, Sam2Processor

    repo_id = resolve_sam2_checkpoint(variant)
    logger.info(f"Pulling down and verifying SAM2 weights for '{repo_id}'...")

    try:
        Sam2Processor.from_pretrained(
            repo_id,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
        )
        Sam2Model.from_pretrained(
            repo_id,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
        )
        logger.info(f"Successfully integrated SAM2 weights for '{repo_id}'.")
        return repo_id
    except Exception as exc:
        logger.error(f"Failed to pull SAM2 weights for '{repo_id}': {exc}")
        raise exc


class SAM2Segmenter(nn.Module):
    """
    SAM2 model with QLoRA (4-bit/8-bit quantization + LoRA adapters) for binary mask segmentation.

    Args:
        pretrained_model_name_or_path (str): Hugging Face repo or path (default: 'facebook/sam2-hiera-small').
        variant (str): Model variant shorthand ('small', 'large', 'tiny'). Overrides default if given.
        load_in_4bit (bool): Whether to quantize in 4-bit via bitsandbytes (default: False).
        load_in_8bit (bool): Whether to quantize in 8-bit (default: False).
        lora_r (int): LoRA rank dimension (default: 8).
        lora_alpha (int): LoRA scaling factor (default: 16).
        lora_dropout (float): Dropout probability for LoRA layers (default: 0.05).
        lora_target_modules (List[str]): Module names to attach LoRA adapters to (default: ['qkv', 'proj']).
        aux_classifier (bool): Whether to enable auxiliary classification head (default: False).
        num_classes (int): Number of target classes for auxiliary head (default: 3).
        in_channels (int): Input image channels (default: 3).
        out_channels (int): Output mask channels (default: 1).
        target_size (Tuple[int, int]): Native input resolution for SAM2 Hiera ViT (default: (1024, 1024)).
        device (str or torch.device): Target device (default: 'auto').
    """

    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        variant: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        aux_classifier: bool = False,
        num_classes: int = 3,
        in_channels: int = 3,
        out_channels: int = 1,
        target_size: Tuple[int, int] = (1024, 1024),
        use_mask_adapter: bool = True,
        freeze_decoder: bool = False,
        reinit_decoder: bool = False,
        device: Optional[Union[str, torch.device]] = "auto",
        **kwargs: Any,
    ):
        super().__init__()
        self.pretrained_model_name_or_path = resolve_sam2_checkpoint(pretrained_model_name_or_path, variant=variant)
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or ["qkv", "proj"]
        self.aux_classifier = aux_classifier
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.target_size = tuple(target_size)
        self.use_mask_adapter = use_mask_adapter

        # Decoder training and weight re-initialization options
        if "train_decoder" in kwargs:
            self.freeze_decoder = not bool(kwargs.pop("train_decoder"))
        else:
            self.freeze_decoder = bool(freeze_decoder)
        self.reinit_decoder = bool(reinit_decoder)

        # Resolve device
        if device == "auto" or device is None:
            self._target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self._target_device = torch.device(device)
        else:
            self._target_device = device

        self._init_processor_and_model()

        # Trainable mask combination/adaptation head over the 3 candidate masks
        if self.use_mask_adapter:
            self.mask_adapter = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1),
            )
            # Initialize with Xavier uniform and zero bias so predicted logits start centered
            # at the decision boundary (0.0). Avoids negative bias saturation (e.g. -2.0)
            # which forces all predicted masks to 0 and traps training IoU at 0.0000.
            for m in self.mask_adapter.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        else:
            self.mask_adapter = None

        # Auxiliary classification head
        if self.aux_classifier:
            self.classifier_head = nn.Linear(1, self.num_classes)
        else:
            self.classifier_head = None

    def _init_processor_and_model(self):
        """Initialize SAM2 processor and model with optional QLoRA / LoRA adapters."""
        try:
            from transformers import Sam2Model, Sam2Processor, BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise ImportError(
                "SAM2 models require transformers, peft, and bitsandbytes. "
                "Please install them via `pip install 'sid-unet[sam]'`."
            ) from exc

        # 1. Load processor
        self.processor = self._load_processor()

        # 2. Determine quantization configuration
        is_cuda = (self._target_device.type == "cuda") and torch.cuda.is_available()
        bnb_config = None
        fallback_to_gpu_16bit = False

        # If decoder is unfrozen, skip quantizing mask_decoder modules so its parameters remain float and trainable
        skip_modules = ["mask_decoder"] if not self.freeze_decoder else None

        if self.load_in_4bit and is_cuda:
            compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=skip_modules,
            )
            device_map = "auto"
        elif self.load_in_8bit and is_cuda:
            try:
                from sid_unet.utils.compatibility import check_8bit_compatibility
                is_ok, details = check_8bit_compatibility(device=self._target_device)
                if not is_ok:
                    raise RuntimeError(details.get("message", "8-bit environment incompatible"))
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_skip_modules=skip_modules,
                )
                device_map = "auto"
            except Exception as e:
                logger.warning(
                    f"⚠️ SAM2 8-bit quantization unavailable ({e}). "
                    f"Falling back to GPU 16-bit mode."
                )
                bnb_config = None
                device_map = None
                fallback_to_gpu_16bit = True
        else:
            device_map = None

        # 3. Load base Sam2Model
        base_model = self._load_base_model(bnb_config=bnb_config, device_map=device_map)

        # 4. Prepare for k-bit training if quantized, or 16-bit GPU mode if falling back
        if bnb_config is not None:
            base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)
        else:
            if is_cuda and (fallback_to_gpu_16bit or self.load_in_8bit):
                compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
                base_model = base_model.to(dtype=compute_dtype, device=self._target_device)
            for p in base_model.parameters():
                p.requires_grad = False

        # 5. Apply LoRA if rank > 0
        if self.lora_r > 0:
            lora_cfg = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=self.lora_dropout,
                bias="none",
            )
            self.model = get_peft_model(base_model, lora_cfg)
        else:
            self.model = base_model

        # 6. Configure mask decoder parameter training and weight initialization
        decoder_mod = self.get_mask_decoder()
        if decoder_mod is not None:
            if self.reinit_decoder:
                self._reinit_decoder_weights(decoder_mod)
                logger.info("Re-initialized SAM2 mask_decoder weights from scratch.")
            if not self.freeze_decoder:
                for p in decoder_mod.parameters():
                    p.requires_grad = True
                logger.info("Unfroze SAM2 mask_decoder parameters for active training.")
            else:
                for p in decoder_mod.parameters():
                    p.requires_grad = False
                logger.info("Frozen SAM2 mask_decoder parameters.")

        # Pre-compute default full-image bounding box coordinates [0, 0, W, H]
        h_t, w_t = self.target_size
        default_box = torch.tensor([[[0.0, 0.0, float(w_t), float(h_t)]]], dtype=torch.float32)
        self.register_buffer("base_box", default_box, persistent=False)

        logger.info(
            f"Initialized SAM2Segmenter with base='{self.pretrained_model_name_or_path}', "
            f"4bit={self.load_in_4bit}, 8bit={self.load_in_8bit}, r={self.lora_r}, alpha={self.lora_alpha}, "
            f"freeze_decoder={self.freeze_decoder}, reinit_decoder={self.reinit_decoder}"
        )

    def get_mask_decoder(self) -> Optional[nn.Module]:
        """
        Retrieve mask decoder submodule safely whether wrapped with PEFT or base model.
        """
        curr = getattr(self, "model", None)
        if curr is None:
            return None
        if hasattr(curr, "base_model") and hasattr(curr.base_model, "model"):
            curr = curr.base_model.model
        elif hasattr(curr, "model"):
            curr = curr.model
        return getattr(curr, "mask_decoder", None)

    def _reinit_decoder_weights(self, module: nn.Module) -> None:
        """
        Re-initialize mask decoder weights to eliminate salient-object segmentation inductive bias.
        """
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _load_processor(self):
        """Attempt to load processor with fallback handling."""
        from transformers import Sam2Processor
        for cand in [self.pretrained_model_name_or_path, DEFAULT_SAM2_SMALL_CHECKPOINT, FALLBACK_SAM2_CHECKPOINT]:
            try:
                return Sam2Processor.from_pretrained(cand)
            except Exception as e:
                logger.debug(f"Processor loading from '{cand}' failed: {e}")
                continue
        raise RuntimeError(f"Could not load Sam2Processor from '{self.pretrained_model_name_or_path}' or fallbacks.")

    def _load_base_model(self, bnb_config=None, device_map=None):
        """Attempt to load base Sam2Model with fallback handling."""
        from transformers import Sam2Model
        candidates = [self.pretrained_model_name_or_path]
        if DEFAULT_SAM2_SMALL_CHECKPOINT not in candidates:
            candidates.append(DEFAULT_SAM2_SMALL_CHECKPOINT)
        if FALLBACK_SAM2_CHECKPOINT not in candidates:
            candidates.append(FALLBACK_SAM2_CHECKPOINT)

        last_err = None
        for cand in candidates:
            try:
                logger.info(f"Loading base Sam2Model from '{cand}'...")
                load_kwargs: Dict[str, Any] = {}
                if bnb_config is not None:
                    load_kwargs["quantization_config"] = bnb_config
                if device_map is not None:
                    load_kwargs["device_map"] = device_map

                model = Sam2Model.from_pretrained(cand, **load_kwargs)
                self.pretrained_model_name_or_path = cand
                return model
            except Exception as e:
                last_err = e
                logger.warning(f"Could not load base SAM2 model from '{cand}': {e}. Trying fallback...")
                continue

        raise RuntimeError(f"Failed to load Sam2Model from any candidate: {last_err}")

    def forward(
        self,
        x: torch.Tensor,
        input_boxes: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for SAM2 dense segmentation.

        Args:
            x (torch.Tensor): Input images of shape [B, 3, H, W].
            input_boxes (Optional[torch.Tensor]): Bounding box prompts of shape [B, 1, 4] or None for full-image.

        Returns:
            torch.Tensor: Predicted mask logits of shape [B, 1, H, W].
            If aux_classifier is True, returns Tuple[torch.Tensor, torch.Tensor] (mask_logits, cls_logits).
        """
        b_sz, _, orig_h, orig_w = x.shape
        model_device, param_dtype = get_submodule_device_dtype(
            self.model, x.device, torch.float32
        )

        # Resize to SAM2 native resolution if necessary
        if (orig_h, orig_w) != self.target_size:
            x_proc = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)
        else:
            x_proc = x

        # Ensure correct tensor type and device
        if param_dtype in (torch.float16, torch.bfloat16):
            x_proc = x_proc.to(device=model_device, dtype=param_dtype)
        else:
            x_proc = x_proc.to(device=model_device)

        # Prepare batched bounding boxes
        if input_boxes is not None:
            batch_boxes = input_boxes.to(device=model_device, dtype=torch.float32)
        else:
            batch_boxes = self.base_box.repeat(b_sz, 1, 1).to(device=model_device)

        # Forward pass through SAM2 model
        outputs = self.model(
            pixel_values=x_proc,
            input_boxes=batch_boxes,
        )

        # outputs.pred_masks is [B, 1, num_masks, H_m, W_m]
        pred_masks = outputs.pred_masks
        if pred_masks.ndim == 5:
            pred_masks_squeezed = pred_masks.squeeze(1)  # [B, num_masks, H_m, W_m]
        elif pred_masks.ndim == 4:
            pred_masks_squeezed = pred_masks
        else:
            raise RuntimeError(f"Unexpected pred_masks shape from SAM2: {pred_masks.shape}")

        if self.use_mask_adapter and self.mask_adapter is not None:
            # Route candidate masks through trainable adapter to produce dense forgery logits
            adapter_dev, adapter_dtype = get_submodule_device_dtype(self.mask_adapter, x.device, torch.float32)
            mask_logits_raw = self.mask_adapter(pred_masks_squeezed.to(device=adapter_dev, dtype=adapter_dtype))
        elif hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
            weights = torch.softmax(outputs.iou_scores, dim=-1).unsqueeze(-1).unsqueeze(-1)
            mask_logits_raw = (pred_masks * weights).sum(dim=2)  # [B, 1, H_m, W_m]
        else:
            mask_logits_raw = pred_masks_squeezed.mean(dim=1, keepdim=True)

        # Interpolate mask logits back to original image dimensions [B, 1, orig_h, orig_w]
        mask_logits = F.interpolate(
            mask_logits_raw.float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )

        if self.aux_classifier and self.classifier_head is not None:
            if hasattr(outputs, "object_score_logits") and outputs.object_score_logits is not None:
                obj_score = outputs.object_score_logits.float()
                if obj_score.ndim == 3:
                    obj_score = obj_score.squeeze(1)  # [B, 1]
                cls_logits = self.classifier_head(obj_score)
            else:
                # Fallback: global average pool on mask_logits
                pooled = mask_logits.mean(dim=[2, 3])  # [B, 1]
                cls_logits = self.classifier_head(pooled)
            return mask_logits, cls_logits

        return mask_logits

    @torch.no_grad()
    def predict_mask(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Produce binary mask prediction (values 0.0 or 1.0) given input image tensor."""
        self.eval()
        outputs = self.forward(x)
        mask_logits = outputs[0] if isinstance(outputs, tuple) else outputs
        probs = torch.sigmoid(mask_logits)
        return (probs >= threshold).float()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[Union[str, torch.device]] = None,
        override_config: Optional[Union[Dict[str, Any], Any]] = None,
        strict: bool = False,
        return_config: bool = False,
    ) -> Union[SAM2Segmenter, Tuple[SAM2Segmenter, Any]]:
        """Load SAM2Segmenter model from checkpoint file (.pt)."""
        from sid_unet.models.unet import UNet
        return UNet.from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            override_config=override_config,
            strict=strict,
            return_config=return_config,
        )


# Convenient alias
SAM2QLoRA = SAM2Segmenter
