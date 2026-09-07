"""
SAM3 + QLoRA model for AI-generated and tampered synthetic image segmentation.
Applies 4-bit NormalFloat (NF4) quantization via bitsandbytes and Low-Rank Adaptation (LoRA)
via PEFT to Meta's SAM3 foundation model, enabling parameter-efficient and memory-efficient fine-tuning.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("sid_unet.models.sam3_qlora")

DEFAULT_SAM3_CHECKPOINT = "jetjodh/sam3"
FALLBACK_SAM3_CHECKPOINT = "yujiepan/sam3-tiny-random"


class SAM3QLoRA(nn.Module):
    """
    SAM3 model with QLoRA (4-bit quantization + LoRA adapters) for binary mask segmentation.

    Args:
        pretrained_model_name_or_path (str): Hugging Face model repository or local path
            (e.g. 'jetjodh/sam3', 'facebook/sam3', or 'yujiepan/sam3-tiny-random').
        load_in_4bit (bool): Whether to quantize the base model in 4-bit via bitsandbytes (default: True).
        load_in_8bit (bool): Whether to quantize in 8-bit (default: False).
        lora_r (int): LoRA rank dimension (default: 8).
        lora_alpha (int): LoRA scaling factor (default: 16).
        lora_dropout (float): Dropout probability for LoRA layers (default: 0.05).
        lora_target_modules (List[str]): Module names to attach LoRA adapters to.
        prompt_text (str): Conditioning text prompt for SAM3 (default: 'tampered region').
        aux_classifier (bool): Whether to enable auxiliary classification head (default: False).
        num_classes (int): Number of target classes for auxiliary head (default: 3).
        in_channels (int): Input image channels (default: 3).
        out_channels (int): Output mask channels (default: 1).
        target_size (Tuple[int, int]): Native input resolution for SAM3 ViT (default: (1008, 1008)).
        device (str or torch.device): Target device (default: 'auto').
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = DEFAULT_SAM3_CHECKPOINT,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        prompt_text: str = "tampered region",
        aux_classifier: bool = False,
        num_classes: int = 3,
        in_channels: int = 3,
        out_channels: int = 1,
        target_size: Tuple[int, int] = (1008, 1008),
        device: Optional[Union[str, torch.device]] = "auto",
        **kwargs: Any,
    ):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or ["q_proj", "v_proj"]
        self.prompt_text = prompt_text
        self.aux_classifier = aux_classifier
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.target_size = tuple(target_size)

        # Resolve device
        if device == "auto" or device is None:
            self._target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self._target_device = torch.device(device)
        else:
            self._target_device = device

        self._init_processor_and_model()

        # Auxiliary classification head
        if self.aux_classifier:
            self.classifier_head = nn.Linear(1, self.num_classes)
        else:
            self.classifier_head = None

    def _init_processor_and_model(self):
        """Initialize SAM3 processor and QLoRA model with fallback."""
        from transformers import Sam3Model, Sam3Processor, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        # 1. Load processor / tokenizer
        self.processor = self._load_processor()

        # Pre-tokenize prompt text
        text_inputs = self.processor(text=self.prompt_text, return_tensors="pt")
        self.register_buffer("base_input_ids", text_inputs["input_ids"], persistent=False)
        self.register_buffer("base_attention_mask", text_inputs["attention_mask"], persistent=False)

        # 2. Determine quantization configuration
        is_cuda = (self._target_device.type == "cuda") and torch.cuda.is_available()
        bnb_config = None

        if self.load_in_4bit and is_cuda:
            compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            device_map = "auto"
        elif self.load_in_8bit and is_cuda:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            device_map = "auto"
        else:
            device_map = None

        # 3. Load base Sam3Model
        base_model = self._load_base_model(bnb_config=bnb_config, device_map=device_map)

        # 4. Prepare for k-bit training if quantized
        if bnb_config is not None:
            base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)
        else:
            for p in base_model.parameters():
                p.requires_grad = False

        # 5. Apply LoRA
        lora_cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
        )
        self.model = get_peft_model(base_model, lora_cfg)
        logger.info(
            f"Initialized SAM3-QLoRA with base='{self.pretrained_model_name_or_path}', "
            f"4bit={self.load_in_4bit}, r={self.lora_r}, alpha={self.lora_alpha}"
        )

    def _load_processor(self):
        """Attempt to load processor with fallback handling."""
        from transformers import Sam3Processor
        for cand in [self.pretrained_model_name_or_path, DEFAULT_SAM3_CHECKPOINT, FALLBACK_SAM3_CHECKPOINT]:
            try:
                return Sam3Processor.from_pretrained(cand)
            except Exception as e:
                logger.debug(f"Processor loading from '{cand}' failed: {e}")
                continue
        raise RuntimeError(f"Could not load Sam3Processor from '{self.pretrained_model_name_or_path}' or fallbacks.")

    def _load_base_model(self, bnb_config=None, device_map=None):
        """Attempt to load base Sam3Model with fallback handling."""
        from transformers import Sam3Model
        candidates = [self.pretrained_model_name_or_path]
        if DEFAULT_SAM3_CHECKPOINT not in candidates:
            candidates.append(DEFAULT_SAM3_CHECKPOINT)
        if FALLBACK_SAM3_CHECKPOINT not in candidates:
            candidates.append(FALLBACK_SAM3_CHECKPOINT)

        last_err = None
        for cand in candidates:
            try:
                logger.info(f"Loading base Sam3Model from '{cand}'...")
                load_kwargs: Dict[str, Any] = {}
                if bnb_config is not None:
                    load_kwargs["quantization_config"] = bnb_config
                if device_map is not None:
                    load_kwargs["device_map"] = device_map

                model = Sam3Model.from_pretrained(cand, **load_kwargs)
                self.pretrained_model_name_or_path = cand
                return model
            except Exception as e:
                last_err = e
                logger.warning(f"Could not load base model from '{cand}': {e}. Trying fallback...")
                continue

        raise RuntimeError(f"Failed to load Sam3Model from any candidate: {last_err}")

    def forward(
        self,
        x: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for SAM3-QLoRA.

        Args:
            x (torch.Tensor): Input images of shape [B, 3, H, W].

        Returns:
            torch.Tensor: Predicted mask logits of shape [B, 1, H, W].
            If aux_classifier is True, returns Tuple[torch.Tensor, torch.Tensor] (mask_logits, cls_logits).
        """
        b_sz, _, orig_h, orig_w = x.shape
        model_device = next(self.model.parameters()).device

        # Resize to SAM3 native resolution if necessary
        if (orig_h, orig_w) != self.target_size:
            x_proc = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)
        else:
            x_proc = x

        # Ensure correct tensor type and device
        # If model has half/bfloat16 parameters, cast input accordingly
        param_dtype = next(
            (p.dtype for p in self.model.parameters() if p.is_floating_point()),
            torch.float32,
        )
        if param_dtype in (torch.float16, torch.bfloat16):
            x_proc = x_proc.to(device=model_device, dtype=param_dtype)
        else:
            x_proc = x_proc.to(device=model_device)

        # Prepare batched prompt tokens
        batch_input_ids = self.base_input_ids.repeat(b_sz, 1).to(model_device)
        batch_attention_mask = self.base_attention_mask.repeat(b_sz, 1).to(model_device)

        # Forward pass through PEFT SAM3 model
        outputs = self.model(
            pixel_values=x_proc,
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
        )

        # outputs.semantic_seg is shape [B, 1, H_fpn, W_fpn]
        semantic_seg = outputs.semantic_seg
        if semantic_seg is None:
            # Fallback if semantic_seg is absent: pool pred_masks across queries
            # pred_masks is [B, num_queries, H_fpn, W_fpn]
            if hasattr(outputs, "pred_masks") and outputs.pred_masks is not None:
                semantic_seg = outputs.pred_masks.mean(dim=1, keepdim=True)
            else:
                raise RuntimeError("SAM3 output contains neither semantic_seg nor pred_masks.")

        # Interpolate mask logits back to original image dimensions [B, 1, orig_h, orig_w]
        mask_logits = F.interpolate(
            semantic_seg.float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )

        if self.aux_classifier and self.classifier_head is not None:
            if hasattr(outputs, "presence_logits") and outputs.presence_logits is not None:
                pres = outputs.presence_logits.float()
                if pres.ndim == 1:
                    pres = pres.unsqueeze(-1)
                cls_logits = self.classifier_head(pres)
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
        """
        Produce binary mask prediction (values 0.0 or 1.0) given input image tensor.
        """
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
    ) -> Union[SAM3QLoRA, Tuple[SAM3QLoRA, Any]]:
        """Load SAM3-QLoRA model from checkpoint file (.pt)."""
        from sid_unet.models.unet import UNet
        return UNet.from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            override_config=override_config,
            strict=strict,
            return_config=return_config,
        )
