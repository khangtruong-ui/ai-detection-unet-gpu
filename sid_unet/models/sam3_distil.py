"""
SAM3-Distil (EfficientSAM3) + LoRA / QLoRA model for AI-generated and tampered synthetic image segmentation.
Wraps the distilled, lightweight EfficientSAM3 foundation model (TinyViT, EfficientViT, RepViT backbones)
with Low-Rank Adaptation (LoRA) via PEFT to achieve parameter-efficient fine-tuning on consumer GPUs.
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

warnings.filterwarnings("ignore", message=".*memory_attention_rope_theta.*")
warnings.filterwarnings("ignore", message=".*Importing from timm.models.layers.*")

logger = logging.getLogger("sid_unet.models.sam3_distil")

DEFAULT_TINYVIT_CHECKPOINT = "/workspace/sam3-distil/checkpoints/efficientsam3_ft/efficientsam3_tinyvit.pt"
DEFAULT_EFFICIENTVIT_CHECKPOINT = "/workspace/sam3-distil/checkpoints/efficientsam3_ft/efficientsam3_efficientvit.pt"
DEFAULT_SAM3_DISTIL_CHECKPOINT = DEFAULT_TINYVIT_CHECKPOINT

DEFAULT_EFFICIENTSAM3_HF_REPO = "Simon7108528/EfficientSAM3"
DEFAULT_EFFICIENTSAM3_REPO = DEFAULT_EFFICIENTSAM3_HF_REPO

BACKBONE_TO_HF_FILE = {
    "tinyvit": "efficientsam3_ft/efficientsam3_tinyvit.pt",
    "tiny_vit": "efficientsam3_ft/efficientsam3_tinyvit.pt",
    "tv": "efficientsam3_ft/efficientsam3_tinyvit.pt",
    "tvm": "efficientsam3_ft/efficientsam3_tinyvit.pt",
    "efficientvit": "efficientsam3_ft/efficientsam3_efficientvit.pt",
    "efficient_vit": "efficientsam3_ft/efficientsam3_efficientvit.pt",
    "ev": "efficientsam3_ft/efficientsam3_efficientvit.pt",
    "evm": "efficientsam3_ft/efficientsam3_efficientvit.pt",
    "repvit": "efficientsam3_ft/efficientsam3_repvit.pt",
    "rep_vit": "efficientsam3_ft/efficientsam3_repvit.pt",
    "rv": "efficientsam3_ft/efficientsam3_repvit.pt",
    "rvm": "efficientsam3_ft/efficientsam3_repvit.pt",
}

KNOWN_CHECKPOINT_FILENAMES = {
    "efficientsam3_tinyvit.pt": (DEFAULT_EFFICIENTSAM3_HF_REPO, "efficientsam3_ft/efficientsam3_tinyvit.pt"),
    "efficientsam3_efficientvit.pt": (DEFAULT_EFFICIENTSAM3_HF_REPO, "efficientsam3_ft/efficientsam3_efficientvit.pt"),
    "efficientsam3_repvit.pt": (DEFAULT_EFFICIENTSAM3_HF_REPO, "efficientsam3_ft/efficientsam3_repvit.pt"),
}


def resolve_sam3_distil_checkpoint(
    checkpoint_path: Optional[str] = DEFAULT_SAM3_DISTIL_CHECKPOINT,
    backbone_type: str = "tinyvit",
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    force_download: bool = False,
) -> Optional[str]:
    """
    Resolve checkpoint path with automated fallbacks to local repository paths and
    Hugging Face Hub downloading with caching in the standard HF cache folder ($HF_HOME/hub).

    Args:
        checkpoint_path: Explicit path, filename, or HF repo ID/URI. If None or 'none', returns None.
        backbone_type: 'tinyvit', 'efficientvit', or 'repvit'.
        cache_dir: Optional local directory to store downloaded checkpoint.
        token: Optional HF authentication token.
        force_download: Whether to force re-download from HF even if cached.

    Returns:
        Validated absolute path or None if no checkpoint is available or requested.
    """
    if checkpoint_path is None or str(checkpoint_path).strip().lower() in ("none", "", "null"):
        return None

    raw_path = str(checkpoint_path).strip()

    # 1. Existing local file
    if os.path.isfile(raw_path):
        return os.path.abspath(raw_path)

    # 2. Check relative to /workspace/sam3-distil or current working directory
    for base in ["/workspace/sam3-distil", "/workspace", os.getcwd()]:
        cand = os.path.join(base, raw_path.lstrip("/"))
        if os.path.isfile(cand):
            return os.path.abspath(cand)

    from sid_unet.utils.checkpoint import is_hf_repo_id, download_hf_checkpoint

    # 3. Explicit HF repo ID or URI (e.g. 'Simon7108528/EfficientSAM3', 'hf://...')
    if is_hf_repo_id(raw_path) or raw_path.startswith(("hf://", "https://huggingface.co/")):
        try:
            hf_res = download_hf_checkpoint(
                repo_id_or_uri=raw_path,
                filename=BACKBONE_TO_HF_FILE.get(backbone_type.lower(), "efficientsam3_ft/efficientsam3_tinyvit.pt") if ":" not in raw_path else None,
                cache_dir=cache_dir,
                token=token,
            )
            return hf_res["checkpoint_path"]
        except Exception as exc:
            logger.warning(f"Could not download checkpoint from HF repo '{raw_path}': {exc}")
            return None

    # 4. Check if filename matches a known checkpoint for automated HF Hub download into cache
    basename = os.path.basename(raw_path)
    if basename in KNOWN_CHECKPOINT_FILENAMES or raw_path in (DEFAULT_SAM3_DISTIL_CHECKPOINT, DEFAULT_TINYVIT_CHECKPOINT, DEFAULT_EFFICIENTVIT_CHECKPOINT):
        if basename in KNOWN_CHECKPOINT_FILENAMES:
            repo_id, hf_filename = KNOWN_CHECKPOINT_FILENAMES[basename]
        else:
            repo_id = DEFAULT_EFFICIENTSAM3_HF_REPO
            hf_filename = BACKBONE_TO_HF_FILE.get(backbone_type.lower(), "efficientsam3_ft/efficientsam3_tinyvit.pt")

        logger.info(
            f"Local checkpoint '{raw_path}' not found on disk. "
            f"Fetching '{hf_filename}' from Hugging Face Hub '{repo_id}' into cache..."
        )
        try:
            hf_res = download_hf_checkpoint(
                repo_id_or_uri=repo_id,
                filename=hf_filename,
                cache_dir=cache_dir,
                token=token,
            )
            return hf_res["checkpoint_path"]
        except Exception as exc:
            logger.warning(
                f"Could not download checkpoint from Hugging Face Hub '{repo_id}/{hf_filename}': {exc}. "
                "Proceeding with randomly initialized weights for demonstration or testing."
            )
            return None

    logger.warning(
        f"⚠️ SAM3-Distil checkpoint '{checkpoint_path}' not found on disk or Hugging Face Hub. "
        "Proceeding with randomly initialized weights for demonstration or testing."
    )
    return None


class SAM3DistilLoRA(nn.Module):
    """
    EfficientSAM3 (sam3-distil) model with LoRA (Low-Rank Adaptation) adapters
    for binary mask segmentation and optional auxiliary classification.

    Args:
        checkpoint_path: Path to pre-distilled EfficientSAM3 checkpoint (.pt) or Hugging Face repo ID.
        backbone_type: Vision backbone architecture ('tinyvit', 'efficientvit', 'repvit').
        model_name: Backbone variant (e.g. '11m', 'b0', 'm1.1').
        text_encoder_type: Text encoder type (e.g. 'MobileCLIP-S0').
        text_encoder_context_length: Context token length (default: 16).
        load_in_4bit: Whether to enable 4-bit NF4 quantization for base model (default: False).
        load_in_8bit: Whether to enable 8-bit quantization for base model (default: False).
        lora_r: LoRA rank dimension (default: 16).
        lora_alpha: LoRA scaling factor (default: 32).
        lora_dropout: Dropout probability for LoRA layers (default: 0.05).
        lora_target_modules: Module names to attach LoRA adapters to.
        prompt_text: Text conditioning prompt (default: 'tampered region').
        aux_classifier: Whether to enable auxiliary classification head (default: True).
        num_classes: Number of target classes for auxiliary head (default: 3).
        in_channels: Input image channels (default: 3).
        out_channels: Output mask channels (default: 1).
        target_size: Native input resolution for EfficientSAM3 (default: (1008, 1008)).
        input_rescale: Whether to normalize [0, 1] inputs to [-1, 1] (default: True).
        device: Target device (default: 'auto').
        cache_dir: Optional custom Hugging Face cache directory.
        token: Optional Hugging Face authentication token.
        force_download: Whether to force re-download from Hugging Face Hub even if cached.
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[str] = None,
        *args,
        checkpoint_path: Optional[str] = None,
        backbone_type: str = "tinyvit",
        model_name: str = "11m",
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = "auto",
        **kwargs: Any,
    ) -> SAM3DistilLoRA:
        """Hugging Face style from_pretrained loader for SAM3DistilLoRA.

        Loads pre-distilled EfficientSAM3 weights from local disk or Hugging Face Hub,
        automatically downloading into the HF cache folder ($HF_HOME/hub).
        """
        ckpt = pretrained_model_name_or_path or checkpoint_path or DEFAULT_SAM3_DISTIL_CHECKPOINT
        return cls(
            checkpoint_path=ckpt,
            backbone_type=backbone_type,
            model_name=model_name,
            cache_dir=cache_dir,
            token=token,
            device=device,
            **kwargs,
        )

    def __init__(
        self,
        checkpoint_path: Optional[str] = DEFAULT_SAM3_DISTIL_CHECKPOINT,
        backbone_type: str = "tinyvit",
        model_name: str = "11m",
        text_encoder_type: Optional[str] = "MobileCLIP-S0",
        text_encoder_context_length: int = 16,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        prompt_text: str = "tampered region",
        aux_classifier: bool = True,
        num_classes: int = 3,
        in_channels: int = 3,
        out_channels: int = 1,
        target_size: Tuple[int, int] = (1008, 1008),
        input_rescale: bool = True,
        device: Optional[Union[str, torch.device]] = "auto",
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        force_download: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.backbone_type = backbone_type
        self.model_name = model_name
        self.text_encoder_type = text_encoder_type
        self.text_encoder_context_length = text_encoder_context_length
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or [
            "qkv", "qkv_proj", "out_proj", "proj", "linear1", "linear2"
        ]
        self.prompt_text = prompt_text
        self.aux_classifier = aux_classifier
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.target_size = tuple(target_size)
        self.input_rescale = input_rescale
        self.cache_dir = cache_dir
        self.token = token
        self.force_download = force_download

        # Resolve device
        if device == "auto" or device is None:
            self._target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self._target_device = torch.device(device)
        else:
            self._target_device = device

        self._text_cache: Dict[Tuple[str, str], Dict[str, torch.Tensor]] = {}

        self._init_model()

        # Auxiliary classification head
        if self.aux_classifier:
            self.classifier_head = nn.Linear(1, self.num_classes).to(self._target_device)
        else:
            self.classifier_head = None

    def _init_model(self):
        """Build base EfficientSAM3 model and attach LoRA/QLoRA adapters."""
        try:
            from sam3.model_builder import build_efficientsam3_image_model
            from sam3.model.data_misc import FindStage
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "SAM3-Distil models require sam3-distil, peft, and dependencies (including einops and pycocotools). "
                "Please install them via `pip install 'sid-unet[sam3-distil]'`."
            ) from exc

        # 1. Resolve checkpoint path
        ckpt = resolve_sam3_distil_checkpoint(
            self.checkpoint_path,
            backbone_type=self.backbone_type,
            cache_dir=self.cache_dir,
            token=self.token,
            force_download=self.force_download,
        )

        # Resolve bpe vocabulary path for student text encoder
        bpe_path = None
        try:
            import sam3
            sam3_pkg_dir = os.path.dirname(sam3.__file__)
            for candidate in [
                os.path.join(sam3_pkg_dir, "assets", "bpe_simple_vocab_16e6.txt.gz"),
                os.path.join(sam3_pkg_dir, "..", "assets", "bpe_simple_vocab_16e6.txt.gz"),
            ]:
                if os.path.exists(candidate):
                    bpe_path = os.path.abspath(candidate)
                    break
        except Exception:
            pass

        # 2. Build base EfficientSAM3 image model
        base_model = build_efficientsam3_image_model(
            checkpoint_path=ckpt,
            bpe_path=bpe_path,
            backbone_type=self.backbone_type,
            model_name=self.model_name,
            text_encoder_type=self.text_encoder_type,
            text_encoder_context_length=self.text_encoder_context_length,
            device=str(self._target_device),
            eval_mode=False,
            cache_dir=self.cache_dir,
            token=self.token,
            force_download=self.force_download,
        )

        # Freeze base model parameters
        for p in base_model.parameters():
            p.requires_grad = False

        # 3. Patch forward_grounding to safely bypass DETR bipartite matching when find_target is None
        orig_forward_grounding = base_model.forward_grounding

        def safe_forward_grounding(
            backbone_out,
            find_input,
            find_target=None,
            geometric_prompt=None,
        ):
            was_training = base_model.training
            if find_target is None:
                base_model.training = False
            try:
                return orig_forward_grounding(
                    backbone_out=backbone_out,
                    find_input=find_input,
                    find_target=find_target,
                    geometric_prompt=geometric_prompt,
                )
            finally:
                base_model.training = was_training

        base_model.forward_grounding = safe_forward_grounding

        # 4. Attach LoRA adapters
        # Resolve target module names to valid leaf layers (Linear, Conv2d) to avoid
        # failures on custom composite layers (e.g. EfficientViT ConvLayer wrappers).
        resolved_targets: Union[List[str], str] = self.lora_target_modules
        if isinstance(self.lora_target_modules, (list, tuple)):
            matched = []
            for name, module in base_model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    parts = name.split(".")
                    if any(t in parts for t in self.lora_target_modules) or any(
                        name.endswith("." + t) or name == t for t in self.lora_target_modules
                    ):
                        matched.append(name)
            if matched:
                resolved_targets = matched

        lora_cfg = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=resolved_targets,
            lora_dropout=self.lora_dropout,
            bias="none",
        )
        self.peft_model = get_peft_model(base_model, lora_cfg)
        self.base_model = base_model

        # Cached prompt structures
        self.dummy_prompt = self.base_model._get_dummy_prompt()

        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        logger.info(
            f"Initialized SAM3-Distil-LoRA ({self.backbone_type}-{self.model_name}) | "
            f"Trainable LoRA params: {trainable_params:,} / {total_params:,} "
            f"({100.0 * trainable_params / max(total_params, 1):.2f}%) | "
            f"Rank: {self.lora_r}, Alpha: {self.lora_alpha}"
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for semantic segmentation and optional auxiliary classification.

        Args:
            x: Input image tensor of shape [B, C, H, W] (e.g. RGB images normalized to [0, 1]).

        Returns:
            mask_logits: Unnormalized logits of shape [B, 1, H, W].
            (mask_logits, class_logits): If aux_classifier is True, returns tuple where
                class_logits is [B, num_classes].
        """
        from sam3.model.data_misc import FindStage

        b_sz, _, orig_h, orig_w = x.shape
        model_device, param_dtype = get_submodule_device_dtype(
            self.peft_model, x.device, torch.float32
        )

        # 1. Resize to native target size (1008, 1008)
        if (orig_h, orig_w) != self.target_size:
            x_proc = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)
        else:
            x_proc = x

        # 2. Rescale from [0, 1] to [-1, 1] if enabled
        if self.input_rescale:
            x_proc = (x_proc - 0.5) / 0.5

        # 3. Match device and precision
        if param_dtype in (torch.float16, torch.bfloat16):
            x_proc = x_proc.to(device=model_device, dtype=param_dtype)
        else:
            x_proc = x_proc.to(device=model_device)

        # 4. Extract visual features for the whole batch
        backbone_out = self.base_model.backbone.forward_image(x_proc)

        # 5. Extract text prompt embeddings once (cached with safe cuDNN fallback)
        cache_key = (self.prompt_text, str(model_device))
        if cache_key not in self._text_cache:
            with torch.no_grad():
                try:
                    out_t = self.base_model.backbone.forward_text(
                        [self.prompt_text], device=model_device
                    )
                except RuntimeError as exc:
                    if "FIND was unable to find an engine" in str(exc) or "cuDNN" in str(exc):
                        with torch.backends.cudnn.flags(enabled=False):
                            out_t = self.base_model.backbone.forward_text(
                                [self.prompt_text], device=model_device
                            )
                    else:
                        raise exc
                self._text_cache[cache_key] = {k: v.clone() for k, v in out_t.items()}
        text_outputs = {k: v.clone() for k, v in self._text_cache[cache_key].items()}

        # 6. Reusable find_stage
        find_stage = FindStage(
            img_ids=torch.tensor([0], device=model_device, dtype=torch.long),
            text_ids=torch.tensor([0], device=model_device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

        semantic_segs = []
        presence_logits = []

        # 7. Grounding decoder per batch item
        for i in range(b_sz):
            item_backbone = {
                "vision_features": backbone_out["vision_features"][i:i+1],
                "vision_pos_enc": [p[i:i+1] for p in backbone_out["vision_pos_enc"]],
                "backbone_fpn": [f[i:i+1] for f in backbone_out["backbone_fpn"]],
            }
            if "sam2_backbone_out" in backbone_out and backbone_out["sam2_backbone_out"] is not None:
                s2 = backbone_out["sam2_backbone_out"]
                if isinstance(s2, dict):
                    item_s2 = {}
                    for k_s2, v_s2 in s2.items():
                        if isinstance(v_s2, torch.Tensor):
                            item_s2[k_s2] = v_s2[i:i+1]
                        elif isinstance(v_s2, (list, tuple)):
                            item_s2[k_s2] = [t[i:i+1] if isinstance(t, torch.Tensor) else t for t in v_s2]
                        else:
                            item_s2[k_s2] = v_s2
                    item_backbone["sam2_backbone_out"] = item_s2
                else:
                    item_backbone["sam2_backbone_out"] = s2

            item_backbone.update(text_outputs)

            out = self.base_model.forward_grounding(
                backbone_out=item_backbone,
                find_input=find_stage,
                geometric_prompt=self.dummy_prompt,
                find_target=None,
            )

            # Retrieve mask logits
            if "semantic_seg" in out and out["semantic_seg"] is not None:
                seg = out["semantic_seg"]
            elif "pred_masks" in out and out["pred_masks"] is not None:
                if "pred_logits" in out and out["pred_logits"] is not None:
                    q_weights = torch.sigmoid(out["pred_logits"])
                    seg = (torch.sigmoid(out["pred_masks"]) * q_weights.unsqueeze(-1)).sum(dim=1, keepdim=True)
                else:
                    seg = out["pred_masks"].mean(dim=1, keepdim=True)
            else:
                raise RuntimeError("EfficientSAM3 output contains neither semantic_seg nor pred_masks.")

            # Retrieve presence / confidence score
            if "presence_logit_dec" in out and out["presence_logit_dec"] is not None:
                pres = out["presence_logit_dec"]
            elif "presence_logit" in out and out["presence_logit"] is not None:
                pres = out["presence_logit"]
            else:
                pres = seg.mean(dim=[-2, -1])

            if pres.ndim == 1:
                pres = pres.unsqueeze(-1)

            semantic_segs.append(seg)
            presence_logits.append(pres)

        # 8. Concatenate batch results
        batch_masks = torch.cat(semantic_segs, dim=0)
        batch_pres = torch.cat(presence_logits, dim=0)

        # 9. Resize mask logits back to input resolution [B, 1, orig_h, orig_w]
        mask_logits = F.interpolate(
            batch_masks.float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )

        if self.aux_classifier and self.classifier_head is not None:
            pres_input = batch_pres.to(
                device=self.classifier_head.weight.device,
                dtype=self.classifier_head.weight.dtype,
            )
            cls_logits = self.classifier_head(pres_input)
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

    def clear_text_cache(self):
        """Clear cached text prompt embeddings."""
        self._text_cache.clear()
        if hasattr(self.base_model, "backbone") and hasattr(self.base_model.backbone, "clear_text_cache"):
            self.base_model.backbone.clear_text_cache()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[Union[str, torch.device]] = None,
        override_config: Optional[Union[Dict[str, Any], Any]] = None,
        strict: bool = False,
        return_config: bool = False,
    ) -> Union[SAM3DistilLoRA, Tuple[SAM3DistilLoRA, Any]]:
        """Load SAM3DistilLoRA model from checkpoint file (.pt)."""
        from sid_unet.models.unet import UNet
        return UNet.from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            override_config=override_config,
            strict=strict,
            return_config=return_config,
        )


# Convenient alias
SAM3Distil = SAM3DistilLoRA
