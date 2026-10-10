"""
Modular Model Cache Extractors for High-Dimensional Forensics Representations.
Extracts frozen model latents and embeddings into compact tensors for offline caching.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseCacheExtractor(abc.ABC):
    """
    Abstract base class for model cache feature extractors.
    Subclasses define how high-dimensional representations are extracted from input images.
    """

    def __init__(self, model: nn.Module, fp16: bool = True):
        self.model = model
        self.fp16 = bool(fp16)

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Name identifier of the model architecture."""
        pass

    @property
    @abc.abstractmethod
    def total_channels(self) -> int:
        """Total channel dimension of the cached tensor."""
        pass

    @abc.abstractmethod
    def extract_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract high-dimensional tensor representation from a batch of RGB images.

        Args:
            images: Tensor of shape [B, 3, H, W] in range [0, 1] or [-1, 1].

        Returns:
            Cached tensor of shape [B, total_channels, H_latent, W_latent].
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata dict to persist alongside the cached dataset."""
        return {
            "model_name": self.model_name,
            "total_channels": self.total_channels,
            "fp16": self.fp16,
        }


class DiffusionDiffMinimizedExtractor(BaseCacheExtractor):
    """Cache extractor for DiffusionDiffMinimizedModel."""

    @property
    def model_name(self) -> str:
        return "diffusion_diff_minimized"

    @property
    def total_channels(self) -> int:
        return getattr(self.model, "total_z_channels", 84)

    @torch.no_grad()
    def extract_batch(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        if hasattr(self.model, "extract_cache_tensors"):
            z = self.model.extract_cache_tensors(images)
        else:
            raise NotImplementedError("Model does not implement extract_cache_tensors")

        if self.fp16 and z.dtype != torch.float16:
            z = z.half()
        return z

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        meta.update({
            "timesteps": getattr(self.model, "timesteps", [250]),
            "timestep_embed_dim": getattr(self.model, "timestep_embed_dim", 32),
            "sigma_embed_dim": getattr(self.model, "sigma_embed_dim", 32),
            "scaling_factor": getattr(self.model, "scaling_factor", 0.18215),
            "latent_channels": getattr(self.model, "latent_channels", 4),
            "pretrained_model": getattr(self.model, "pretrained_model_name_or_path", "segmind/tiny-sd"),
        })
        return meta


class DiffusionDiffV2Extractor(BaseCacheExtractor):
    """Cache extractor for DiffusionDiffV2Model."""

    @property
    def model_name(self) -> str:
        return "diffusion_diff_v2"

    @property
    def total_channels(self) -> int:
        return getattr(self.model, "total_z_channels", 244)

    @torch.no_grad()
    def extract_batch(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        if hasattr(self.model, "extract_cache_tensors"):
            z = self.model.extract_cache_tensors(images)
        else:
            raise NotImplementedError("Model does not implement extract_cache_tensors")

        if self.fp16 and z.dtype != torch.float16:
            z = z.half()
        return z

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        meta.update({
            "timesteps": getattr(self.model, "timesteps", [100, 250, 500]),
            "timestep_embed_dim": getattr(self.model, "timestep_embed_dim", 32),
            "sigma_embed_dim": getattr(self.model, "sigma_embed_dim", 32),
            "scaling_factor": getattr(self.model, "scaling_factor", 0.18215),
            "latent_channels": getattr(self.model, "latent_channels", 4),
            "pretrained_model": getattr(self.model, "pretrained_model_name_or_path", "runwayml/stable-diffusion-v1-5"),
        })
        return meta


class DiffusionDiffExtractor(BaseCacheExtractor):
    """Cache extractor for standard DiffusionDiffModel."""

    @property
    def model_name(self) -> str:
        return "diffusion_diff"

    @property
    def total_channels(self) -> int:
        return getattr(self.model, "total_z_channels", 244)

    @torch.no_grad()
    def extract_batch(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        if hasattr(self.model, "extract_cache_tensors"):
            z = self.model.extract_cache_tensors(images)
        else:
            raise NotImplementedError("Model does not implement extract_cache_tensors")

        if self.fp16 and z.dtype != torch.float16:
            z = z.half()
        return z

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        meta.update({
            "timesteps": getattr(self.model, "timesteps", [100, 250, 500]),
            "timestep_embed_dim": getattr(self.model, "timestep_embed_dim", 32),
            "sigma_embed_dim": getattr(self.model, "sigma_embed_dim", 32),
            "scaling_factor": getattr(self.model, "scaling_factor", 0.18215),
            "latent_channels": getattr(self.model, "latent_channels", 4),
            "pretrained_model": getattr(self.model, "pretrained_model_name_or_path", "runwayml/stable-diffusion-v1-5"),
        })
        return meta


class GAPSAMCacheExtractor(BaseCacheExtractor):
    """Cache extractor for GAPSAM models (Global Artifact Prior with sam-distil backbones)."""

    @property
    def model_name(self) -> str:
        return "gap_sam"

    @property
    def total_channels(self) -> int:
        return 256

    @torch.no_grad()
    def extract_batch(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        if hasattr(self.model, "extract_cache_tensors"):
            z = self.model.extract_cache_tensors(images)
        else:
            raise NotImplementedError("GAPSAM model does not implement extract_cache_tensors")

        if self.fp16 and z.dtype != torch.float16:
            z = z.half()
        return z

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        meta.update({
            "backbone_type": getattr(self.model, "backbone_type", "tinyvit"),
            "model_variant": getattr(self.model, "model_name", "11m"),
            "vae_pretrained_model_name_or_path": getattr(
                self.model, "vae_pretrained_model_name_or_path", "stabilityai/sd-vae-ft-mse"
            ),
            "target_size": getattr(self.model, "target_size", (1008, 1008)),
        })
        return meta


# Registry of model extractors
_EXTRACTOR_REGISTRY: Dict[str, Type[BaseCacheExtractor]] = {
    "diffusion_diff_minimized": DiffusionDiffMinimizedExtractor,
    "diffusion-diff-minimized": DiffusionDiffMinimizedExtractor,
    "diff_minimized": DiffusionDiffMinimizedExtractor,
    "diffusion_minimized": DiffusionDiffMinimizedExtractor,
    "diffusion_diff_v2": DiffusionDiffV2Extractor,
    "diffusion-diff-v2": DiffusionDiffV2Extractor,
    "diff_v2": DiffusionDiffV2Extractor,
    "diffusion_diff": DiffusionDiffExtractor,
    "diffusion-diff": DiffusionDiffExtractor,
    "gap_sam": GAPSAMCacheExtractor,
    "gap-sam": GAPSAMCacheExtractor,
    "gapsam": GAPSAMCacheExtractor,
}


def register_extractor(model_name: str, extractor_cls: Type[BaseCacheExtractor]) -> None:
    """Register a new cache extractor for future model architectures."""
    _EXTRACTOR_REGISTRY[model_name.lower()] = extractor_cls


def get_extractor_for_model(model_or_name: Union[str, nn.Module], **kwargs: Any) -> BaseCacheExtractor:
    """
    Factory function to get the appropriate cache extractor for a given model or name.
    """
    if isinstance(model_or_name, str):
        key = model_or_name.lower()
        if key in _EXTRACTOR_REGISTRY:
            return _EXTRACTOR_REGISTRY[key](**kwargs)
        raise ValueError(f"No cache extractor registered for model name '{model_or_name}'. Registered: {list(_EXTRACTOR_REGISTRY.keys())}")

    # Inspect class name
    cls_name = model_or_name.__class__.__name__.lower()
    for key, ext_cls in _EXTRACTOR_REGISTRY.items():
        if key.replace("-", "").replace("_", "") in cls_name.replace("-", "").replace("_", ""):
            return ext_cls(model=model_or_name, **kwargs)

    # Check if model has extract_cache_tensors directly
    if hasattr(model_or_name, "extract_cache_tensors"):
        class _GenericExtractor(BaseCacheExtractor):
            @property
            def model_name(self) -> str:
                return model_or_name.__class__.__name__

            @property
            def total_channels(self) -> int:
                return getattr(self.model, "total_z_channels", getattr(self.model, "in_channels", 84))

            @torch.no_grad()
            def extract_batch(self, images: torch.Tensor) -> torch.Tensor:
                self.model.eval()
                res = self.model.extract_cache_tensors(images)
                return res.half() if self.fp16 else res

        return _GenericExtractor(model=model_or_name, **kwargs)

    raise ValueError(f"Could not determine cache extractor for model class '{model_or_name.__class__.__name__}'")
