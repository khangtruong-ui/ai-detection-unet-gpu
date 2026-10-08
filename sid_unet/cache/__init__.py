"""
Dataset Caching System for SID-UNet.
Provides modular extraction, serialization, loading, and Hugging Face Hub synchronization
of high-dimensional latent representations for fast diffusion forensics training.
"""

from sid_unet.cache.extractor import (
    BaseCacheExtractor,
    DiffusionDiffMinimizedExtractor,
    DiffusionDiffV2Extractor,
    DiffusionDiffExtractor,
    get_extractor_for_model,
)
from sid_unet.cache.dataset import CachedTensorDataset, CachedStreamingDataset
from sid_unet.cache.manager import DatasetCacheManager

__all__ = [
    "BaseCacheExtractor",
    "DiffusionDiffMinimizedExtractor",
    "DiffusionDiffV2Extractor",
    "DiffusionDiffExtractor",
    "get_extractor_for_model",
    "CachedTensorDataset",
    "CachedStreamingDataset",
    "DatasetCacheManager",
]
