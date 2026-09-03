"""
SID-UNet: Config-driven UNet for AI-Generated Synthetic Image Masking and Classification.
Supports streaming datasets on saberzl/SID_Set.
"""

__version__ = "0.1.0"

from sid_unet.postprocessing import (
    MaskPostProcessor,
    remove_small_components,
    fill_mask_holes,
    apply_morphology,
    get_postprocessor_from_config,
)

__all__ = [
    "__version__",
    "MaskPostProcessor",
    "remove_small_components",
    "fill_mask_holes",
    "apply_morphology",
    "get_postprocessor_from_config",
]
