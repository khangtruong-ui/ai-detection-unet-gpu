"""
SID-UNet: Config-driven UNet for AI-Generated Synthetic Image Masking and Classification.
Supports streaming datasets on saberzl/SID_Set.
"""

__version__ = "0.1.0"

import logging
import warnings

# Suppress deprecated property warnings from upstream transformers SAM modules
warnings.filterwarnings("ignore", message=".*memory_attention_rope_theta.*")

class _TransformersDeprecationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "memory_attention_rope_theta" not in msg

logging.getLogger("transformers").addFilter(_TransformersDeprecationFilter())
try:
    import transformers.utils.logging as _hf_logging
    if hasattr(_hf_logging, "_default_handler") and _hf_logging._default_handler:
        _hf_logging._default_handler.addFilter(_TransformersDeprecationFilter())
except ImportError:
    pass

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
