"""
Configuration management system for SID-UNet.
Handles YAML loading, default validation, deep merging, CLI overrides, and serialization.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional, Union
import yaml


class ConfigDict(dict):
    """Dictionary subclass supporting attribute access (e.g. cfg.model.in_channels)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            if isinstance(v, dict) and not isinstance(v, ConfigDict):
                self[k] = ConfigDict(v)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Configuration has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"Configuration has no attribute '{name}'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert recursively back to standard python dictionary."""
        result = {}
        for k, v in self.items():
            if isinstance(v, ConfigDict):
                result[k] = v.to_dict()
            elif isinstance(v, list):
                result[k] = [item.to_dict() if isinstance(item, ConfigDict) else item for item in v]
            else:
                result[k] = v
        return result


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge dictionary `update` into `base`.
    """
    result = copy.deepcopy(base)
    for k, v in update.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def parse_value(val_str: str) -> Any:
    """Parse string value into appropriate Python type (int, float, bool, list, dict)."""
    val_str_strip = val_str.strip()
    try:
        parsed = yaml.safe_load(val_str_strip)
        return parsed
    except Exception:
        if val_str_strip.lower() == "true":
            return True
        if val_str_strip.lower() == "false":
            return False
        if val_str_strip.lower() in ("none", "null"):
            return None
        return val_str_strip


def apply_overrides(config: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    """
    Apply dot-notation CLI overrides to config dict.
    Example: ['training.batch_size=16', 'data.streaming=false', 'model.features=[16,32]']
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format '{override}'. Expected 'key.nested=value'.")
        key_path, raw_value = override.split("=", 1)
        keys = key_path.strip().split(".")
        value = parse_value(raw_value.strip())

        curr = config
        for key in keys[:-1]:
            if key not in curr or not isinstance(curr[key], dict):
                curr[key] = {}
            curr = curr[key]
        curr[keys[-1]] = value
    return config


DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {
        "name": "sid_unet",
        "seed": 42,
        "device": "auto",  # 'auto', 'cuda', or 'cpu'
        "output_dir": "outputs",
    },
    "data": {
        "dataset_name": "saberzl/SID_Set",
        "streaming": True,
        "image_size": [256, 256],
        "batch_size": 16,
        "num_workers": 2,
        "pin_memory": True,
        "shuffle_buffer_size": 1000,
        "train_split": "train",
        "val_split": "validation",
        "train_samples_per_epoch": 2000,  # For streaming dataset epoch delineation
        "val_samples": 400,              # Max samples for validation pass
        "augmentations": {
            "horizontal_flip": 0.5,
            "vertical_flip": 0.2,
            "random_rotate90": 0.5,
            "color_jitter": 0.0,
        },
    },
    "model": {
        "name": "unet",
        "in_channels": 3,
        "out_channels": 1,
        "features": [64, 128, 256, 512],
        "bilinear": True,
        "dropout": 0.1,
        "aux_classifier": True,      # 3-class auxiliary classification head
        "num_classes": 3,            # Real (0), Fully AI (1), Partially AI (2)
    },
    "loss": {
        "mask_loss_type": "combined", # 'bce', 'dice', 'focal', 'combined'
        "bce_weight": 0.5,
        "dice_weight": 0.5,
        "focal_gamma": 2.0,
        "focal_alpha": 0.25,
        "aux_loss_type": "cross_entropy",
        "aux_weight": 0.2,            # Weight for auxiliary classification loss
    },
    "training": {
        "epochs": 10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "optimizer": "adamw",         # 'adam', 'adamw', 'sgd'
        "scheduler": "cosine",        # 'cosine', 'step', 'plateau', 'none'
        "warmup_epochs": 1,
        "min_lr": 1e-6,
        "grad_clip_norm": 1.0,
        "amp": True,                  # Automatic mixed precision
        "save_best": True,
        "save_latest": True,
        "eval_interval": 1,           # Validate every N epochs
        "early_stopping_patience": 5,
        "early_stopping_metric": "val_iou",
        "early_stopping_mode": "max",
    },
    "logging": {
        "log_interval": 20,           # Log training metrics every N steps
    },
}


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[list[str]] = None,
) -> ConfigDict:
    """
    Load configuration from YAML file, merged over DEFAULT_CONFIG,
    and apply optional command-line overrides.
    """
    config = copy.deepcopy(DEFAULT_CONFIG)

    if config_path is not None and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f)
            if user_config and isinstance(user_config, dict):
                config = deep_merge(config, user_config)
    elif config_path is not None:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if overrides:
        config = apply_overrides(config, overrides)

    return ConfigDict(config)


def save_config(config: Union[ConfigDict, Dict[str, Any]], save_path: str) -> None:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    dict_data = config.to_dict() if isinstance(config, ConfigDict) else config
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict_data, f, default_flow_style=False, sort_keys=False)
