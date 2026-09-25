"""
Bootstrapping v1.0 Kickstarting Module for SID-UNet.

Implements parameter isolation, selective component freezing, calibrated initialization,
subset kickstart fitting (e.g., 512 or 2048 samples), early parameter release,
and seamless diagnostic integration with nn-toolbox to notice and verify kickstart success.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset

logger = logging.getLogger("sid_unet.bootstrapping")


class InMemoryListDataset(Dataset):
    """Simple in-memory dataset wrapping pre-collected samples from streaming or map loaders."""

    def __init__(self, samples: List[Any]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Any:
        return self.samples[idx]


def initialize_bootstrap_parameters(
    model: nn.Module,
    initialization: str = "kaiming_normal",
    target_modules: Optional[List[str]] = None,
    unfrozen_only: bool = True,
    channel_masks: Optional[Dict[str, torch.Tensor]] = None,
    saved_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """Initialize unfrozen/trainable model components with calibrated variance.

    Supports:
    - 'kaiming_normal' (He normal with fan_out)
    - 'kaiming_uniform' (He uniform)
    - 'xavier_normal' (Glorot normal)
    - 'xavier_uniform' (Glorot uniform)
    - 'orthogonal'
    - 'none' (leave weights untouched)
    """
    if initialization.lower() == "none":
        return

    init_scheme = initialization.lower()

    for name, module in model.named_modules():
        if target_modules:
            if not any(tm in name for tm in target_modules):
                continue

        # Check if module parameters are unfrozen
        if unfrozen_only:
            mod_params = list(module.parameters(recurse=False))
            if not mod_params or not any(p.requires_grad for p in mod_params):
                continue

        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            if hasattr(module, "weight") and module.weight is not None and module.weight.requires_grad:
                if init_scheme == "kaiming_normal":
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                elif init_scheme == "kaiming_uniform":
                    nn.init.kaiming_uniform_(module.weight, mode="fan_out", nonlinearity="relu")
                elif init_scheme == "xavier_normal":
                    nn.init.xavier_normal_(module.weight)
                elif init_scheme == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif init_scheme == "orthogonal":
                    nn.init.orthogonal_(module.weight)

            if hasattr(module, "bias") and module.bias is not None and module.bias.requires_grad:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Linear):
            if hasattr(module, "weight") and module.weight is not None and module.weight.requires_grad:
                if init_scheme in ("kaiming_normal", "kaiming_uniform"):
                    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                elif init_scheme in ("xavier_normal", "xavier_uniform"):
                    nn.init.xavier_uniform_(module.weight)
                elif init_scheme == "orthogonal":
                    nn.init.orthogonal_(module.weight)

            if hasattr(module, "bias") and module.bias is not None and module.bias.requires_grad:
                nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)):
            if hasattr(module, "weight") and module.weight is not None and module.weight.requires_grad:
                nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None and module.bias.requires_grad:
                nn.init.zeros_(module.bias)

    # Re-apply channel stream masks if present so tail weights remain strictly zeroed
    if channel_masks:
        params_dict = dict(model.named_parameters())
        for name, mask in channel_masks.items():
            if name in params_dict:
                param = params_dict[name]
                if saved_weights is not None:
                    saved_weights[name] = param.data.clone()
                param.data.mul_(mask.to(param.device))


class BootstrapFreezeState(dict):
    """Encapsulates model freeze state for Bootstrapping v1.0.

    Supports both legacy key-lookup (`state[name] -> bool`) and channel stream masking metadata.
    """

    def __init__(
        self,
        strategy: str = "channel_stream",
        stream_ratio: float = 0.5,
        saved_requires_grad: Optional[Dict[str, bool]] = None,
        saved_weights: Optional[Dict[str, torch.Tensor]] = None,
        channel_masks: Optional[Dict[str, torch.Tensor]] = None,
        hooks: Optional[List[Any]] = None,
        active_stream_channels: int = 0,
        frozen_tail_channels: int = 0,
    ):
        super().__init__(saved_requires_grad or {})
        self.strategy = strategy
        self.stream_ratio = stream_ratio
        self.saved_requires_grad = saved_requires_grad or {}
        self.saved_weights = saved_weights or {}
        self.channel_masks = channel_masks or {}
        self.hooks = hooks or []
        self.active_stream_channels = active_stream_channels
        self.frozen_tail_channels = frozen_tail_channels


def apply_bootstrap_freeze(
    model: nn.Module,
    strategy: str = "channel_stream",
    stream_ratio: float = 0.5,
    custom_modules: Optional[List[str]] = None,
    min_dim_for_split: int = 4,
) -> BootstrapFreezeState:
    """Selectively configure model freezing for the Bootstrapping v1.0 kickstarting phase.

    Default Strategy:
    - 'channel_stream' / 'dimension_stream' / 'auto':
      Instead of coarsely freezing entire layers (which prevents gradients from reaching end-to-end),
      freezes and zeroes out the last (1 - stream_ratio) channels of dimension D across all intermediate
      linear and convolutional weight/bias matrices. This creates a calibrated, low-capacity sub-network
      stream that runs strictly end-to-end from input to output with zero gradient leakage to frozen channels.

    Alternative Strategies:
    - 'whole_layer': Coarsely freezes entire encoder/backbone layers, training only heads/decoders.
    - 'backbone' or 'encoder': Freezes all layers matching encoder, backbone, vae, inc, down.
    - 'except_head': Freezes everything except decoder, classifier, head, outc.
    - 'custom': Freezes layers matching names in `custom_modules`.

    Returns:
        BootstrapFreezeState: Encapsulates freeze masks, saved tensors, and gradient hooks.
    """
    saved_requires_grad: Dict[str, bool] = {}
    for name, param in model.named_parameters():
        saved_requires_grad[name] = param.requires_grad

    strat = strategy.lower()
    ratio = float(min(max(stream_ratio, 0.05), 1.0))

    channel_masks: Dict[str, torch.Tensor] = {}
    saved_weights: Dict[str, torch.Tensor] = {}
    hooks: List[Any] = []
    active_stream_channels = 0
    frozen_tail_channels = 0

    if strat in ("channel_stream", "dimension_stream", "auto"):
        # End-to-end channel stream masking across all trainable layers
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if param.dim() in (2, 4):  # Linear [out, in] or Conv2d [out, in, kH, kW]
                D_out = param.shape[0]
                D_in = param.shape[1]

                # Only split dimensions greater than min_dim_for_split to protect raw input (RGB=3) and final heads (e.g. 1 mask or 3 classes)
                K_out = max(1, int(D_out * ratio)) if D_out > min_dim_for_split else D_out
                K_in = max(1, int(D_in * ratio)) if D_in > min_dim_for_split else D_in

                if K_out < D_out or K_in < D_in:
                    mask = torch.zeros_like(param.data)
                    if param.dim() == 4:
                        mask[:K_out, :K_in, ...] = 1.0
                    else:
                        mask[:K_out, :K_in] = 1.0

                    channel_masks[name] = mask
                    saved_weights[name] = param.data.clone()
                    # Zero out frozen tail channels
                    param.data.mul_(mask)
                    # Register gradient hook to enforce zero gradient leakage into tail channels
                    h = param.register_hook(lambda g, m=mask: g * m if g is not None else None)
                    hooks.append(h)
                    active_stream_channels += K_out
                    frozen_tail_channels += (D_out - K_out)

            elif param.dim() == 1 and param.shape[0] > min_dim_for_split:  # Bias or Norm parameter
                D = param.shape[0]
                K = max(1, int(D * ratio))
                if K < D:
                    mask = torch.zeros_like(param.data)
                    mask[:K] = 1.0
                    channel_masks[name] = mask
                    saved_weights[name] = param.data.clone()
                    param.data.mul_(mask)
                    h = param.register_hook(lambda g, m=mask: g * m if g is not None else None)
                    hooks.append(h)

        return BootstrapFreezeState(
            strategy="channel_stream",
            stream_ratio=ratio,
            saved_requires_grad=saved_requires_grad,
            saved_weights=saved_weights,
            channel_masks=channel_masks,
            hooks=hooks,
            active_stream_channels=active_stream_channels,
            frozen_tail_channels=frozen_tail_channels,
        )

    # Legacy whole-layer freeze strategies
    if strat == "whole_layer":
        model_cls_name = model.__class__.__name__.lower()
        if "diffusiondiff" in model_cls_name:
            for name, param in model.named_parameters():
                if any(k in name for k in ["vae", "diffuser"]):
                    param.requires_grad = False
        elif "unet" in model_cls_name:
            for name, param in model.named_parameters():
                if any(k in name for k in ["inc", "down", "encoder", "backbone", "features"]):
                    param.requires_grad = False
        elif "vaefinetune" in model_cls_name:
            for name, param in model.named_parameters():
                if any(k in name for k in ["encoder", "quant_conv"]):
                    param.requires_grad = False
        else:
            param_list = list(model.named_parameters())
            cutoff = int(len(param_list) * 0.65)
            for idx, (name, param) in enumerate(param_list):
                if idx < cutoff:
                    param.requires_grad = False

    elif strat in ("backbone", "encoder"):
        for name, param in model.named_parameters():
            if any(k in name.lower() for k in ["encoder", "backbone", "inc", "down", "features", "vae"]):
                param.requires_grad = False

    elif strat in ("except_head", "heads_only"):
        for name, param in model.named_parameters():
            if not any(k in name.lower() for k in ["decoder", "classifier", "head", "outc", "fc"]):
                param.requires_grad = False

    elif strat == "custom" and custom_modules:
        for name, param in model.named_parameters():
            if any(cm in name for cm in custom_modules):
                param.requires_grad = False

    return BootstrapFreezeState(
        strategy=strat,
        stream_ratio=1.0,
        saved_requires_grad=saved_requires_grad,
        saved_weights={},
        channel_masks={},
        hooks=[],
    )


def release_bootstrap_freeze(
    model: nn.Module,
    saved_states: Union[BootstrapFreezeState, Dict[str, Any]],
    release_mode: str = "restore",
) -> None:
    """Release frozen channel stream or whole-layer parameters back to their full capacity state."""
    # 1. Channel stream release
    if isinstance(saved_states, BootstrapFreezeState):
        for h in saved_states.hooks:
            try:
                h.remove()
            except Exception:
                pass
        saved_states.hooks.clear()

        # Restore or calibrate tail channels while strictly preserving the trained core stream
        params_dict = dict(model.named_parameters())
        for name, mask in saved_states.channel_masks.items():
            if name not in params_dict:
                continue
            param = params_dict[name]
            orig_data = saved_states.saved_weights.get(name, None)
            if orig_data is None:
                continue

            orig_data = orig_data.to(param.device)
            mask = mask.to(param.device)

            if release_mode == "restore":
                param.data = param.data * mask + orig_data * (1.0 - mask)
            elif release_mode == "calibrated":
                tail_init = torch.empty_like(orig_data)
                if param.dim() >= 2:
                    nn.init.kaiming_normal_(tail_init, mode="fan_in")
                else:
                    nn.init.zeros_(tail_init)
                param.data = param.data * mask + (tail_init * 0.1) * (1.0 - mask)
            elif release_mode == "zero":
                param.data = param.data * mask

        # Restore original requires_grad status
        for name, req_grad in saved_states.saved_requires_grad.items():
            if name in params_dict:
                params_dict[name].requires_grad = req_grad

    elif isinstance(saved_states, dict):
        for name, param in model.named_parameters():
            if name in saved_states:
                param.requires_grad = saved_states[name]


def is_map_style_dataset(ds: Any) -> bool:
    """Check if a dataset is a true map-style dataset supporting indexing and safe len()."""
    if ds is None or isinstance(ds, IterableDataset):
        return False
    try:
        # A map-style dataset must implement __len__ returning an int > 0
        total_len = len(ds)
        if not isinstance(total_len, int) or total_len <= 0:
            return False
        # And must support indexing without NotImplementedError, TypeError, or AttributeError
        _ = ds[0]
        return True
    except (TypeError, NotImplementedError, AttributeError, IndexError, Exception):
        return False


def _unbatch_dict_sample(batch: Dict[str, Any], target_count: int, current_count: int) -> List[Dict[str, Any]]:
    """Slice a batch dict into individual sample dictionaries."""
    batch_size = None
    if "image" in batch and hasattr(batch["image"], "shape"):
        batch_size = batch["image"].shape[0]
    elif "images" in batch and hasattr(batch["images"], "shape"):
        batch_size = batch["images"].shape[0]
    else:
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.ndim > 0:
                batch_size = v.shape[0]
                break
            elif isinstance(v, (list, tuple)) and len(v) > 0:
                batch_size = len(v)
                break
    if batch_size is None:
        batch_size = 1

    take = min(batch_size, target_count - current_count)
    unbatched = [{} for _ in range(take)]
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == batch_size:
            v_cpu = v.detach().cpu()
            for i in range(take):
                unbatched[i][k] = v_cpu[i].clone()
        elif isinstance(v, (list, tuple)) and len(v) == batch_size:
            for i in range(take):
                unbatched[i][k] = v[i]
        else:
            for i in range(take):
                unbatched[i][k] = v
    return unbatched


def _unbatch_sequence_sample(batch: Union[list, tuple], target_count: int, current_count: int) -> List[Any]:
    """Slice a batch list/tuple into individual sample tuples."""
    batch_size = None
    for elem in batch:
        if isinstance(elem, torch.Tensor) and elem.ndim > 0:
            batch_size = elem.shape[0]
            break
        elif isinstance(elem, (list, tuple)) and len(elem) > 0:
            batch_size = len(elem)
            break
    if batch_size is None:
        batch_size = 1

    take = min(batch_size, target_count - current_count)
    is_tuple = isinstance(batch, tuple)
    unbatched = [[] for _ in range(take)]
    for elem in batch:
        if isinstance(elem, torch.Tensor) and elem.ndim > 0 and elem.shape[0] == batch_size:
            elem_cpu = elem.detach().cpu()
            for i in range(take):
                unbatched[i].append(elem_cpu[i].clone())
        elif isinstance(elem, (list, tuple)) and len(elem) == batch_size:
            for i in range(take):
                unbatched[i].append(elem[i])
        else:
            for i in range(take):
                unbatched[i].append(elem)
    return [tuple(u) if is_tuple else u for u in unbatched]


def create_bootstrap_loader(
    train_loader: Any,
    num_samples: int = 512,
    batch_size: Optional[int] = None,
) -> DataLoader:
    """Extract a small sample subset (e.g. 512 or 2048 samples) for kickstarting.

    Supports both map-style datasets and streaming/iterable datasets (such as
    SIDStreamingDataset or BackgroundPrefetcher).
    """
    bs = batch_size or getattr(train_loader, "batch_size", None)
    if bs is None and hasattr(train_loader, "loader"):
        bs = getattr(train_loader.loader, "batch_size", None)
    bs = bs or 8

    dataset = getattr(train_loader, "dataset", None)
    if dataset is None and hasattr(train_loader, "loader"):
        dataset = getattr(train_loader.loader, "dataset", None)

    # 1. Map-style dataset with confirmed random-access and valid length
    if is_map_style_dataset(dataset):
        try:
            total_len = len(dataset)
            sample_count = min(num_samples, total_len)
            indices = list(range(sample_count))
            subset = Subset(dataset, indices)
            return DataLoader(
                subset,
                batch_size=bs,
                shuffle=True,
                num_workers=0,  # Avoid worker spawn overhead on tiny kickstart subset
                pin_memory=False,
            )
        except Exception as e:
            logger.debug(f"Map-style Subset creation failed: {e}; falling back to stream collection.")

    # 2. Streaming / Iterable dataset or unindexed loader: drain up to num_samples into in-memory list
    collected: List[Any] = []
    try:
        for batch in train_loader:
            if isinstance(batch, dict):
                samples = _unbatch_dict_sample(batch, num_samples, len(collected))
                collected.extend(samples)
            elif isinstance(batch, (tuple, list)):
                samples = _unbatch_sequence_sample(batch, num_samples, len(collected))
                collected.extend(samples)
            elif isinstance(batch, torch.Tensor) and batch.ndim > 0:
                b_sz = batch.shape[0]
                take = min(b_sz, num_samples - len(collected))
                b_cpu = batch.detach().cpu()
                for i in range(take):
                    collected.append(b_cpu[i].clone())
            else:
                collected.append(batch)

            if len(collected) >= num_samples:
                break
    finally:
        # Cleanly close background workers/threads (e.g. BackgroundPrefetcher or raw streams)
        if hasattr(train_loader, "close") and callable(getattr(train_loader, "close", None)):
            try:
                train_loader.close()
            except Exception:
                pass
        if dataset is not None and hasattr(dataset, "close") and callable(getattr(dataset, "close", None)):
            try:
                dataset.close()
            except Exception:
                pass

    if not collected:
        raise ValueError("Could not extract any samples from training loader for bootstrapping.")

    in_mem_ds = InMemoryListDataset(collected)
    return DataLoader(
        in_mem_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )


def run_bootstrapping_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    loss_fn: Callable[..., Any],
    config: Any,
    logger: logging.Logger,
    device: torch.device,
) -> Dict[str, Any]:
    """Execute Bootstrapping v1.0 kickstarting phase.

    1. Applies calibrated initialization to unfrozen heads/decoders.
    2. Freezes upstream parts of the model according to strategy.
    3. Trains unfrozen parameters on a small sample subset (e.g. 512 or 2048) for bootstrap_epochs.
    4. Evaluates fit, loss reduction, and target score.
    5. Utilizes nn-toolbox to notice and verify kickstart success and parameter isolation.
    6. Releases all frozen parameters early, restoring the model for full normal training.
    """
    boot_cfg = getattr(config, "bootstrapping", {}) if hasattr(config, "bootstrapping") else (
        config.get("bootstrapping", {}) if hasattr(config, "get") else {}
    )

    epochs = int(boot_cfg.get("bootstrap_epochs", boot_cfg.get("epochs", 5)))
    num_samples = int(boot_cfg.get("bootstrap_examples", boot_cfg.get("num_samples", 512)))
    freeze_strategy = str(boot_cfg.get("freeze_strategy", boot_cfg.get("strategy", "channel_stream")))
    freeze_modules = boot_cfg.get("freeze_modules", None)
    stream_ratio = float(boot_cfg.get("stream_ratio", boot_cfg.get("bootstrap_stream_ratio", 0.5)))
    release_mode = str(boot_cfg.get("release_mode", "restore"))
    init_scheme = str(boot_cfg.get("initialization", "kaiming_normal"))
    target_score = boot_cfg.get("target_score", 0.50)
    target_score = float(target_score) if target_score is not None else None
    min_loss_drop = float(boot_cfg.get("min_loss_drop", 0.15))
    early_stopping = bool(boot_cfg.get("early_stopping", False))
    patience = int(boot_cfg.get("patience", 3))
    batch_size = boot_cfg.get("batch_size", None)
    lr = boot_cfg.get("learning_rate", boot_cfg.get("bootstrap_lr", None))
    if lr is None:
        training_cfg = getattr(config, "training", {}) if hasattr(config, "training") else (
            config.get("training", {}) if hasattr(config, "get") else {}
        )
        lr = float(training_cfg.get("learning_rate", 3e-4))
    else:
        lr = float(lr)

    logger.info("=" * 70)
    logger.info("🚀 [BOOTSTRAPPING v1.0] Commencing kickstart phase")
    logger.info(f"   Configuration: epochs={epochs}, samples={num_samples}, strategy='{freeze_strategy}', stream_ratio={stream_ratio}, init='{init_scheme}'")
    logger.info(f"   Kickstart LR: {lr:.2e} | Target score: {target_score} | Min loss drop: {min_loss_drop*100:.1f}%")
    logger.info("=" * 70)

    # 1. Selectively freeze components and snapshot original requires_grad states
    saved_states = apply_bootstrap_freeze(
        model,
        strategy=freeze_strategy,
        stream_ratio=stream_ratio,
        custom_modules=freeze_modules,
    )

    if hasattr(saved_states, "strategy") and saved_states.strategy == "channel_stream":
        logger.info(
            f"🌊 [BOOTSTRAPPING v1.0] Channel Stream Active: {saved_states.active_stream_channels} core channels trainable "
            f"| {saved_states.frozen_tail_channels} tail channels frozen & zeroed across {len(saved_states.channel_masks)} layers."
        )

    # 2. Calibrate unfrozen module weights and preserve zeroed tail channels on trainable parameters
    initialize_bootstrap_parameters(
        model,
        initialization=init_scheme,
        unfrozen_only=True,
        channel_masks=getattr(saved_states, "channel_masks", None),
        saved_weights=getattr(saved_states, "saved_weights", None),
    )

    # Count parameters
    frozen_params = [name for name, p in model.named_parameters() if not p.requires_grad]
    trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
    trainable_tensors = [p for p in model.parameters() if p.requires_grad]

    logger.info(f"🔒 [BOOTSTRAPPING v1.0] Frozen parameter tensors: {len(frozen_params)} | Trainable tensors: {len(trainable_params)}")

    if not trainable_tensors:
        logger.warning("⚠️ [BOOTSTRAPPING v1.0] No parameters are trainable during kickstart! Aborting kickstart phase.")
        release_bootstrap_freeze(model, saved_states, release_mode=release_mode)
        return {
            "enabled": True,
            "run_bootstrap": True,
            "acceptable_fit": False,
            "initial_loss": 0.0,
            "final_loss": 0.0,
            "loss_drop": 0.0,
            "best_score": 0.0,
            "epochs_trained": 0,
            "bootstrap_examples": num_samples,
            "bootstrap_epochs": epochs,
        }

    # 3. Create bootstrap subset DataLoader
    logger.info(f"📦 [BOOTSTRAPPING v1.0] Assembling kickstart sample pool ({num_samples} examples)...")
    try:
        boot_loader = create_bootstrap_loader(train_loader, num_samples=num_samples, batch_size=batch_size)
        actual_samples = len(boot_loader.dataset) if hasattr(boot_loader, "dataset") and hasattr(boot_loader.dataset, "__len__") else num_samples
        logger.info(f"✅ [BOOTSTRAPPING v1.0] Kickstart sample pool ready with {actual_samples} samples.")
    except Exception as e:
        logger.warning(f"⚠️ [BOOTSTRAPPING v1.0] Failed to build subset loader: {e}; falling back to standard loader")
        boot_loader = train_loader

    # 4. Optimizer specifically for kickstarted parameters
    opt = torch.optim.AdamW(trainable_tensors, lr=lr, weight_decay=1e-4)
    warmup_epochs = min(3, max(1, epochs // 5))

    # Pre-evaluate initial baseline loss on the kickstart pool before training updates
    model.eval()
    init_loss_accum = 0.0
    init_steps = 0
    with torch.no_grad():
        for init_b in boot_loader:
            if isinstance(init_b, dict):
                bx = init_b.get("image", init_b.get("input", init_b.get("x"))).to(device)
                bm = init_b.get("mask", init_b.get("masks"))
                bl = init_b.get("label", init_b.get("labels"))
                if bm is not None and torch.is_tensor(bm):
                    bm = bm.to(device)
                if bl is not None and torch.is_tensor(bl):
                    bl = bl.to(device)
            elif isinstance(init_b, (tuple, list)):
                bx = init_b[0].to(device)
                bm = init_b[1].to(device) if len(init_b) > 1 and torch.is_tensor(init_b[1]) else None
                bl = init_b[2].to(device) if len(init_b) > 2 and torch.is_tensor(init_b[2]) else None
            else:
                bx = init_b.to(device)
                bm, bl = None, None

            b_out = model(bx)
            if bm is not None:
                if bl is not None:
                    b_l, _ = loss_fn(b_out, bm, bl)
                else:
                    b_l, _ = loss_fn(b_out, bm)
            else:
                b_l = loss_fn(b_out)
            if isinstance(b_l, (tuple, list)):
                b_l = b_l[0]
            init_loss_accum += float(b_l.item())
            init_steps += 1
            if init_steps >= min(4, len(boot_loader) if hasattr(boot_loader, "__len__") else 4):
                break
    initial_loss = (init_loss_accum / max(1, init_steps)) if init_steps > 0 else None
    model.train()

    # 5. Training loop across kickstart epochs
    history = []
    smoothed_loss = None
    best_loss = float("inf")
    best_score = 0.0
    stagnant_epochs = 0
    consecutive_fit_epochs = 0
    grad_leak = False
    min_bootstrap_epochs = min(epochs, max(2, min(5, epochs // 3)))

    for ep in range(1, epochs + 1):
        # Linear learning rate warmup across early kickstart epochs to prevent gradient shocks
        warmup_factor = min(1.0, float(ep) / float(max(1, warmup_epochs)))
        current_lr = lr * (0.2 + 0.8 * warmup_factor)
        for pg in opt.param_groups:
            pg["lr"] = current_lr

        running_loss = 0.0
        running_iou = 0.0
        step_count = 0

        for batch in boot_loader:
            # Unpack batch
            if isinstance(batch, dict):
                x = batch.get("image", batch.get("input", batch.get("x"))).to(device)
                m = batch.get("mask", batch.get("masks"))
                l = batch.get("label", batch.get("labels"))
                if m is not None and torch.is_tensor(m):
                    m = m.to(device)
                if l is not None and torch.is_tensor(l):
                    l = l.to(device)
            elif isinstance(batch, (tuple, list)):
                x = batch[0].to(device)
                m = batch[1].to(device) if len(batch) > 1 and torch.is_tensor(batch[1]) else None
                l = batch[2].to(device) if len(batch) > 2 and torch.is_tensor(batch[2]) else None
            else:
                x = batch.to(device)
                m, l = None, None

            opt.zero_grad()
            out = model(x)

            # Compute loss
            if m is not None:
                if l is not None:
                    loss, metrics_dict = loss_fn(out, m, l)
                else:
                    loss, metrics_dict = loss_fn(out, m)
            else:
                loss = loss_fn(out)
                metrics_dict = {}

            if isinstance(loss, (tuple, list)):
                loss = loss[0]

            loss_val = float(loss.item())
            running_loss += loss_val

            # Compute IoU if segmentation logits
            iou_val = float(metrics_dict.get("iou", 0.0))
            if iou_val == 0.0 and m is not None and torch.is_tensor(m):
                out_t = out[0] if isinstance(out, (tuple, list)) else out
                pred_binary = (torch.sigmoid(out_t) > 0.5).float()
                intersection = (pred_binary * m).sum().item()
                union = (pred_binary + m).clamp(0, 1).sum().item()
                iou_val = (intersection + 1e-6) / (union + 1e-6)
            running_iou += iou_val
            step_count += 1

            loss.backward()

            # Verify freeze isolation during backward pass
            if hasattr(saved_states, "channel_masks") and saved_states.channel_masks:
                for name, p in model.named_parameters():
                    if name in saved_states.channel_masks and p.grad is not None:
                        mask = saved_states.channel_masks[name]
                        frozen_grad = p.grad * (1.0 - mask)
                        if float(frozen_grad.abs().sum().item()) > 1e-9:
                            grad_leak = True
            else:
                for name, p in model.named_parameters():
                    if name in frozen_params and p.grad is not None:
                        if float(p.grad.abs().sum().item()) > 1e-9:
                            grad_leak = True

            torch.nn.utils.clip_grad_norm_(trainable_tensors, max_norm=1.0)
            opt.step()

        avg_loss = running_loss / max(1, step_count)
        avg_iou = running_iou / max(1, step_count)

        if initial_loss is None:
            initial_loss = avg_loss

        # Exponential moving average smoothing of kickstart loss
        smoothed_loss = avg_loss if smoothed_loss is None else (0.6 * smoothed_loss + 0.4 * avg_loss)
        cur_drop = (initial_loss - smoothed_loss) / initial_loss if initial_loss > 1e-8 else 0.0
        best_score = max(best_score, avg_iou)

        logger.info(
            f"   Epoch [{ep}/{epochs}] - Kickstart Loss: {avg_loss:.4f} (Smoothed: {smoothed_loss:.4f}, Drop: {cur_drop*100:+.1f}%) | "
            f"Kickstart IoU/Score: {avg_iou:.4f} | Best Score: {best_score:.4f}"
        )

        history.append({
            "epoch": ep,
            "loss": avg_loss,
            "smoothed_loss": smoothed_loss,
            "iou": avg_iou,
            "loss_drop": cur_drop,
        })

        # Check early stopping / acceptable fit
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            stagnant_epochs = 0
        else:
            stagnant_epochs += 1

        reached_target_score = (target_score is not None and best_score >= target_score)
        reached_loss_drop = (cur_drop >= min_loss_drop)

        if reached_loss_drop or reached_target_score:
            consecutive_fit_epochs += 1
        else:
            consecutive_fit_epochs = 0

        # Only stop early if explicitly configured and sustained across burn-in epochs
        if early_stopping and consecutive_fit_epochs >= 2 and ep >= min_bootstrap_epochs:
            logger.info(
                f"🎯 [BOOTSTRAPPING v1.0] Acceptable kickstart fit reached and sustained for {consecutive_fit_epochs} epochs at epoch {ep} "
                f"(Loss drop: {cur_drop*100:.1f}%, Score: {best_score:.4f})."
            )
            break

        if early_stopping and stagnant_epochs >= patience and ep >= min_bootstrap_epochs:
            logger.info(f"🛑 [BOOTSTRAPPING v1.0] Kickstart plateau reached after {patience} stagnant epochs at epoch {ep}; completing kickstarting.")
            break

    final_loss = history[-1]["loss"] if history else initial_loss or 0.0
    initial_loss_val = initial_loss if initial_loss is not None else final_loss
    loss_drop_final = (initial_loss_val - final_loss) / initial_loss_val if initial_loss_val > 1e-8 else 0.0
    acceptable_fit = (loss_drop_final >= min_loss_drop or (target_score is not None and best_score >= target_score)) and not grad_leak

    summary: Dict[str, Any] = {
        "enabled": True,
        "run_bootstrap": True,
        "strategy": freeze_strategy,
        "stream_ratio": stream_ratio,
        "active_stream_channels": getattr(saved_states, "active_stream_channels", 0),
        "frozen_tail_channels": getattr(saved_states, "frozen_tail_channels", 0),
        "initial_loss": float(initial_loss_val),
        "final_loss": float(final_loss),
        "loss_drop": float(loss_drop_final),
        "loss_drop_pct": float(loss_drop_final * 100),
        "best_score": float(best_score),
        "target_score": target_score,
        "min_loss_drop": min_loss_drop,
        "epochs_trained": len(history),
        "bootstrap_epochs": epochs,
        "bootstrap_examples": num_samples,
        "frozen_param_count": len(frozen_params),
        "trainable_param_count": len(trainable_params),
        "frozen_param_grad_leak": grad_leak,
        "acceptable_fit": acceptable_fit,
        "history": history,
    }

    # 6. Consult nn-toolbox to notice and verify kickstart success
    logger.info("🔬 [BOOTSTRAPPING v1.0] Consulting nn-toolbox to evaluate kickstart fit and isolation integrity...")
    try:
        from nn_toolbox.experiments.bootstrapping import verify_bootstrapping
        tb_result = verify_bootstrapping(
            model=model,
            bootstrap_results=summary,
            min_loss_drop=min_loss_drop,
            target_score=target_score,
        )
        findings = tb_result.get("findings", [])
        summary["diagnostic_findings"] = [f.to_dict() if hasattr(f, "to_dict") else dict(f) for f in findings]
        summary["nn_toolbox_verified"] = bool(tb_result.get("acceptable_fit", acceptable_fit))

        for f in findings:
            sev = getattr(f, "severity", "info").upper()
            obs = getattr(f, "observation", str(f))
            if sev == "INFO":
                logger.info(f"   ✓ [nn-toolbox] [{sev}] {obs}")
            elif sev == "WARNING":
                logger.warning(f"   ! [nn-toolbox] [{sev}] {obs}")
            else:
                logger.error(f"   ✗ [nn-toolbox] [{sev}] {obs}")

    except ImportError:
        logger.info("   ℹ️ [nn-toolbox] Package not found; using internal verification.")
        summary["nn_toolbox_verified"] = acceptable_fit
    except Exception as e:
        logger.warning(f"   ⚠️ [nn-toolbox] Error during bootstrap assessment: {e}")
        summary["nn_toolbox_verified"] = acceptable_fit

    # 7. Release frozen parameters early, restoring original state for normal training
    logger.info("🔓 [BOOTSTRAPPING v1.0] Releasing all frozen channel streams/parameters back to original configuration for normal training.")
    release_bootstrap_freeze(model, saved_states, release_mode=release_mode)

    active_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✨ [BOOTSTRAPPING v1.0] Kickstarting complete. Total active parameters for normal training: {active_now:,}")
    logger.info("=" * 70)

    return summary
