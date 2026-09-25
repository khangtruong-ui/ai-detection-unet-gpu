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
from torch.utils.data import DataLoader, Dataset, Subset

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


def apply_bootstrap_freeze(
    model: nn.Module,
    strategy: str = "auto",
    custom_modules: Optional[List[str]] = None,
) -> Dict[str, bool]:
    """Selectively freeze upstream model components (e.g. encoder/backbone) for kickstarting.

    Saves the exact original `requires_grad` state of each parameter so they can be
    cleanly released upon kickstart completion.

    Strategies:
    - 'auto': Architecture-aware detection:
        * DiffusionDiff / DiffusionDiffV2: Freezes VAE encoder and Diffuser UNet;
          keeps trainable decoder, z_norm, and auxiliary classifier head unfrozen.
        * UNet: Freezes encoder stages (inc, down1-down4, features);
          keeps decoder stages (up1-up4, outc) and auxiliary head unfrozen.
        * DiffusionVAEFinetune: Freezes encoder; keeps decoder unfrozen.
        * Generic: Freezes the first ~65% of parameter depth.
    - 'backbone' or 'encoder': Freezes all layers matching encoder, backbone, vae, inc, down.
    - 'except_head' or 'heads_only': Freezes everything except decoder, classifier, head, outc.
    - 'custom': Freezes layers matching names in `custom_modules`.

    Returns:
        Dict[str, bool]: Mapping of parameter name -> original requires_grad state.
    """
    saved_states: Dict[str, bool] = {}
    for name, param in model.named_parameters():
        saved_states[name] = param.requires_grad

    strat = strategy.lower()

    if strat == "auto":
        model_cls_name = model.__class__.__name__.lower()
        if "diffusiondiff" in model_cls_name:
            # Freeze vae and diffuser unet, keep decoder and classifier trainable
            for name, param in model.named_parameters():
                if any(k in name for k in ["vae", "diffuser"]):
                    param.requires_grad = False
        elif "unet" in model_cls_name:
            # Freeze UNet encoder stages, keep decoder and classifier trainable
            for name, param in model.named_parameters():
                if any(k in name for k in ["inc", "down", "encoder", "backbone", "features"]):
                    param.requires_grad = False
        elif "vaefinetune" in model_cls_name:
            for name, param in model.named_parameters():
                if any(k in name for k in ["encoder", "quant_conv"]):
                    param.requires_grad = False
        else:
            # Generic fallback: freeze first 65% of parameter tensors
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

    return saved_states


def release_bootstrap_freeze(
    model: nn.Module,
    saved_states: Dict[str, bool],
) -> None:
    """Release frozen parameters back to their configured pre-bootstrapping states."""
    for name, param in model.named_parameters():
        if name in saved_states:
            param.requires_grad = saved_states[name]


def create_bootstrap_loader(
    train_loader: DataLoader,
    num_samples: int = 512,
    batch_size: Optional[int] = None,
) -> DataLoader:
    """Extract a small sample subset (e.g. 512 or 2048 samples) for kickstarting.

    Supports both map-style datasets and streaming/iterable datasets.
    """
    bs = batch_size or train_loader.batch_size or 8
    dataset = train_loader.dataset

    # Map-style dataset with known length
    if hasattr(dataset, "__len__"):
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

    # Streaming / Iterable dataset: drain the first num_samples into in-memory list
    collected = []
    for batch in train_loader:
        if isinstance(batch, dict):
            b_sz = next(iter(batch.values())).shape[0] if hasattr(next(iter(batch.values())), "shape") else 1
            for i in range(b_sz):
                item = {k: v[i] if hasattr(v, "__getitem__") and hasattr(v, "shape") else v for k, v in batch.items()}
                collected.append(item)
                if len(collected) >= num_samples:
                    break
        elif isinstance(batch, (tuple, list)):
            b_sz = batch[0].shape[0] if hasattr(batch[0], "shape") else 1
            for i in range(b_sz):
                item = tuple(elem[i] if hasattr(elem, "__getitem__") and hasattr(elem, "shape") else elem for elem in batch)
                collected.append(item)
                if len(collected) >= num_samples:
                    break
        else:
            collected.append(batch)

        if len(collected) >= num_samples:
            break

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
    freeze_strategy = str(boot_cfg.get("freeze_strategy", "auto"))
    freeze_modules = boot_cfg.get("freeze_modules", None)
    init_scheme = str(boot_cfg.get("initialization", "kaiming_normal"))
    target_score = boot_cfg.get("target_score", 0.50)
    target_score = float(target_score) if target_score is not None else None
    min_loss_drop = float(boot_cfg.get("min_loss_drop", 0.15))
    early_stopping = bool(boot_cfg.get("early_stopping", True))
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
    logger.info(f"   Configuration: epochs={epochs}, samples={num_samples}, strategy='{freeze_strategy}', init='{init_scheme}'")
    logger.info(f"   Kickstart LR: {lr:.2e} | Target score: {target_score} | Min loss drop: {min_loss_drop*100:.1f}%")
    logger.info("=" * 70)

    # 1. Selectively freeze components and snapshot original requires_grad states
    saved_states = apply_bootstrap_freeze(model, strategy=freeze_strategy, custom_modules=freeze_modules)

    # 2. Calibrate unfrozen module weights
    initialize_bootstrap_parameters(model, initialization=init_scheme, unfrozen_only=True)

    # Count parameters
    frozen_params = [name for name, p in model.named_parameters() if not p.requires_grad]
    trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
    trainable_tensors = [p for p in model.parameters() if p.requires_grad]

    logger.info(f"🔒 [BOOTSTRAPPING v1.0] Frozen parameter tensors: {len(frozen_params)} | Trainable tensors: {len(trainable_params)}")

    if not trainable_tensors:
        logger.warning("⚠️ [BOOTSTRAPPING v1.0] No parameters are trainable during kickstart! Aborting kickstart phase.")
        release_bootstrap_freeze(model, saved_states)
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
    except Exception as e:
        logger.warning(f"⚠️ [BOOTSTRAPPING v1.0] Failed to build subset loader: {e}; falling back to standard loader")
        boot_loader = train_loader

    # 4. Optimizer specifically for kickstarted parameters
    opt = torch.optim.AdamW(trainable_tensors, lr=lr, weight_decay=1e-4)

    # 5. Training loop across kickstart epochs
    model.train()
    history = []
    initial_loss = None
    best_loss = float("inf")
    best_score = 0.0
    stagnant_epochs = 0
    grad_leak = False

    for ep in range(1, epochs + 1):
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

        cur_drop = (initial_loss - avg_loss) / initial_loss if initial_loss > 1e-8 else 0.0
        best_score = max(best_score, avg_iou)

        logger.info(
            f"   Epoch [{ep}/{epochs}] - Kickstart Loss: {avg_loss:.4f} (Drop: {cur_drop*100:+.1f}%) | "
            f"Kickstart IoU/Score: {avg_iou:.4f} | Best Score: {best_score:.4f}"
        )

        history.append({
            "epoch": ep,
            "loss": avg_loss,
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

        if early_stopping and (reached_loss_drop or reached_target_score) and ep >= 2:
            logger.info(f"🎯 [BOOTSTRAPPING v1.0] Acceptable kickstart fit reached at epoch {ep} (Loss drop: {cur_drop*100:.1f}%, Score: {best_score:.4f}).")
            break

        if early_stopping and stagnant_epochs >= patience:
            logger.info(f"⏹️ [BOOTSTRAPPING v1.0] Kickstart plateau reached at epoch {ep}; completing kickstarting early.")
            break

    final_loss = history[-1]["loss"] if history else initial_loss or 0.0
    initial_loss_val = initial_loss if initial_loss is not None else final_loss
    loss_drop_final = (initial_loss_val - final_loss) / initial_loss_val if initial_loss_val > 1e-8 else 0.0
    acceptable_fit = (loss_drop_final >= min_loss_drop or (target_score is not None and best_score >= target_score)) and not grad_leak

    summary: Dict[str, Any] = {
        "enabled": True,
        "run_bootstrap": True,
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
    logger.info("🔓 [BOOTSTRAPPING v1.0] Releasing all frozen parameters back to original configuration for normal training.")
    release_bootstrap_freeze(model, saved_states)

    active_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✨ [BOOTSTRAPPING v1.0] Kickstarting complete. Total active parameters for normal training: {active_now:,}")
    logger.info("=" * 70)

    return summary
