"""
Tests for FOCAL Strategy B (Prior-Preserving Parameter-Efficient Model Surgery at 512x512).

Covers all acceptance criteria from `docs/focal_finetuning_strategy_b_requirements.md`:
1. Parameter freezing & no full-backbone leakage (ViT PEFT LoRA/DoRA + HRNet stage-wise & BN freezing).
2. Safe `set_backbone_trainable(False/True)` and `train(True)` invariant across Epoch 1 -> Epoch 2 warmup.
3. Native 512x512 forward/backward pass for Mode 1A (`supervised`) and Mode 1B (`contrastive`).
4. Prior-drift containment (`prior_anchor_weight > 0`, `neck_l2_anchor_weight > 0`, bit-identical frozen base weights).
5. Zero-initialized `FeatureAdapter` (`F + Adapter(F) == F` at step 0).
6. Optimizer parameter groups (`head`, `lora`, `neck`/`backbone_partial`) with LR scaling & reduced WD.
7. Checkpoint save/reload & lossless `merge_peft_weights()` round-trip parity (LoRA & DoRA).
8. AMP / TorchAO FP8 safety (`fp8_linear_module_filter` excluding `.base_linear`) & DDP `requires_grad` check.
9. Strategy B YAML configs verification.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sid_unet.losses import SIDTotalLoss, build_loss
from sid_unet.models.focal import FOCAL, FeatureAdapter
from sid_unet.models.mask_clip import SurgicalLoRALinear
from sid_unet.models.unet import UNet
from sid_unet.training.trainer import Trainer
from sid_unet.utils.config import load_config
from sid_unet.utils.precision import (
    fp8_linear_module_filter,
    prepare_model_for_fp8_training,
    resolve_precision_spec,
)


class _SyntheticSegDataset512(Dataset):
    """Deterministic synthetic dataset with clear tampered region."""

    def __init__(self, n_samples: int = 4, size: int = 512):
        self.n_samples = n_samples
        self.size = size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        img = torch.full((3, self.size, self.size), 0.25, dtype=torch.float32)
        mask = torch.zeros((1, self.size, self.size), dtype=torch.float32)
        q = self.size // 4
        img[:, q : 3 * q, q : 3 * q] = 0.85
        mask[:, q : 3 * q, q : 3 * q] = 1.0
        return {
            "image": img,
            "mask": mask,
            "label": torch.tensor(2, dtype=torch.long),
        }


class TestFocalStrategyBFreezingAndSurgery:
    """Requirement 1 & 3: ViT PEFT surgery, HRNet stage/BN locking, and Epoch 2 freeze safety."""

    @pytest.mark.timeout(180)
    def test_vit_peft_dora_freezing_scope_and_param_footprint(self):
        model = FOCAL(
            backbone="vit",
            pretrained=True,
            finetune_mode="supervised",
            force_vit_1024=False,
            peft_enabled=True,
            lora_r=8,
            lora_alpha=16.0,
            lora_dropout=0.05,
            use_dora=True,
            lora_block_scope="late_and_global",
            lora_target_modules=["qkv", "proj"],
            train_neck=True,
            train_vit_norms=False,
        )

        # Verify adapted block indices = {5, 11, 17, 18, 19, 20, 21, 22, 23}
        assert model._adapted_block_indices == [5, 11, 17, 18, 19, 20, 21, 22, 23]

        enc = model.net.image_encoder
        # 1. patch_embed, pos_embed, rel_pos_h/w MUST be frozen
        assert all(not p.requires_grad for p in enc.patch_embed.parameters())
        assert not enc.pos_embed.requires_grad
        for idx, blk in enumerate(enc.blocks):
            assert not blk.attn.rel_pos_h.requires_grad
            assert not blk.attn.rel_pos_w.requires_grad
            assert all(not p.requires_grad for p in blk.norm1.parameters())
            assert all(not p.requires_grad for p in blk.norm2.parameters())
            if idx in model._adapted_block_indices:
                assert isinstance(blk.attn.qkv, SurgicalLoRALinear)
                assert isinstance(blk.attn.proj, SurgicalLoRALinear)
                assert blk.attn.qkv.use_dora is True
                assert all(not p.requires_grad for p in blk.attn.qkv.base_linear.parameters())
                assert all(not p.requires_grad for p in blk.attn.proj.base_linear.parameters())
                assert blk.attn.qkv.lora_A.requires_grad
                assert blk.attn.qkv.lora_B.requires_grad
                assert blk.attn.qkv.dora_m.requires_grad
                # MLP layers untouched by default
                assert not isinstance(blk.mlp.lin1, SurgicalLoRALinear)
                assert all(not p.requires_grad for p in blk.mlp.parameters())
            else:
                assert not isinstance(blk.attn.qkv, SurgicalLoRALinear)
                assert all(not p.requires_grad for p in blk.parameters())

        # 2. neck is trainable when train_neck=True
        assert all(p.requires_grad for p in enc.neck.parameters())

        # 3. Backbone trainable params < 1.0% of FOCALViTNet
        summary = model.get_surgery_summary()
        assert summary["backbone_trainable_pct"] < 1.0
        assert summary["backbone_trainable_params"] > 0

        # 4. Epoch 1 -> Epoch 2 transition: set_backbone_trainable(False) then (True)
        model.set_backbone_trainable(False)
        assert all(not p.requires_grad for p in model.net.parameters())
        assert all(p.requires_grad for p in model.decoder.parameters())

        model.set_backbone_trainable(True)
        assert all(not p.requires_grad for p in enc.patch_embed.parameters())
        assert not enc.pos_embed.requires_grad
        assert all(not p.requires_grad for p in enc.blocks[0].parameters())
        assert all(not p.requires_grad for p in enc.blocks[5].attn.qkv.base_linear.parameters())
        assert enc.blocks[5].attn.qkv.lora_A.requires_grad
        assert all(p.requires_grad for p in enc.neck.parameters())

    @pytest.mark.timeout(180)
    def test_vit_block_scope_variants_and_train_neck_false(self):
        model = FOCAL(
            backbone="vit",
            pretrained=True,
            finetune_mode="contrastive",
            peft_enabled=True,
            lora_block_scope="global_only",
            lora_target_modules=["qkv", "proj", "lin1", "lin2"],
            train_neck=False,
        )
        assert model._adapted_block_indices == [5, 11, 17, 23]
        enc = model.net.image_encoder
        assert all(not p.requires_grad for p in enc.neck.parameters())
        assert isinstance(enc.blocks[5].mlp.lin1, SurgicalLoRALinear)
        assert isinstance(enc.blocks[5].mlp.lin2, SurgicalLoRALinear)
        assert not isinstance(enc.blocks[6].attn.qkv, SurgicalLoRALinear)

    @pytest.mark.timeout(120)
    def test_hrnet_stage_freezing_bn_eval_lock_and_zero_init_adapter(self):
        model = FOCAL(
            backbone="hrnet",
            pretrained=True,
            finetune_mode="supervised",
            hrnet_trainable_stages=["stage4"],
            freeze_backbone_bn=True,
            use_feature_adapter=True,
        )
        assert isinstance(model.feature_adapter, FeatureAdapter)

        # Verify frozen stages and BN eval lock
        assert all(not p.requires_grad for p in model.net.conv1.parameters())
        assert all(not p.requires_grad for p in model.net.stage2.parameters())
        assert all(not p.requires_grad for p in model.net.stage3.parameters())
        assert any(p.requires_grad for p in model.net.stage4.parameters())
        assert len(model._frozen_bn_modules) > 0

        # Calling model.train(True) MUST keep frozen BNs in .eval() and requires_grad=False
        model.train(True)
        assert all(not bn.training for bn in model._frozen_bn_modules)
        assert all(not p.requires_grad for bn in model._frozen_bn_modules for p in bn.parameters())

        # Epoch 1 -> Epoch 2 transition
        model.set_backbone_trainable(False)
        assert all(not p.requires_grad for p in model.net.parameters())
        assert all(p.requires_grad for p in model.feature_adapter.parameters())
        model.set_backbone_trainable(True)
        assert all(not p.requires_grad for p in model.net.stage3.parameters())
        assert any(p.requires_grad for p in model.net.stage4.parameters())
        assert all(not bn.training for bn in model._frozen_bn_modules)

        # Verify zero-initialized FeatureAdapter identity at step 0
        x_feat = torch.randn(2, 32, 32, 32)
        assert torch.equal(model.feature_adapter(x_feat), x_feat)


class TestFocalStrategyBForwardBackwardAndPriorAnchor:
    """Requirements 2 & 4: Native 512x512 forward/backward, prior anchoring, and optimizer groups."""

    @pytest.mark.timeout(240)
    @pytest.mark.parametrize("mode", ["supervised", "contrastive"])
    def test_vit_512_native_forward_backward_and_ddp_unused_param_safety(self, mode):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FOCAL(
            backbone="vit",
            pretrained=True,
            finetune_mode=mode,
            force_vit_1024=False,
            peft_enabled=True,
            lora_r=8,
            lora_alpha=16.0,
            use_dora=True,
            lora_block_scope="late_and_global",
            train_neck=True,
            prior_anchor_weight=0.1 if mode == "contrastive" else 0.0,
        ).to(device)
        model.train()

        x = torch.rand(1, 3, 512, 512, device=device)
        gt = torch.zeros(1, 1, 512, 512, device=device)
        gt[:, :, 128:384, 128:384] = 1.0

        out = model(x, return_dict=True, gt_masks=gt if mode == "contrastive" else None)
        assert out["features"].shape == (1, 256, 32, 32)
        assert out["mask_logits"].shape == (1, 1, 512, 512)

        loss_fn = SIDTotalLoss(aux_classifier=False, artifact_weight=1.0).to(device)
        loss, loss_metrics = loss_fn(out, gt)
        assert torch.isfinite(loss)
        loss.backward()

        # Every requires_grad=True parameter MUST receive a gradient (DDP find_unused_parameters=False safe)
        missing_grads = [
            n for n, p in model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert missing_grads == [], f"Unused trainable parameters detected: {missing_grads}"

    @pytest.mark.timeout(240)
    def test_prior_drift_containment_and_frozen_base_bit_identical(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FOCAL(
            backbone="vit",
            pretrained=True,
            finetune_mode="contrastive",
            force_vit_1024=False,
            peft_enabled=True,
            lora_r=8,
            lora_alpha=16.0,
            use_dora=True,
            lora_block_scope="late_and_global",
            train_neck=True,
            prior_anchor_weight=0.1,
            neck_l2_anchor_weight=0.05,
            neck_lr_multiplier=0.25,
            lora_weight_decay=0.01,
        ).to(device)
        model.train()

        # Snapshot frozen base weights before optimizer steps
        enc = model.net.image_encoder
        frozen_snap = {
            "patch_embed": enc.patch_embed.proj.weight.detach().clone(),
            "pos_embed": enc.pos_embed.detach().clone(),
            "rel_pos_h_5": enc.blocks[5].attn.rel_pos_h.detach().clone(),
            "qkv_base_5": enc.blocks[5].attn.qkv.base_linear.weight.detach().clone(),
            "early_blk_0": enc.blocks[0].attn.qkv.weight.detach().clone(),
        }
        lora_b_before = enc.blocks[5].attn.qkv.lora_B.detach().clone()
        neck_before = enc.neck[0].weight.detach().clone()

        param_groups = model.get_param_groups(lr=1e-4)
        group_names = {g["name"]: g for g in param_groups}
        assert "lora" in group_names
        assert "neck" in group_names
        assert group_names["lora"]["weight_decay"] == pytest.approx(0.01)
        assert group_names["neck"]["lr"] == pytest.approx(2.5e-5)

        opt = torch.optim.AdamW(param_groups)
        loss_fn = SIDTotalLoss(aux_classifier=False, artifact_weight=1.0).to(device)

        x = torch.rand(1, 3, 256, 256, device=device)
        gt = torch.zeros(1, 1, 256, 256, device=device)
        gt[:, :, 64:192, 64:192] = 1.0

        anchor_losses = []
        for _ in range(3):
            opt.zero_grad()
            out = model(x, return_dict=True, gt_masks=gt)
            loss, metrics = loss_fn(out, gt)
            assert "prior_anchor_loss" in metrics
            anchor_losses.append(metrics["prior_anchor_loss"])
            loss.backward()
            opt.step()

        # Base weights MUST remain bit-identical
        assert torch.equal(enc.patch_embed.proj.weight, frozen_snap["patch_embed"])
        assert torch.equal(enc.pos_embed, frozen_snap["pos_embed"])
        assert torch.equal(enc.blocks[5].attn.rel_pos_h, frozen_snap["rel_pos_h_5"])
        assert torch.equal(enc.blocks[5].attn.qkv.base_linear.weight, frozen_snap["qkv_base_5"])
        assert torch.equal(enc.blocks[0].attn.qkv.weight, frozen_snap["early_blk_0"])

        # Adapters & neck updated, and anchor loss became strictly positive after step 0
        assert not torch.equal(enc.blocks[5].attn.qkv.lora_B, lora_b_before)
        assert not torch.equal(enc.neck[0].weight, neck_before)
        assert anchor_losses[-1] > 0.0 and torch.isfinite(torch.tensor(anchor_losses[-1]))

    @pytest.mark.timeout(240)
    def test_hrnet_trainer_strategy_b_epoch2_warmup_preserves_frozen_stages(self, tmp_path):
        cfg = load_config("configs/experiments/focal/focal_hrnet_peft.yaml")
        cfg.project.output_dir = str(tmp_path / "run_hrnet_b")
        cfg.project.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.data.image_size = [128, 128]
        cfg.data.batch_size = 2
        cfg.training.epochs = 2
        cfg.training.freeze_backbone_epochs = 1
        cfg.training.gradient_accumulation_steps = 1
        cfg.training.auto_batch_size = False
        cfg.training.early_stopping_patience = 10

        ds = _SyntheticSegDataset512(n_samples=4, size=128)
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        trainer = Trainer(config=cfg, train_loader=loader, val_loader=loader)

        raw = getattr(trainer.model, "module", trainer.model)
        conv1_before = raw.net.conv1.weight.detach().clone()
        bn1_mean_before = raw.net.bn1.running_mean.detach().clone()

        res = trainer.train()
        assert len(res["history"]) == 2
        # Even after Epoch 2 unfreezing, frozen stem/stages and BN stats must remain untouched
        assert torch.equal(raw.net.conv1.weight, conv1_before)
        assert torch.equal(raw.net.bn1.running_mean, bn1_mean_before)
        assert all(not p.requires_grad for p in raw.net.stage3.parameters())
        assert any(p.requires_grad for p in raw.net.stage4.parameters())


class TestFocalStrategyBMergeCheckpointAndFP8:
    """Requirement 5: Checkpoint reload, lossless merge_peft_weights, and TorchAO FP8 safety."""

    @pytest.mark.timeout(240)
    @pytest.mark.parametrize("use_dora", [True, False])
    def test_checkpoint_reload_and_merge_peft_weights_parity(self, tmp_path, use_dora):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FOCAL(
            backbone="vit",
            pretrained=True,
            finetune_mode="supervised",
            force_vit_1024=False,
            peft_enabled=True,
            lora_r=4,
            lora_alpha=8.0,
            lora_dropout=0.0,
            use_dora=use_dora,
            lora_block_scope=[23],
            lora_target_modules=["qkv", "proj"],
        ).to(device)

        # Perturb LoRA/DoRA weights so merge math is non-trivial
        with torch.no_grad():
            qkv = model.net.image_encoder.blocks[23].attn.qkv
            qkv.lora_B.fill_(0.02)
            if use_dora and qkv.dora_m is not None:
                qkv.dora_m.mul_(1.05)

        ckpt_path = tmp_path / f"focal_vit_peft_{use_dora}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "model": {
                        "name": "focal",
                        "backbone": "vit",
                        "pretrained": True,
                        "finetune_mode": "supervised",
                        "force_vit_1024": False,
                        "peft_enabled": True,
                        "lora_r": 4,
                        "lora_alpha": 8.0,
                        "lora_dropout": 0.0,
                        "use_dora": use_dora,
                        "lora_block_scope": [23],
                        "lora_target_modules": ["qkv", "proj"],
                    }
                },
            },
            str(ckpt_path),
        )

        reloaded = UNet.from_checkpoint(str(ckpt_path), device=device)
        reloaded.eval()
        model.eval()

        x = torch.rand(1, 3, 256, 256, device=device)
        with torch.no_grad():
            out_unmerged = reloaded(x, return_dict=True)["mask_logits"]

        reloaded.merge_peft_weights()
        assert not any(isinstance(m, SurgicalLoRALinear) for m in reloaded.modules())

        with torch.no_grad():
            out_merged = reloaded(x, return_dict=True)["mask_logits"]

        assert torch.allclose(out_unmerged, out_merged, atol=2e-3, rtol=2e-3)
        assert (out_unmerged - out_merged).abs().mean().item() < 5e-4

    @pytest.mark.timeout(240)
    @pytest.mark.parametrize("use_dora", [True, False])
    def test_fp8_training_excludes_base_linear_and_runs_cleanly(self, use_dora):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            pytest.skip("CUDA required for FP8 / AMP verification.")

        model = FOCAL(
            backbone="vit",
            pretrained=True,
            finetune_mode="supervised",
            force_vit_1024=False,
            peft_enabled=True,
            lora_r=8,
            lora_alpha=16.0,
            use_dora=use_dora,
            lora_block_scope=[23],
            lora_target_modules=["qkv", "proj"],
        ).to(device)

        # Filter guard unit check
        qkv_wrapper = model.net.image_encoder.blocks[23].attn.qkv
        assert fp8_linear_module_filter(qkv_wrapper.base_linear, "net.image_encoder.blocks.23.attn.qkv.base_linear") is False
        assert fp8_linear_module_filter(model.net.image_encoder.blocks[0].attn.qkv, "net.image_encoder.blocks.0.attn.qkv") is True

        spec = resolve_precision_spec(mode="auto", device=device)
        fp8_info = prepare_model_for_fp8_training(model, spec)
        if spec.fp8_enabled:
            assert fp8_info["converted_linear_layers"] > 0
            assert not ("Float8Linear" in type(qkv_wrapper.base_linear).__name__)

        model.train()
        x = torch.rand(1, 3, 256, 256, device=device)
        gt = torch.zeros(1, 1, 256, 256, device=device)
        gt[:, :, 64:192, 64:192] = 1.0

        loss_fn = SIDTotalLoss(aux_classifier=False).to(device)
        with torch.amp.autocast("cuda", dtype=spec.amp_dtype, enabled=spec.amp_enabled):
            out = model(x, return_dict=True)
            loss, _ = loss_fn(out, gt)
        assert torch.isfinite(loss)
        loss.backward()
        assert qkv_wrapper.lora_B.grad is not None


class TestFocalStrategyBConfigs:
    """Validate Strategy B YAML experiment configs."""

    @pytest.mark.parametrize(
        "cfg_path,expected_backbone,expected_mode,expected_peft",
        [
            ("configs/experiments/focal/focal_vit.yaml", "vit", "supervised", True),
            ("configs/experiments/focal/focal_vit_contrastive.yaml", "vit", "contrastive", True),
            ("configs/experiments/focal/focal_vit_peft.yaml", "vit", "supervised", True),
            ("configs/experiments/focal/focal_vit_contrastive_peft.yaml", "vit", "contrastive", True),
            ("configs/experiments/focal/focal_hrnet_peft.yaml", "hrnet", "supervised", True),
            ("configs/experiments/focal/focal_hrnet_contrastive_peft.yaml", "hrnet", "contrastive", True),
        ],
    )
    def test_strategy_b_configs_valid(self, cfg_path, expected_backbone, expected_mode, expected_peft):
        cfg = load_config(cfg_path)
        assert cfg.data.normalize is False
        assert list(cfg.data.image_size) == [512, 512]
        assert cfg.model.backbone == expected_backbone
        assert cfg.model.finetune_mode == expected_mode
        assert bool(cfg.model.peft_enabled) is expected_peft
        assert cfg.training.precision == "auto"
        if expected_backbone == "vit":
            assert cfg.model.force_vit_1024 is False
            assert cfg.model.use_dora is True
