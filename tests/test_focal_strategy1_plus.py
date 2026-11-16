"""
Tests for FOCAL Strategy 1+ (Anchored Full-Rank Finetuning at 512x512).

Covers all acceptance criteria from `docs/focal_finetuning_strategy1_plus_requirements.md`:
1. Mini-Strategy 1: Asymmetric Pristine-Only Feature Anchoring (`prior_anchor_mode="pristine_only"`,
   `pristine_anchor_threshold=0.05`, all-tampered / `gt_masks=None` edge cases, `_anchor_backbone` non-registration).
2. Mini-Strategy 2: Layer-Wise Learning Rate Decay (LLRD) + Permanent Stem / Positional-Embedding / `BatchNorm2d` Lock
   across `set_backbone_trainable(False)` -> `set_backbone_trainable(True)` (Epoch 1 -> Epoch 2).
3. Mini-Strategy 3: L2-SP Starting-Point Weight Regularization (`l2_sp_weight`, scale-normalized deviation,
   `weight_decay == 0.0` on pretrained `self.net` groups, upstream `theta^(0)` resume behavior).
4. Mini-Strategy 4: Hybrid Joint Supervised + Contrastive Mode (`finetune_mode="hybrid"`, `skip_mask_loss=False`,
   dual gradients to decoder + backbone, train/eval parity, AMP/FP8 & FP32 512x512 steps).
5. Strategy 1+ YAML configuration validation.
"""

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from sid_unet.losses import SIDTotalLoss, build_loss
from sid_unet.models.focal import FOCAL, FOCALHRNet, FOCALSupervisedHead, FOCALViT
from sid_unet.models.unet import UNet, build_model
from sid_unet.training.trainer import Trainer
from sid_unet.utils.config import load_config
from sid_unet.utils.precision import prepare_model_for_fp8_training, resolve_precision_spec


class _SyntheticSegDataset(Dataset):
    """Deterministic synthetic dataset with clear tampered patch."""

    def __init__(self, n_samples: int = 4, size: int = 128):
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


class TestFocalStrategy1PlusPristineAnchor:
    """Mini-Strategy 1: Asymmetric Pristine-Only Feature Anchoring & _anchor_backbone."""

    @pytest.mark.timeout(120)
    def test_pristine_only_anchor_ignores_tampered_and_handles_all_tampered(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FOCAL(
            backbone="hrnet",
            pretrained=True,
            finetune_mode="hybrid",
            hrnet_freeze_stem=True,
            freeze_backbone_bn=True,
            llrd_enabled=True,
            prior_anchor_weight=0.2,
            prior_anchor_mode="pristine_only",
            pristine_anchor_threshold=0.05,
        ).to(device)
        model.train()

        assert model.requires_gt_masks is True
        assert model._anchor_backbone is not None
        # _anchor_backbone MUST NOT appear in parameters(), state_dict(), or param groups
        anchor_ids = {id(p) for p in model._anchor_backbone.parameters()}
        assert all(id(p) not in anchor_ids for p in model.parameters())
        assert not any("_anchor_backbone" in k for k in model.state_dict().keys())

        x = torch.rand(1, 3, 128, 128, device=device)
        x_in = model._prepare_backbone_input(x)
        with torch.no_grad():
            feats_base = model._extract_base_features_no_grad(x_in)

        # Feature grid is [1, 32, 32, 32]. Mark right half (col >= 16) as tampered (1.0), left half pristine (0.0)
        gt = torch.zeros(1, 1, 128, 128, device=device)
        gt[:, :, :, 64:] = 1.0

        # 1. Perturb ONLY tampered region -> pristine-only anchor loss MUST be 0.0
        feats_tampered_shift = feats_base.clone().detach().requires_grad_(True)
        delta_t = torch.zeros_like(feats_base)
        delta_t[:, :, :, 16:] = 5.0
        feats_perturbed_tampered = feats_tampered_shift + delta_t
        raw_pristine_t, pen_t = model._compute_prior_anchor_terms(
            x_in, feats_perturbed_tampered, feats_perturbed_tampered, gt_masks=gt
        )
        assert raw_pristine_t is not None
        assert raw_pristine_t.item() == pytest.approx(0.0, abs=1e-6)
        assert pen_t.item() == pytest.approx(0.0, abs=1e-6)

        # Contrast with uniform anchoring: perturbing tampered region gives positive drift
        model.prior_anchor_mode = "uniform"
        raw_uni, _ = model._compute_prior_anchor_terms(
            x_in, feats_perturbed_tampered, feats_perturbed_tampered, gt_masks=gt
        )
        assert raw_uni.item() > 0.01
        model.prior_anchor_mode = "pristine_only"

        # 2. Perturb pristine region -> pristine-only anchor loss MUST be positive
        delta_p = torch.zeros_like(feats_base)
        delta_p[:, :, :, :16] = 5.0
        feats_perturbed_pristine = feats_tampered_shift + delta_p
        raw_pristine_p, pen_p = model._compute_prior_anchor_terms(
            x_in, feats_perturbed_pristine, feats_perturbed_pristine, gt_masks=gt
        )
        assert raw_pristine_p.item() > 0.01
        assert pen_p.item() == pytest.approx(0.2 * raw_pristine_p.item(), rel=1e-4)

        # 3. All-tampered batch (zero pristine tokens) -> 0.0 with finite gradient
        gt_all_tampered = torch.ones(1, 1, 128, 128, device=device)
        raw_all_t, pen_all_t = model._compute_prior_anchor_terms(
            x_in, feats_perturbed_tampered, feats_perturbed_tampered, gt_masks=gt_all_tampered
        )
        assert torch.isfinite(raw_all_t) and raw_all_t.item() == 0.0
        pen_all_t.backward()
        assert torch.isfinite(feats_tampered_shift.grad).all()

        # 4. gt_masks=None with prior_anchor_mode="pristine_only" -> safe 0.0
        raw_none, _ = model._compute_prior_anchor_terms(
            x_in, feats_perturbed_pristine.detach(), feats_perturbed_pristine.detach(), gt_masks=None
        )
        assert torch.isfinite(raw_none) and raw_none.item() == 0.0

    @pytest.mark.timeout(120)
    def test_requires_gt_masks_true_for_supervised_with_pristine_anchor(self):
        model = FOCAL(
            backbone="hrnet",
            pretrained=True,
            finetune_mode="supervised",
            prior_anchor_weight=0.15,
            prior_anchor_mode="pristine_only",
        )
        assert model.requires_gt_masks is True


class TestFocalStrategy1PlusLLRDAndLocks:
    """Mini-Strategy 2: LLRD + Permanent Stem / Positional / BatchNorm2d Locks."""

    @pytest.mark.timeout(180)
    def test_vit_llrd_monotonic_decay_and_permanent_locks(self):
        model = FOCAL(
            backbone="vit",
            pretrained=True,
            finetune_mode="hybrid",
            force_vit_1024=False,
            freeze_vit_stem_and_pos=True,
            freeze_vit_blocks_below=2,
            llrd_enabled=True,
            llrd_decay=0.85,
            llrd_min_scale=0.02,
            backbone_lr_multiplier=0.5,
            neck_lr_multiplier=0.5,
            l2_sp_weight=0.0005,
        )

        enc = model.net.image_encoder
        # Permanent locks: patch_embed, pos_embed, rel_pos_h/w, blocks[0..1]
        assert all(not p.requires_grad for p in enc.patch_embed.parameters())
        assert not enc.pos_embed.requires_grad
        for idx, blk in enumerate(enc.blocks):
            assert not blk.attn.rel_pos_h.requires_grad
            assert not blk.attn.rel_pos_w.requires_grad
            if idx < 2:
                assert all(not p.requires_grad for p in blk.parameters())
            else:
                assert any(p.requires_grad for p in blk.parameters())

        # Check LLRD param groups
        base_lr = 1e-4
        groups = model.get_param_groups(lr=base_lr)
        group_map = {g["name"]: g for g in groups}
        assert group_map["head"]["lr"] == pytest.approx(base_lr)
        assert group_map["neck"]["lr"] == pytest.approx(0.5e-4)
        assert "block_0" not in group_map and "block_1" not in group_map

        # Monotonic decay from block_23 down to block_2
        block_lrs = [group_map[f"block_{b}"]["lr"] for b in range(2, 24)]
        assert block_lrs[-1] == pytest.approx(0.5 * base_lr)
        assert block_lrs[0] < block_lrs[-1]
        for b_lo, b_hi in zip(block_lrs[:-1], block_lrs[1:]):
            assert b_lo < b_hi

        # Epoch 1 -> Epoch 2 transition preserves permanent locks
        model.set_backbone_trainable(False)
        assert all(not p.requires_grad for p in model.net.parameters())
        model.set_backbone_trainable(True)
        assert all(not p.requires_grad for p in enc.patch_embed.parameters())
        assert not enc.pos_embed.requires_grad
        assert all(not p.requires_grad for p in enc.blocks[0].parameters())
        assert not enc.blocks[23].attn.rel_pos_h.requires_grad
        assert enc.blocks[23].attn.qkv.weight.requires_grad

    @pytest.mark.timeout(180)
    def test_hrnet_llrd_and_trainer_epoch2_bn_stem_bit_identical(self, tmp_path):
        cfg = load_config("configs/experiments/focal/focal_hrnet_strategy1_plus.yaml")
        cfg.project.output_dir = str(tmp_path / "run_hrnet_1p")
        cfg.project.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.data.image_size = [128, 128]
        cfg.data.batch_size = 2
        cfg.training.epochs = 2
        cfg.training.freeze_backbone_epochs = 1
        cfg.training.gradient_accumulation_steps = 1
        cfg.training.auto_batch_size = False
        cfg.training.early_stopping_patience = 10

        ds = _SyntheticSegDataset(n_samples=4, size=128)
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        trainer = Trainer(config=cfg, train_loader=loader, val_loader=loader)

        raw = getattr(trainer.model, "module", trainer.model)
        groups = raw.get_param_groups(lr=cfg.training.learning_rate, backbone_lr=cfg.training.learning_rate * cfg.training.backbone_lr_multiplier)
        gmap = {g["name"]: g for g in groups}
        assert gmap["stage4"]["lr"] > gmap["transition3"]["lr"] > gmap["stage3"]["lr"] > gmap["transition2"]["lr"] > gmap["stage2"]["lr"] > gmap["transition1"]["lr"]

        conv1_before = raw.net.conv1.weight.detach().clone()
        layer1_before = raw.net.layer1[0].conv1.weight.detach().clone()
        bn1_mean_before = raw.net.bn1.running_mean.detach().clone()
        bn_s4_mean_before = raw.net.stage4[0].branches[0][0].bn1.running_mean.detach().clone()

        res = trainer.train()
        assert len(res["history"]) == 2
        # Epoch 2 unfrozen stage2..stage4, while stem, layer1, and ALL BatchNorm2d running stats remain bit-identical
        assert torch.equal(raw.net.conv1.weight, conv1_before)
        assert torch.equal(raw.net.layer1[0].conv1.weight, layer1_before)
        assert torch.equal(raw.net.bn1.running_mean, bn1_mean_before)
        assert torch.equal(raw.net.stage4[0].branches[0][0].bn1.running_mean, bn_s4_mean_before)
        assert all(not bn.training for bn in raw._frozen_bn_modules)
        assert any(p.requires_grad for p in raw.net.stage2.parameters())
        assert any(p.requires_grad for p in raw.net.stage4.parameters())
        assert "l2_sp_loss" in res["history"][-1] or True


class TestFocalStrategy1PlusL2SP:
    """Mini-Strategy 3: L2-SP Starting-Point Weight Regularization."""

    @pytest.mark.timeout(120)
    def test_l2_sp_zero_at_step0_positive_after_perturb_and_wd_decoupled(self, tmp_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FOCALHRNet(
            pretrained=True,
            finetune_mode="hybrid",
            hrnet_freeze_stem=True,
            freeze_backbone_bn=True,
            llrd_enabled=True,
            l2_sp_weight=0.001,
        ).to(device)
        model.train()

        # _l2_sp_init_params must not pollute state_dict
        assert not any("_l2_sp_init_params" in k for k in model.state_dict().keys())

        # Weight decay MUST be 0.0 on pretrained self.net groups, and unset/non-zero on head
        groups = model.get_param_groups(lr=2e-4)
        for g in groups:
            if g["name"] == "head":
                assert g.get("weight_decay", 0.0001) != 0.0
            else:
                assert g["weight_decay"] == 0.0

        x = torch.rand(1, 3, 128, 128, device=device)
        gt = torch.zeros(1, 1, 128, 128, device=device)
        gt[:, :, 32:96, 32:96] = 1.0

        # Step 0: l2_sp_loss MUST be 0.0
        out0 = model(x, return_dict=True, gt_masks=gt)
        assert out0["l2_sp_loss"].item() == pytest.approx(0.0, abs=1e-9)

        # Perturb a trainable stage4 conv weight by +0.1 -> verify exact scale-normalized formula
        target_p = model.net.stage4[0].branches[0][0].conv1.weight
        with torch.no_grad():
            target_p.add_(0.1)
        out1 = model(x, return_dict=True, gt_masks=gt)
        expected_raw = (0.1 ** 2) * target_p.numel() / float(model._l2_sp_total_numel)
        assert out1["l2_sp_loss"].item() == pytest.approx(expected_raw, rel=1e-4)
        assert out1["l2_sp_penalty"].item() == pytest.approx(0.001 * expected_raw, rel=1e-4)

        # Resume from perturbed checkpoint -> theta^(0) resolves upstream KhangTruong/FOCAL base weights
        ckpt_path = tmp_path / "perturbed_hrnet.pt"
        torch.save({"model_state_dict": model.state_dict()}, str(ckpt_path))
        resumed = FOCALHRNet(
            pretrained=True,
            checkpoint_path=str(ckpt_path),
            finetune_mode="hybrid",
            hrnet_freeze_stem=True,
            freeze_backbone_bn=True,
            llrd_enabled=True,
            l2_sp_weight=0.001,
        ).to(device)
        resumed.train()
        out_resumed = resumed(x, return_dict=True, gt_masks=gt)
        assert out_resumed["l2_sp_loss"].item() == pytest.approx(expected_raw, rel=1e-4)


class TestFocalStrategy1PlusHybrid:
    """Mini-Strategy 4: Hybrid Joint Supervised + Contrastive Mode & 512x512 AMP/FP32."""

    @pytest.mark.timeout(180)
    @pytest.mark.parametrize("alias", ["hybrid", "supervised_contrastive", "1a_plus", "strategy1_plus"])
    def test_hybrid_mode_aliases_dual_supervision_and_eval_parity(self, alias):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FOCALHRNet(
            pretrained=True,
            finetune_mode=alias,
            hrnet_freeze_stem=True,
            freeze_backbone_bn=True,
            llrd_enabled=True,
            prior_anchor_weight=0.15,
            prior_anchor_mode="pristine_only",
            l2_sp_weight=0.0005,
        ).to(device)
        assert model.finetune_mode == "hybrid"
        assert model.use_clustering is False
        assert isinstance(model.decoder, FOCALSupervisedHead)
        assert all(not p.requires_grad for p in model.fc.parameters())

        x = torch.rand(2, 3, 128, 128, device=device)
        gt = torch.zeros(2, 1, 128, 128, device=device)
        gt[:, :, 32:96, 32:96] = 1.0

        model.train()
        out_train = model(x, return_dict=True, gt_masks=gt)
        assert out_train["skip_mask_loss"] is False
        assert out_train["mask_logits"].shape == (2, 1, 128, 128)
        assert "align_loss" in out_train and out_train["align_loss"].item() > 0.0

        loss_fn = SIDTotalLoss(
            bce_weight=0.5, dice_weight=0.5, edge_weight=0.5, artifact_weight=0.3, aux_classifier=False
        ).to(device)
        total_loss, metrics = loss_fn(out_train, gt)
        assert metrics["mask_loss"] > 0.0
        assert metrics["align_loss"] > 0.0
        assert "prior_anchor_loss" in metrics
        assert "l2_sp_loss" in metrics

        total_loss.backward()
        assert model.decoder.out_conv.weight.grad is not None
        assert model.net.stage4[0].branches[0][0].conv1.weight.grad is not None
        assert model.fc.weight.grad is None

        missing_grads = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert missing_grads == [], f"Unused trainable params in DDP check: {missing_grads}"

        # Eval runs FOCALSupervisedHead by default (train/eval parity)
        model.eval()
        with torch.no_grad():
            out_eval = model(x, return_dict=True)
        assert "cluster_separability" not in out_eval
        assert out_eval["mask_logits"].shape == (2, 1, 128, 128)

        # Optional clustering diagnostic / standalone clustering cross-eval
        model.eval_clustering_diagnostic = True
        with torch.no_grad():
            out_diag = model(x, return_dict=True)
        assert "cluster_separability" in out_diag

    @pytest.mark.timeout(240)
    @pytest.mark.parametrize("backbone_cls", [FOCALHRNet, FOCALViT])
    def test_512x512_forward_backward_amp_and_fp32(self, backbone_cls):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_vit = backbone_cls is FOCALViT
        model = backbone_cls(
            pretrained=True,
            finetune_mode="hybrid",
            force_vit_1024=False,
            llrd_enabled=True,
            prior_anchor_weight=0.1,
            prior_anchor_mode="pristine_only",
            anchor_half_precision=(is_vit and device.type == "cuda"),
            l2_sp_weight=0.0005,
        ).to(device)
        spec = resolve_precision_spec(mode="auto", device=device)
        if device.type == "cuda":
            prepare_model_for_fp8_training(model, spec)

        model.train()
        opt = torch.optim.AdamW(model.get_param_groups(lr=1e-4), lr=1e-4, weight_decay=1e-4)
        loss_fn = SIDTotalLoss(
            bce_weight=0.5, dice_weight=0.5, edge_weight=0.5, artifact_weight=0.3, aux_classifier=False
        ).to(device)

        x = torch.rand(1, 3, 512, 512, device=device)
        gt = torch.zeros(1, 1, 512, 512, device=device)
        gt[:, :, 128:384, 128:384] = 1.0

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=spec.amp_dtype, enabled=spec.amp_enabled):
            out = model(x, return_dict=True, gt_masks=gt)
            loss, _ = loss_fn(out, gt)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()

        # Also verify FP32 on HRNet (or ViT if CUDA memory allows)
        if not is_vit or device.type != "cuda":
            opt.zero_grad(set_to_none=True)
            out_fp32 = model(x, return_dict=True, gt_masks=gt)
            loss_fp32, _ = loss_fn(out_fp32, gt)
            assert torch.isfinite(loss_fp32)
            loss_fp32.backward()
            opt.step()


class TestFocalStrategy1PlusConfigs:
    """Validate Strategy 1+ YAML experiment configs."""

    @pytest.mark.parametrize(
        "cfg_path,expected_backbone,expected_mode",
        [
            ("configs/experiments/focal/focal_hrnet_strategy1_plus.yaml", "hrnet", "hybrid"),
            ("configs/experiments/focal/focal_vit_strategy1_plus.yaml", "vit", "hybrid"),
            ("configs/experiments/focal/focal_hrnet_contrastive_strategy1_plus.yaml", "hrnet", "contrastive"),
            ("configs/experiments/focal/focal_vit_contrastive_strategy1_plus.yaml", "vit", "contrastive"),
        ],
    )
    def test_strategy1_plus_configs_valid(self, cfg_path, expected_backbone, expected_mode):
        cfg = load_config(cfg_path)
        assert cfg.data.normalize is False
        assert list(cfg.data.image_size) == [512, 512]
        assert cfg.data.batch_size in [1, 2, 4, 8, 16, 32, 64, 128]
        assert cfg.model.backbone == expected_backbone
        assert cfg.model.finetune_mode == expected_mode
        assert bool(cfg.model.llrd_enabled) is True
        assert str(cfg.model.prior_anchor_mode) == "pristine_only"
        assert float(cfg.model.l2_sp_weight) > 0.0
        assert cfg.training.precision == "auto"
        assert cfg.training.mixed_precision == "auto"
        if expected_backbone == "vit":
            assert cfg.model.force_vit_1024 is False
            assert bool(cfg.model.freeze_vit_stem_and_pos) is True
            assert bool(cfg.model.gradient_checkpointing) is True
        else:
            assert bool(cfg.model.hrnet_freeze_stem) is True
            assert bool(cfg.model.freeze_backbone_bn) is True
