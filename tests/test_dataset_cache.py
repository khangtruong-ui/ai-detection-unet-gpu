"""
Unit and integration tests for the dataset caching pipeline,
high-dimensional forensics latents extraction, cached DataLoader loading,
and cached training with Diffusion-Diff-Minimized.
"""

import io
import os
import shutil
import tempfile
import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pyarrow as pa
import pyarrow.parquet as pq

from sid_unet.models.diffusion_diff_minimized import DiffusionDiffMinimizedModel
from sid_unet.models.diffusion_diff import DiffusionDiffModel
from sid_unet.models.diffusion_diff_v2 import DiffusionDiffV2Model
from sid_unet.cache.extractor import (
    DiffusionDiffMinimizedExtractor,
    get_extractor_for_model,
)
from sid_unet.cache.dataset import (
    CachedTensorDataset,
    CachedStreamingDataset,
    CachedIndexedDataset,
    SpatialJointTransform,
    decode_cached_tensor,
    decode_cached_image,
    decode_mask_tensor,
)
from sid_unet.cache.manager import DatasetCacheManager
from sid_unet.cache.cli import parse_args
from sid_unet.dataset.loader import (
    create_dataloaders,
    create_cached_dataloaders,
    resolve_cached_parquet_files,
    _generate_mock_cached_parquet_shard,
)
from sid_unet.utils.config import ConfigDict


@pytest.fixture
def dummy_minimized_model():
    return DiffusionDiffMinimizedModel(use_dummy=True)


@pytest.fixture
def temp_cache_dir():
    d = tempfile.mkdtemp(prefix="test_sid_cache_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_extractor_diffusion_diff_minimized(dummy_minimized_model):
    """Test extractor produces correct 84-channel float16 tensor from RGB input."""
    extractor = get_extractor_for_model(dummy_minimized_model, fp16=True)
    assert extractor.model_name == "diffusion_diff_minimized"
    assert extractor.total_channels == 84

    x = torch.randn(2, 3, 256, 256)
    z = extractor.extract_batch(x)
    assert z.shape[:2] == (2, 84)
    assert z.dtype == torch.float16


def test_forward_cached_and_routing(dummy_minimized_model):
    """Test fast forward_cached path and automatic routing in forward()."""
    dummy_minimized_model.eval()
    x = torch.randn(2, 3, 256, 256)
    z = dummy_minimized_model.extract_cache_tensors(x)
    assert z.shape[:2] == (2, 84)

    # 1. Direct forward_cached call with target spatial dimensions
    mask_logits_cached, class_logits_cached = dummy_minimized_model.forward_cached(z, target_h=256, target_w=256)
    assert mask_logits_cached.shape == (2, 1, 256, 256)
    assert class_logits_cached.shape == (2, 3)

    # 2. Forward auto-routing when passing 84-channel tensor of latent size 32x32
    z_standard = torch.randn(2, 84, 32, 32)
    mask_logits_routed, class_logits_routed = dummy_minimized_model(z_standard)
    assert mask_logits_routed.shape == (2, 1, 256, 256)
    assert class_logits_routed.shape == (2, 3)


def test_bypass_diffuser_for_cached_training(dummy_minimized_model):
    """Test that Diffuser UNet can be completely unloaded while cached forward continues to work."""
    z = torch.randn(2, 84, 32, 32)
    assert dummy_minimized_model.diffuser is not None

    dummy_minimized_model.bypass_diffuser_for_cached_training()
    assert dummy_minimized_model.diffuser is None

    # Cached forward works without diffuser
    mask_logits, class_logits = dummy_minimized_model(z)
    assert mask_logits.shape == (2, 1, 256, 256)
    assert class_logits.shape == (2, 3)


def test_diffusion_diff_and_v2_cache_methods():
    """Verify caching methods exist and work across the entire diffusion family."""
    m_v1 = DiffusionDiffModel(use_dummy=True)
    m_v2 = DiffusionDiffV2Model(use_dummy=True)

    x = torch.randn(1, 3, 256, 256)
    z1 = m_v1.extract_cache_tensors(x)
    z2 = m_v2.extract_cache_tensors(x)

    assert z1.shape[1] == m_v1.total_z_channels
    assert z2.shape[1] == m_v2.total_z_channels

    out1 = m_v1.forward_cached(z1, target_h=256, target_w=256)
    out2 = m_v2.forward_cached(z2, target_h=256, target_w=256)
    assert out1[0].shape == (1, 1, 256, 256)
    assert out2[0].shape == (1, 1, 256, 256)


def test_spatial_joint_transform():
    """Test synchronous spatial flips and rotations on (Z, mask)."""
    transform = SpatialJointTransform(horizontal_flip=1.0, vertical_flip=1.0, random_rotate90=1.0)
    z = torch.arange(84 * 32 * 32, dtype=torch.float32).reshape(84, 32, 32)
    mask = torch.ones(1, 256, 256, dtype=torch.float32)

    z_aug, mask_aug = transform(z, mask)
    assert z_aug.shape == (84, 32, 32)
    assert mask_aug.shape == (1, 256, 256)


def test_cached_parquet_dataset_load(temp_cache_dir):
    """Test creating and reading cached Parquet shards with CachedTensorDataset."""
    shards = _generate_mock_cached_parquet_shard(
        split="train",
        image_size=(256, 256),
        channels=84,
        num_samples=10,
    )

    dataset = CachedTensorDataset(
        parquet_files=shards,
        target_image_size=(256, 256),
        expected_channels=84,
    )
    assert len(dataset) == 10

    sample = dataset[0]
    assert sample["image"].shape == (84, 32, 32)
    assert sample["mask"].shape == (1, 256, 256)
    assert sample["label"].dim() == 0
    assert sample["is_cached"] is True


def test_create_cached_dataloaders(temp_cache_dir):
    """Test loader integration via create_dataloaders with cached_hf_repo."""
    # Create train and val shards in temp dir
    train_shards = _generate_mock_cached_parquet_shard(split="train", num_samples=16)
    val_shards = _generate_mock_cached_parquet_shard(split="validation", num_samples=8)

    cfg = ConfigDict({
        "project": {"seed": 42},
        "data": {
            "cached_hf_repo": os.path.dirname(train_shards[0]),
            "image_size": [256, 256],
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "train_samples_per_epoch": -1,
            "val_samples_per_epoch": -1,
            "augmentations": {"horizontal_flip": 0.5},
        },
        "model": {"total_z_channels": 84},
    })

    train_loader, val_loader = create_dataloaders(cfg, include_test=False)
    batch = next(iter(train_loader))
    assert batch["image"].shape == (4, 84, 32, 32)
    assert batch["mask"].shape == (4, 1, 256, 256)
    assert batch["label"].shape == (4,)
    assert batch["is_cached"].all()


def test_training_step_with_cache(dummy_minimized_model, temp_cache_dir):
    """Test full forward and backward pass on cached data with trainable decoder optimization."""
    from sid_unet.losses.auxiliary import SIDTotalLoss
    from sid_unet.losses.combined import CombinedMaskLoss

    dummy_minimized_model.train()
    dummy_minimized_model.bypass_diffuser_for_cached_training()

    loss_fn = SIDTotalLoss(mask_loss_type="combined", aux_classifier=True)
    optimizer = torch.optim.AdamW(dummy_minimized_model.parameters(), lr=1e-3)

    z_batch = torch.randn(2, 84, 32, 32, requires_grad=False)
    mask_batch = torch.randint(0, 2, (2, 1, 256, 256)).float()
    label_batch = torch.tensor([1, 2], dtype=torch.long)

    optimizer.zero_grad()
    outputs = dummy_minimized_model(z_batch)
    loss, loss_dict = loss_fn(outputs, mask_batch, label_batch)
    assert torch.isfinite(loss)

    loss.backward()
    optimizer.step()
    assert dummy_minimized_model.decoder.stages[0].res_blocks[0].conv1.weight.grad is not None


def test_cache_cli_parser():
    """Test CLI argument parsing for sid-cache."""
    args = parse_args([
        "--config", "configs/experiments/diffusion_diff_minimized/default.yaml",
        "--dataset", "KhangTruong/COCO-inpainted",
        "--hf-repo", "KhangTruong/COCO-inpainted-cache",
        "--batch-size", "16",
        "--max-samples", "100",
        "--push-to-hub",
    ])
    assert args.config == "configs/experiments/diffusion_diff_minimized/default.yaml"
    assert args.dataset_name == "KhangTruong/COCO-inpainted"
    assert args.hf_repo == "KhangTruong/COCO-inpainted-cache"
    assert args.batch_size == 16
    assert args.max_samples == 100
    assert args.push_to_hub is True


def test_sid_train_with_cached_hf_repo_integration(temp_cache_dir):
    """Test full sid-train execution using --cached-hf-repo flag and dummy diffusion model."""
    from sid_unet.train import train_single_run

    # Generate mock cached shards for train and validation
    _generate_mock_cached_parquet_shard(split="train", num_samples=8)
    _generate_mock_cached_parquet_shard(split="validation", num_samples=4)

    mock_cache_dir = os.path.dirname(_generate_mock_cached_parquet_shard(split="train", num_samples=2)[0])
    output_run_dir = os.path.join(temp_cache_dir, "cached_run")

    overrides = [
        "model.use_dummy=true",
        "training.epochs=1",
        "data.batch_size=2",
        "data.train_samples_per_epoch=4",
        "data.val_samples_per_epoch=2",
        "training.amp=false",
        "training.use_8bit_optimizer=false",
        "bootstrapping.enabled=false",
        "hard_mining.enabled=false",
        f"project.output_dir={output_run_dir}",
    ]

    results = train_single_run(
        config_path="configs/experiments/diffusion_diff_minimized/default.yaml",
        overrides=overrides,
        cached_hf_repo=mock_cache_dir,
        auto_resume=False,
    )

    assert results is not None
    assert "best_score" in results
    assert os.path.exists(output_run_dir)


def test_async_hub_uploader_and_progressive_caching(dummy_minimized_model, temp_cache_dir, monkeypatch):
    """Test AsyncHubUploader non-blocking background queue and progressive upload orchestration."""
    from sid_unet.cache.manager import AsyncHubUploader

    uploaded_calls = []

    class MockHfApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, repo_id, repo_type="dataset", exist_ok=True):
            pass

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type="dataset", commit_message=None):
            uploaded_calls.append((path_in_repo, commit_message))

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", MockHfApi)

    uploader = AsyncHubUploader(repo_id="KhangTruong/test-cache", max_pending=2)
    test_file = os.path.join(temp_cache_dir, "test.parquet")
    with open(test_file, "wb") as f:
        f.write(b"mock_parquet_data")

    uploader.submit_upload(
        local_path=test_file,
        rel_repo_path="train/train-00000.parquet",
        commit_message="Add shard 00000",
    )
    uploader.wait_all()
    uploader.close()

    assert len(uploaded_calls) == 1
    assert uploaded_calls[0][0] == "train/train-00000.parquet"
    assert "Add shard 00000" in uploaded_calls[0][1]


def test_cached_streaming_dataset_and_prefetcher(temp_cache_dir):
    """Test CachedStreamingDataset length reporting, shuffle buffer, and iteration."""
    shard1 = _generate_mock_cached_parquet_shard(split="train", num_samples=8)[0]
    shard2 = os.path.join(temp_cache_dir, "mock_train_00001.parquet")
    import shutil
    shutil.copyfile(shard1, shard2)

    ds = CachedStreamingDataset(
        parquet_files=[shard1, shard2],
        target_image_size=(256, 256),
        expected_channels=84,
        shuffle=True,
        shuffle_buffer_size=4,
        max_samples=12,
    )
    assert len(ds) == 12

    items = list(ds)
    assert len(items) == 12
    assert items[0]["image"].shape == (84, 32, 32)
    assert items[0]["mask"].shape == (1, 256, 256)
    assert items[0]["is_cached"] is True


def test_create_cached_dataloaders_streaming_mode(temp_cache_dir):
    """Test create_cached_dataloaders with streaming=True using BackgroundPrefetcher."""
    train_shards = _generate_mock_cached_parquet_shard(split="train", num_samples=16)
    val_shards = _generate_mock_cached_parquet_shard(split="validation", num_samples=8)

    cfg = ConfigDict({
        "project": {"seed": 42},
        "data": {
            "cached_hf_repo": os.path.dirname(train_shards[0]),
            "streaming": True,
            "image_size": [256, 256],
            "batch_size": 4,
            "num_workers": -1,
            "pin_memory": False,
            "train_samples_per_epoch": 8,
            "val_samples_per_epoch": 4,
            "prefetch_batches": 8,
            "shuffle_buffer_size": 4,
            "augmentations": {"horizontal_flip": 0.5},
        },
        "model": {"total_z_channels": 84},
    })

    train_loader, val_loader = create_dataloaders(cfg, include_test=False)
    assert len(train_loader) == 2
    batch = next(iter(train_loader))
    assert batch["image"].shape == (4, 84, 32, 32)
    assert batch["mask"].shape == (4, 1, 256, 256)
    assert batch["label"].shape == (4,)
    assert batch["is_cached"].all()


def test_cached_indexed_dataset_and_sharding():
    """Test CachedIndexedDataset random access, length reporting, and DDP sharding."""
    from sid_unet.cache.dataset import CachedIndexedDataset, SpatialJointTransform

    class DummyPdlDataset:
        def __init__(self, n=16, indices=None):
            self.n = n
            self.indices = indices if indices is not None else list(range(n))

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            z_t = np.random.randn(84, 32, 32).astype(np.float16).tobytes()
            mask_img = Image.new("L", (256, 256), color=255)
            buf = io.BytesIO()
            mask_img.save(buf, format="PNG")
            return {
                "img_id": f"dummy_{real_idx}",
                "label": real_idx % 3,
                "z_high_dim": z_t,
                "mask": buf.getvalue(),
                "channels": 84,
                "latent_h": 32,
                "latent_w": 32,
            }

        def shard(self, num_shards, index, contiguous=False):
            sharded_indices = [idx for i, idx in enumerate(self.indices) if i % num_shards == index]
            return DummyPdlDataset(n=self.n, indices=sharded_indices)

    raw_pdl = DummyPdlDataset(n=16)
    tf = SpatialJointTransform()
    cached_ds = CachedIndexedDataset(pdl_dataset=raw_pdl, transform=tf, expected_channels=84)

    assert len(cached_ds) == 16
    sample = cached_ds[0]
    assert sample["image"].shape == (84, 32, 32)
    assert sample["mask"].shape == (1, 256, 256)
    assert sample["is_cached"] is True

    # Shard across 4 workers
    sharded = cached_ds.shard(num_shards=4, index=1)
    assert len(sharded) == 4
    sample_shard = sharded[0]
    assert sample_shard["img_id"] == "dummy_1"


def test_count_existing_samples_avoids_overestimating(dummy_minimized_model, temp_cache_dir):
    """Test that _count_existing_samples counts actual rows and does not assume len*samples_per_shard."""
    split_dir = os.path.join(temp_cache_dir, "validation")
    os.makedirs(split_dir, exist_ok=True)

    # Write 3 small shards of 64, 36, and 50 rows = total 150 rows
    row_counts = [64, 36, 50]
    shard_names = []
    for i, rc in enumerate(row_counts):
        s_name = f"validation-{i:05d}.parquet"
        s_path = os.path.join(split_dir, s_name)
        shard_names.append(s_name)

        t = pa.Table.from_pydict({
            "img_id": [f"val_{i}_{j}" for j in range(rc)],
            "label": [j % 3 for j in range(rc)],
            "z_high_dim": [b"\x00" * 10] * rc,
            "mask": [b"\x00" * 10] * rc,
            "channels": [84] * rc,
            "latent_h": [32] * rc,
            "latent_w": [32] * rc,
        })
        pq.write_table(t, s_path, compression="zstd")

    manager = DatasetCacheManager(
        model=dummy_minimized_model,
        output_dir=temp_cache_dir,
        samples_per_shard=2000,
        push_to_hub=False,
    )

    actual_count = manager._count_existing_samples("validation", shard_names)
    assert actual_count == 150, f"Expected 150 actual rows, got {actual_count}"
    assert actual_count != 3 * 2000, "Should not multiply shard count by samples_per_shard!"


def test_checkpoint_manager_non_blocking_hub_push(temp_cache_dir):
    """Test CheckpointManager background ThreadPoolExecutor handles pushes non-blockingly."""
    from sid_unet.training.callbacks import CheckpointManager

    push_executed = False

    def mock_push(*args, **kwargs):
        nonlocal push_executed
        push_executed = True
        return {"status": "success"}

    ckpt_manager = CheckpointManager(
        checkpoint_dir=os.path.join(temp_cache_dir, "ckpts"),
        push_to_hub=True,
        hf_repo="KhangTruong/test-model",
        verify_repo=False,
    )

    import sid_unet.checkpoint_sync
    orig_push = sid_unet.checkpoint_sync.push_checkpoint
    sid_unet.checkpoint_sync.push_checkpoint = mock_push
    try:
        fut = ckpt_manager.push_to_hf(epoch=1, step=10, saved_paths={"latest": "dummy.pt"})
        assert fut is not None
        ckpt_manager.wait_pending_pushes(timeout=5)
        assert push_executed is True
    finally:
        sid_unet.checkpoint_sync.push_checkpoint = orig_push
        ckpt_manager.close()


def test_decode_cached_tensor_safe_reshape_gap_sam_mismatch():
    """
    Test reproducing the exact bug in .logs.txt:
    'ValueError: cannot reshape array of size 256 into shape (256,32,32)'
    When dataset shard metadata incorrectly stored latent_h=32, latent_w=32 for
    a 256-element 1D pooled descriptor, decode_cached_tensor must safely return (256,).
    """
    raw_vec = np.random.randn(256).astype(np.float16)
    raw_bytes = raw_vec.tobytes()

    # Metadata incorrectly specified 32x32 spatial dimensions
    decoded = decode_cached_tensor(
        raw_bytes,
        channels=256,
        latent_h=32,
        latent_w=32,
        target_image_size=(256, 256),
    )
    assert decoded.shape == (256,)
    assert decoded.dtype == torch.float32
    assert torch.allclose(decoded, torch.from_numpy(raw_vec.astype(np.float32)), atol=1e-3)


def test_decode_cached_tensor_safe_shapes():
    """Verify decode_cached_tensor handles standard 3D latents, 1D descriptors, and edge cases."""
    # 1. Standard 84-channel 32x32 latent
    std_arr = np.random.randn(84, 32, 32).astype(np.float16)
    decoded_std = decode_cached_tensor(std_arr.tobytes(), channels=84, latent_h=32, latent_w=32)
    assert decoded_std.shape == (84, 32, 32)

    # 2. 1D descriptor with correct 1x1 metadata
    desc_1d = np.random.randn(256).astype(np.float16)
    decoded_1d = decode_cached_tensor(desc_1d.tobytes(), channels=256, latent_h=1, latent_w=1)
    assert decoded_1d.shape == (256,)

    # 3. Empty buffer fallback
    decoded_empty = decode_cached_tensor(b"", channels=84, latent_h=32, latent_w=32)
    assert decoded_empty.shape == (84, 32, 32)
    assert torch.all(decoded_empty == 0)


def test_spatial_joint_transform_1d_descriptor():
    """Verify SpatialJointTransform handles 1D descriptors without IndexError or spatial distortion."""
    tf = SpatialJointTransform(horizontal_flip=1.0, vertical_flip=1.0, random_rotate90=1.0)
    z_1d = torch.randn(256)
    mask = torch.ones(1, 64, 64)
    mask[0, :32, :32] = 0.0

    z_out, mask_out = tf(z_1d, mask)
    # 1D descriptor must be unchanged
    assert torch.equal(z_out, z_1d)
    assert z_out.shape == (256,)
    # Mask must have been transformed
    assert mask_out.shape == (1, 64, 64)


def test_cached_datasets_gap_sam_descriptor_with_and_without_image(temp_cache_dir):
    """
    Test CachedTensorDataset and CachedIndexedDataset with GAP-SAM 1D descriptor shards:
    1. Modern shard with 'image' column containing JPEG bytes.
    2. Legacy shard (like KhangTruong/gap-sam-coco-cache) without 'image' column.
    """
    split_dir = os.path.join(temp_cache_dir, "test_split")
    os.makedirs(split_dir, exist_ok=True)

    # Prepare dummy JPEG image
    pil_img = Image.new("RGB", (256, 256), color=(128, 64, 32))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    jpeg_bytes = buf.getvalue()

    mask_bytes = np.zeros((1, 256, 256), dtype=np.uint8).tobytes()
    desc_bytes = np.random.randn(256).astype(np.float16).tobytes()

    # Case 1: Shard WITH image column
    shard_with_img = os.path.join(split_dir, "with_img.parquet")
    t1 = pa.Table.from_pydict({
        "img_id": ["sample_0", "sample_1"],
        "label": [0, 1],
        "z_high_dim": [desc_bytes, desc_bytes],
        "mask": [mask_bytes, mask_bytes],
        "channels": [256, 256],
        "latent_h": [1, 1],
        "latent_w": [1, 1],
        "image": [jpeg_bytes, jpeg_bytes],
    })
    pq.write_table(t1, shard_with_img, compression="zstd")

    # Case 2: Shard WITHOUT image column (legacy metadata with 32x32 mismatch!)
    shard_legacy = os.path.join(split_dir, "legacy_no_img.parquet")
    t2 = pa.Table.from_pydict({
        "img_id": ["legacy_0", "legacy_1"],
        "label": [0, 1],
        "z_high_dim": [desc_bytes, desc_bytes],
        "mask": [mask_bytes, mask_bytes],
        "channels": [256, 256],
        "latent_h": [32, 32],  # The buggy 32x32 metadata
        "latent_w": [32, 32],
    })
    pq.write_table(t2, shard_legacy, compression="zstd")

    # Test CachedTensorDataset with both shards
    ds_with_img = CachedTensorDataset([shard_with_img], expected_channels=256)
    sample_img = ds_with_img[0]
    assert sample_img["image"].shape == (3, 256, 256)
    assert sample_img["f_r"].shape == (256,)
    assert sample_img["gap_r"].shape == (256,)
    assert sample_img["mask"].shape == (1, 256, 256)

    ds_legacy = CachedTensorDataset([shard_legacy], expected_channels=256)
    sample_leg = ds_legacy[0]
    # Fallback image should be generated without error
    assert sample_leg["image"].shape == (3, 256, 256)
    assert sample_leg["f_r"].shape == (256,)
    assert sample_leg["mask"].shape == (1, 256, 256)

    # Test with custom image_provider
    provider_called = False

    def custom_provider(img_id, target_size):
        nonlocal provider_called
        provider_called = True
        return torch.ones(3, *target_size)

    ds_legacy_provider = CachedTensorDataset([shard_legacy], expected_channels=256, image_provider=custom_provider)
    sample_leg_p = ds_legacy_provider[0]
    assert provider_called is True
    assert sample_leg_p["image"].shape == (3, 256, 256)
    assert torch.all(sample_leg_p["image"] == 1.0)


def test_cached_indexed_dataset_gap_sam_1d():
    """Verify CachedIndexedDataset handles 1D descriptors and buggy 32x32 metadata."""
    desc_bytes = np.random.randn(256).astype(np.float16).tobytes()
    mask_bytes = np.zeros((1, 256, 256), dtype=np.uint8).tobytes()

    class DummyPdlGapDataset:
        def __init__(self, n=4):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            return {
                "img_id": f"dummy_{idx}",
                "label": 0,
                "z_high_dim": desc_bytes,
                "mask": mask_bytes,
                "channels": 256,
                "latent_h": 32,  # Buggy metadata
                "latent_w": 32,
            }

    raw_pdl = DummyPdlGapDataset(n=4)
    cached_ds = CachedIndexedDataset(pdl_dataset=raw_pdl, expected_channels=256)
    sample = cached_ds[0]
    assert sample["image"].shape == (3, 256, 256)
    assert sample["f_r"].shape == (256,)
    assert sample["gap_r"].shape == (256,)


def test_gap_sam_memory_probing_with_bypassed_vae():
    """Verify find_optimal_batch_size probes GAP-SAM models with vae_bypassed=True without crash."""
    from sid_unet.utils.memory import find_optimal_batch_size

    class DummyGAPSAM(nn.Module):
        def __init__(self):
            super().__init__()
            self.vae_bypassed = True
            self.linear = nn.Linear(256, 1)

        def forward(self, image, f_r=None, **kwargs):
            bs = image.shape[0]
            if f_r is None:
                raise ValueError("Expected f_r when VAE is bypassed!")
            mask_logits = self.linear(f_r).view(bs, 1, 1, 1).expand(bs, 1, 256, 256)
            class_logits = torch.zeros(bs, 3, device=image.device)
            return mask_logits, class_logits

    model = DummyGAPSAM()
    device = torch.device("cpu")
    opt_bs = find_optimal_batch_size(
        model=model,
        device=device,
        image_size=(256, 256),
        in_channels=3,
        min_batch_size=2,
        max_batch_size=4,
        force_probe=True,
    )
    assert opt_bs in (2, 4)


def test_dataset_cache_manager_gap_sam_shards_with_images(temp_cache_dir):
    """Verify DatasetCacheManager produces Parquet shards with 1x1 latent dimensions and 'image' column for GAP-SAM."""
    from torch.utils.data import Dataset

    class DummyImageMaskDataset(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            if idx >= 4:
                raise IndexError(f"Index {idx} out of bounds")
            return {
                "image": torch.randn(3, 256, 256),
                "mask": torch.zeros(1, 256, 256),
                "label": 1,
                "img_id": f"img_{idx}",
            }

    class DummyGAPSAMExtractorModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model_name = "gap_sam"
            self.total_channels = 256

        def extract_cache_tensors(self, x):
            # Returns 1D pooled descriptor of shape (B, 256)
            return torch.randn(x.shape[0], 256)

    model = DummyGAPSAMExtractorModel()
    manager = DatasetCacheManager(
        model=model,
        output_dir=temp_cache_dir,
        batch_size=2,
        samples_per_shard=2,
        push_to_hub=False,
    )
    assert manager.store_images is True

    ds = DummyImageMaskDataset()
    shards = manager.cache_split("train", max_samples=4, dataset=ds)
    assert len(shards) == 2

    # Check Parquet table schema and content
    for shard in shards:
        tbl = pq.read_table(shard)
        assert "image" in tbl.column_names
        assert "z_high_dim" in tbl.column_names
        assert tbl["channels"][0].as_py() == 256
        assert tbl["latent_h"][0].as_py() == 1
        assert tbl["latent_w"][0].as_py() == 1
        # Image bytes can be decoded into PIL image
        img_bytes = tbl["image"][0].as_py()
        assert len(img_bytes) > 0
        img = Image.open(io.BytesIO(img_bytes))
        assert img.size == (256, 256)





