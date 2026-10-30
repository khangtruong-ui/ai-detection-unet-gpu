"""
Comprehensive test suite for Learnable Forensic Vector Memory for Detective SAM.
Covers all 17 required verification areas:
1. Query replacement forward semantics
2. Query replacement gradient semantics
3. CPU memory storage
4. Key/value pairing
5. Similarity computation
6. Retrieval with fewer entries than requested k
7. Empty-memory warm-up policy
8. Zero-norm vector handling
9. Bounded database size
10. Consolidation determinism
11. Correct mask-to-feature coordinate mapping
12. Exclusion of authentic regions from positive-memory collection
13. No memory updates during validation or inference
14. Checkpoint save and reload
15. Baseline behavior when memory is disabled
16. Compatibility between saved memory dimensions and the model
17. End-to-end forward pass using genuine pretrained weights
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sid_unet.models.forensic_memory import (
    ForensicVectorMemory,
    stop_gradient_query_replacement,
)
from sid_unet.models.detective_sam import DetectiveSAM
from sid_unet.models.unet import build_model
from sid_unet.utils.config import ConfigDict


# =============================================================================
# 1. Query replacement forward semantics
# =============================================================================

def test_query_replacement_forward_semantics():
    """Verify forward pass of Rx = R + (Rt - R).detach() evaluates exactly to Rt."""
    b, n, d = 2, 64, 64
    r = torch.randn(b, n, d, requires_grad=True)
    rt = torch.randn(b, n, d, requires_grad=False)

    rx = stop_gradient_query_replacement(r, rt)

    # Forward value must match Rt within numerical precision
    assert rx.shape == rt.shape
    assert torch.allclose(rx, rt, atol=1e-5), "Forward pass of Rx must evaluate to Rt"


# =============================================================================
# 2. Query replacement gradient semantics
# =============================================================================

def test_query_replacement_gradient_semantics():
    """
    Verify gradient semantics:
        backward(dRx/dR) = Identity
        backward(dRx/dRt) = Zero
    """
    b, n, d = 2, 64, 64
    r = torch.randn(b, n, d, requires_grad=True)
    rt = torch.randn(b, n, d, requires_grad=True)

    rx = stop_gradient_query_replacement(r, rt)

    # Upstream loss computation
    weight = torch.randn(b, n, d)
    loss = (rx * weight).sum()
    loss.backward()

    # 1. Gradient with respect to R must equal the identity path (weight)
    assert r.grad is not None, "R must receive gradient"
    assert r.grad.shape == r.shape
    assert torch.allclose(r.grad, weight, atol=1e-7), "dRx/dR must be Identity"

    # 2. Gradient with respect to Rt must be zero / None
    assert rt.grad is None, "Rt must NOT receive any gradient through stop-gradient replacement"


# =============================================================================
# 3. CPU memory storage
# =============================================================================

def test_cpu_memory_storage():
    """Verify memory keys and values reside strictly in CPU RAM even when added from GPU."""
    mem = ForensicVectorMemory(
        memory_enabled=True,
        memory_key_dim=32,
        memory_value_dim=32,
    )

    # Add observations from CUDA if available, else CPU
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    obs_k = torch.randn(10, 32, device=dev, requires_grad=True)
    obs_v = torch.randn(10, 32, device=dev, requires_grad=True)

    added = mem.add_observations(obs_k, obs_v)
    assert added == 10

    # Before consolidation, buffer tensors must be on CPU and detached
    for k_buf, v_buf in zip(mem._epoch_observed_keys, mem._epoch_observed_values):
        assert k_buf.device.type == "cpu"
        assert v_buf.device.type == "cpu"
        assert not k_buf.requires_grad
        assert not v_buf.requires_grad

    # Consolidate and check authoritative storage
    mem.consolidate()
    assert mem._keys.device.type == "cpu"
    assert mem._values.device.type == "cpu"
    assert mem.entry_count == 10
    assert not mem._keys.requires_grad


# =============================================================================
# 4. Key/value pairing
# =============================================================================

def test_key_value_pairing():
    """Verify keys and values remain correctly paired after storage, deduplication, and retrieval."""
    mem = ForensicVectorMemory(
        memory_enabled=True,
        memory_key_dim=4,
        memory_value_dim=4,
        memory_retrieval_k=1,
        memory_similarity_metric="cosine",
    )

    # Create distinct identifiable pairs: key_i has 1.0 at index i, value_i has 100 * (i+1)
    k1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    v1 = torch.tensor([[100.0, 100.0, 100.0, 100.0]], dtype=torch.float32)

    k2 = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    v2 = torch.tensor([[200.0, 200.0, 200.0, 200.0]], dtype=torch.float32)

    k3 = torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    v3 = torch.tensor([[300.0, 300.0, 300.0, 300.0]], dtype=torch.float32)

    mem.add_observations(k1, v1)
    mem.add_observations(k2, v2)
    mem.add_observations(k3, v3)
    mem.consolidate()

    # Query with k2 -> must retrieve v2 exactly
    q = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    ret_v, ret_k, _ = mem.retrieve(q, k=1)

    assert ret_v is not None
    assert torch.allclose(ret_v[0, 0], v2[0]), "Retrieved value must correspond exactly to queried key"
    assert torch.allclose(ret_k[0, 0], k2[0])


# =============================================================================
# 5. Similarity computation
# =============================================================================

def test_similarity_computation():
    """Verify similarity computation metrics (cosine, l2, dot) on CPU."""
    # 1. Cosine similarity
    mem_cos = ForensicVectorMemory(memory_key_dim=3, memory_similarity_metric="cosine")
    q = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    k = torch.tensor([[2.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    sim_cos = mem_cos._compute_similarity(q, k)
    # q[0] aligned with k[0] (cos=1.0), opposite to k[1] (cos=-1.0), orthogonal to k[2] (cos=0.0)
    assert pytest.approx(sim_cos[0, 0].item(), 1e-5) == 1.0
    assert pytest.approx(sim_cos[0, 1].item(), 1e-5) == -1.0
    assert pytest.approx(sim_cos[0, 2].item(), 1e-5) == 0.0

    # 2. L2 distance similarity (negative euclidean)
    mem_l2 = ForensicVectorMemory(memory_key_dim=3, memory_similarity_metric="l2")
    sim_l2 = mem_l2._compute_similarity(q, k)
    # dist(q[0], k[0]) = ||[1,0,0] - [2,0,0]|| = 1.0 -> sim = -1.0
    assert pytest.approx(sim_l2[0, 0].item(), 1e-4) == -1.0

    # 3. Dot product similarity
    mem_dot = ForensicVectorMemory(memory_key_dim=3, memory_similarity_metric="dot")
    sim_dot = mem_dot._compute_similarity(q, k)
    assert pytest.approx(sim_dot[0, 0].item(), 1e-5) == 2.0


# =============================================================================
# 6. Retrieval with fewer entries than requested k
# =============================================================================

def test_retrieval_fewer_entries_than_k():
    """Verify retrieval returns available entries when entry_count < requested k."""
    mem = ForensicVectorMemory(
        memory_enabled=True,
        memory_key_dim=16,
        memory_value_dim=16,
        memory_retrieval_k=2,  # Warm up at 2 entries
    )
    keys = torch.randn(3, 16)
    mem.add_observations(keys)
    mem.consolidate()
    assert mem.entry_count == 3

    # Request k=5 when only 3 entries are stored
    q = torch.randn(1, 16)
    ret_v, ret_k, _ = mem.retrieve(q, k=5)

    assert ret_v is not None
    assert ret_v.shape == (1, 3, 16), "Should return all available 3 entries when requested k=5"
    assert torch.isfinite(ret_v).all()


# =============================================================================
# 7. Empty-memory warm-up policy
# =============================================================================

def test_empty_memory_warmup():
    """Verify empty or under-capacity memory bypasses retrieval cleanly without errors or NaNs."""
    mem = ForensicVectorMemory(
        memory_enabled=True,
        memory_key_dim=16,
        memory_value_dim=16,
        memory_retrieval_k=4,
    )

    # 1. Empty database: entry_count == 0
    assert not mem.is_warmed_up
    q = torch.randn(2, 16)
    ret_v, ret_k, lat = mem.retrieve(q)
    assert ret_v is None, "Empty memory must return None (warm-up bypass)"
    assert ret_k is None
    assert lat == 0.0

    # 2. Undersized database: 2 entries < requested k=4
    mem.add_observations(torch.randn(2, 16))
    mem.consolidate()
    assert mem.entry_count == 2
    assert not mem.is_warmed_up, "Memory must not be marked warmed up when entry_count < retrieval_k"
    ret_v, ret_k, _ = mem.retrieve(q)
    assert ret_v is None, "Undersized memory below retrieval_k must bypass retrieval"


# =============================================================================
# 8. Zero-norm vector handling
# =============================================================================

def test_zero_norm_vector_handling():
    """Verify cosine similarity handles zero-norm query or key vectors safely without producing NaNs."""
    mem = ForensicVectorMemory(
        memory_enabled=True,
        memory_key_dim=4,
        memory_value_dim=4,
        memory_similarity_metric="cosine",
    )
    zero_key = torch.zeros(1, 4)
    normal_key = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mem.add_observations(torch.cat([zero_key, normal_key], dim=0))
    mem.consolidate()

    zero_query = torch.zeros(1, 4)
    sim = mem._compute_similarity(zero_query, mem._keys)

    assert not torch.isnan(sim).any(), "Cosine similarity with zero vectors must not produce NaNs"
    assert torch.isfinite(sim).all()
    assert sim[0, 0].item() == 0.0


# =============================================================================
# 9. Bounded database size
# =============================================================================

def test_bounded_database_size():
    """Verify memory_max_entries strictly caps the authoritative memory size."""
    max_cap = 25
    mem = ForensicVectorMemory(
        memory_enabled=True,
        memory_key_dim=8,
        memory_value_dim=8,
        memory_max_entries=max_cap,
        memory_consolidation_method="dedup_greedy",
    )

    # Add 100 random observations
    for _ in range(5):
        mem.add_observations(torch.randn(20, 8))
    stats = mem.consolidate()

    assert mem.entry_count <= max_cap, f"Database must not exceed max_entries={max_cap}"
    assert stats["final_entry_count"] == max_cap
    assert stats["capacity_pruned"] > 0


# =============================================================================
# 10. Consolidation determinism
# =============================================================================

def test_consolidation_determinism():
    """Verify consolidation produces bitwise identical results given the same seed."""
    torch.manual_seed(123)
    obs = torch.randn(60, 16)

    # Run 1
    mem1 = ForensicVectorMemory(
        memory_enabled=True,
        memory_key_dim=16,
        memory_value_dim=16,
        memory_max_entries=20,
        memory_seed=42,
    )
    mem1.add_observations(obs.clone())
    mem1.consolidate()

    # Run 2
    mem2 = ForensicVectorMemory(
        memory_enabled=True,
        memory_key_dim=16,
        memory_value_dim=16,
        memory_max_entries=20,
        memory_seed=42,
    )
    mem2.add_observations(obs.clone())
    mem2.consolidate()

    assert torch.equal(mem1._keys, mem2._keys), "Consolidation must be deterministic given identical seed"
    assert torch.equal(mem1._values, mem2._values)


# =============================================================================
# 11. Correct mask-to-feature coordinate mapping
# =============================================================================

def test_correct_mask_to_feature_coordinate_mapping():
    """Verify downscaling of ground truth masks maps coordinates accurately to feature maps."""
    model = DetectiveSAM(
        layer_width=64,
        target_size=(128, 128),
        use_dummy=True,
        memory_enabled=True,
        memory_key_dim=64,
        memory_value_dim=64,
        memory_vectors_per_image=4,
    )
    model.train()

    # Feature map resolution s_hat = (32, 32) for 128x128 input
    f_fuse = torch.zeros(1, 64, 32, 32)
    # Ground truth mask with a 16x16 tampered patch in the top-left quadrant of 128x128 image
    gt_mask = torch.zeros(1, 1, 128, 128)
    gt_mask[0, 0, :32, :32] = 1.0

    collected = model._collect_tampered_memory(f_fuse, gt_mask)
    assert collected > 0, "Tampered patch must be detected and mapped to feature map"


# =============================================================================
# 12. Exclusion of authentic regions from positive-memory collection
# =============================================================================

def test_exclusion_of_authentic_regions():
    """Verify authentic (all-zero mask) images contribute exactly zero vectors to memory."""
    model = DetectiveSAM(
        layer_width=64,
        use_dummy=True,
        memory_enabled=True,
        memory_key_dim=64,
        memory_value_dim=64,
    )
    model.train()

    f_fuse = torch.randn(2, 64, 32, 32)
    authentic_mask = torch.zeros(2, 1, 128, 128)  # All zeros (untampered)

    collected = model._collect_tampered_memory(f_fuse, authentic_mask)
    assert collected == 0, "Authentic images must contribute 0 vectors to forensic memory"
    assert len(model.forensic_memory._epoch_observed_keys) == 0


# =============================================================================
# 13. No memory updates during validation or inference
# =============================================================================

def test_no_memory_updates_during_validation_or_inference():
    """Verify memory is never updated when model is in eval mode."""
    model = DetectiveSAM(
        layer_width=64,
        use_dummy=True,
        memory_enabled=True,
        memory_key_dim=64,
        memory_value_dim=64,
    )
    model.eval()

    x = torch.randn(2, 3, 64, 64)
    gt = torch.ones(2, 1, 64, 64)  # Even if gt_masks is accidentally provided in eval

    # Forward in eval mode
    _ = model(x, gt_masks=gt)

    assert len(model.forensic_memory._epoch_observed_keys) == 0, (
        "Memory observations must NEVER be recorded during evaluation/inference"
    )


# =============================================================================
# 14. Checkpoint save and reload
# =============================================================================

def test_checkpoint_save_and_reload():
    """Verify memory state dict saves and restores completely."""
    mem = ForensicVectorMemory(
        memory_enabled=True,
        memory_key_dim=16,
        memory_value_dim=16,
        memory_retrieval_k=2,
    )
    keys = torch.randn(10, 16)
    values = torch.randn(10, 16)
    mem.add_observations(keys, values)
    mem.consolidate()

    state = mem.state_dict()
    assert state["entry_count"] == 10

    # Load into new instance
    new_mem = ForensicVectorMemory(
        memory_enabled=True,
        memory_key_dim=16,
        memory_value_dim=16,
    )
    new_mem.load_state_dict(state)

    assert new_mem.entry_count == 10
    assert torch.equal(new_mem._keys, mem._keys)
    assert torch.equal(new_mem._values, mem._values)

    # Test file-based save and load
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        mem.save(tmp_path)
        loaded_mem = ForensicVectorMemory(memory_key_dim=16, memory_value_dim=16)
        loaded_mem.load(tmp_path)
        assert loaded_mem.entry_count == 10
        assert torch.equal(loaded_mem._keys, mem._keys)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# =============================================================================
# 15. Baseline behavior when memory is disabled
# =============================================================================

def test_baseline_behavior_when_memory_disabled():
    """Verify DetectiveSAM with memory_enabled=False matches original baseline exactly."""
    torch.manual_seed(99)
    x = torch.randn(1, 3, 64, 64)

    # Baseline model
    model_baseline = DetectiveSAM(
        layer_width=64,
        target_size=(64, 64),
        use_dummy=True,
        memory_enabled=False,
    )
    model_baseline.eval()

    # Model with memory disabled
    model_mem_off = DetectiveSAM(
        layer_width=64,
        target_size=(64, 64),
        use_dummy=True,
        memory_enabled=False,
    )
    model_mem_off.eval()

    # Copy identical weights
    model_mem_off.load_state_dict(model_baseline.state_dict())

    with torch.no_grad():
        out_base = model_baseline(x)
        out_mem_off = model_mem_off(x)

    assert torch.equal(out_base, out_mem_off), (
        "Memory-disabled DetectiveSAM must produce bitwise identical output to baseline"
    )


# =============================================================================
# 16. Compatibility between saved memory dimensions and model
# =============================================================================

def test_compatibility_saved_memory_dimensions():
    """Verify loading mismatched memory dimensions raises an informative ValueError."""
    mem_16 = ForensicVectorMemory(memory_enabled=True, memory_key_dim=16, memory_value_dim=16)
    mem_16.add_observations(torch.randn(5, 16))
    mem_16.consolidate()
    state_16 = mem_16.state_dict()

    mem_64 = ForensicVectorMemory(memory_enabled=True, memory_key_dim=64, memory_value_dim=64)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        mem_64.load_state_dict(state_16, strict_dims=True)


# =============================================================================
# 17. End-to-end forward pass using genuine pretrained weights
# =============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_end_to_end_forward_pass_genuine_pretrained():
    """Verify DetectiveSAM with genuine pretrained SAM2 weights executes forward pass with memory retrieval."""
    model = DetectiveSAM(
        variant="small",
        layer_width=64,
        target_size=(512, 512),
        use_dummy=False,
        memory_enabled=True,
        memory_key_dim=64,
        memory_value_dim=64,
        memory_retrieval_k=2,
        device="cuda",
    ).cuda()

    # Pre-populate 5 valid entries into memory
    mem_keys = torch.randn(5, 64)
    mem_vals = torch.randn(5, 64)
    model.forensic_memory.add_observations(mem_keys, mem_vals)
    model.forensic_memory.consolidate()
    assert model.forensic_memory.is_warmed_up

    # Real GPU forward pass
    x = torch.randn(1, 3, 128, 128, device="cuda")
    model.eval()

    with torch.no_grad():
        res = model(x, return_dict=True)

    assert isinstance(res, dict)
    assert "mask_logits" in res
    assert res["mask_logits"].shape == (1, 1, 128, 128)
    assert res["is_warmed_up"] is True
    assert res["retrieval_latency"] >= 0.0
    assert torch.isfinite(res["mask_logits"]).all()


# =============================================================================
# 18. Database Checkpointing, Sibling Loading, and Hub Artifact Discovery
# =============================================================================

def test_checkpoint_manager_saves_and_restores_database(tmp_path):
    """Verify CheckpointManager saves standalone forensic_memory.pt and restores it cleanly."""
    from sid_unet.training.callbacks import CheckpointManager

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    mgr = CheckpointManager(
        checkpoint_dir=str(ckpt_dir),
        save_best=True,
        save_latest=True,
        push_to_hub=False,
    )

    model = DetectiveSAM(
        layer_width=64,
        use_dummy=True,
        memory_enabled=True,
        memory_key_dim=64,
        memory_value_dim=64,
    )
    # Add entries
    model.forensic_memory.add_observations(torch.randn(8, 64), torch.randn(8, 64))
    model.forensic_memory.consolidate()
    assert model.forensic_memory.entry_count == 8

    # Save periodic checkpoint
    saved = mgr.periodic_save(
        model=model,
        epoch=1,
        step=10,
        config={"model": {"name": "detective_sam"}},
    )
    assert "memory" in saved
    assert os.path.exists(saved["memory"])
    assert os.path.basename(saved["memory"]) == "forensic_memory.pt"

    # Save best checkpoint
    saved_best = mgr.save_checkpoint(
        model=model,
        epoch=1,
        step=10,
        current_score=0.95,
        config={"model": {"name": "detective_sam"}},
    )
    assert os.path.exists(os.path.join(str(ckpt_dir), "forensic_memory_best.pt"))

    # Reload into a fresh model
    new_model = DetectiveSAM(
        layer_width=64,
        use_dummy=True,
        memory_enabled=True,
        memory_key_dim=64,
        memory_value_dim=64,
    )
    assert new_model.forensic_memory.entry_count == 0

    mgr.load_checkpoint(
        checkpoint_path=saved["latest"],
        model=new_model,
    )
    assert new_model.forensic_memory.entry_count == 8


def test_unet_from_checkpoint_loads_database_nicely(tmp_path):
    """Verify UNet.from_checkpoint restores forensic memory from both embedded dict and sibling file."""
    from sid_unet.models.unet import UNet

    # 1. Test embedded state dict restoration
    ckpt_path = tmp_path / "model_ckpt.pt"
    mem = ForensicVectorMemory(memory_enabled=True, memory_key_dim=64, memory_value_dim=64)
    mem.add_observations(torch.randn(12, 64))
    mem.consolidate()

    state = {
        "model_state_dict": {},
        "forensic_memory_state_dict": mem.state_dict(),
        "config": {
            "model": {
                "name": "detective_sam",
                "use_dummy": True,
                "memory_enabled": True,
                "memory_key_dim": 64,
                "memory_value_dim": 64,
            }
        }
    }
    torch.save(state, str(ckpt_path))

    loaded_model = UNet.from_checkpoint(str(ckpt_path))
    assert loaded_model.forensic_memory is not None
    assert loaded_model.forensic_memory.entry_count == 12

    # 2. Test sibling file restoration (when forensic_memory_state_dict is omitted from main checkpoint)
    sibling_ckpt_dir = tmp_path / "sibling_run" / "checkpoints"
    sibling_ckpt_dir.mkdir(parents=True)
    sibling_model_pt = sibling_ckpt_dir / "checkpoint_best.pt"
    sibling_mem_pt = sibling_ckpt_dir / "forensic_memory.pt"

    # Save standalone memory file
    mem.save(str(sibling_mem_pt))

    # Save checkpoint without embedded memory state dict
    clean_state = {
        "model_state_dict": {},
        "config": {
            "model": {
                "name": "detective_sam",
                "use_dummy": True,
                "memory_enabled": True,
                "memory_key_dim": 64,
                "memory_value_dim": 64,
            }
        }
    }
    torch.save(clean_state, str(sibling_model_pt))

    loaded_sibling_model = UNet.from_checkpoint(str(sibling_model_pt))
    assert loaded_sibling_model.forensic_memory is not None
    assert loaded_sibling_model.forensic_memory.entry_count == 12


def test_checkpoint_sync_finds_database_artifacts(tmp_path):
    """Verify find_local_checkpoint_artifacts discovers forensic_memory.pt for synchronization."""
    from sid_unet.checkpoint_sync import find_local_checkpoint_artifacts

    run_dir = tmp_path / "outputs" / "RUN" / "detective_sam" / "checkpoints"
    run_dir.mkdir(parents=True)

    torch.save({"dummy": 1}, str(run_dir / "checkpoint_best.pt"))
    torch.save({"keys": torch.randn(5, 64)}, str(run_dir / "forensic_memory.pt"))
    torch.save({"keys": torch.randn(5, 64)}, str(run_dir / "forensic_memory_best.pt"))

    artifacts = find_local_checkpoint_artifacts(source_dir=str(tmp_path / "outputs" / "RUN" / "detective_sam"))
    assert "checkpoint_best.pt" in artifacts["checkpoints"]
    assert "forensic_memory.pt" in artifacts["checkpoints"]
    assert "forensic_memory_best.pt" in artifacts["checkpoints"]

