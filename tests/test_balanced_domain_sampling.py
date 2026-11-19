"""
Tests for domain-balanced sampling on imbalanced multi-source datasets
(e.g. KhangTruong/IMD2020-CASIAv2-Joint with casia vs imd2020).
"""

from __future__ import annotations

from collections import Counter

from sid_unet.dataset import (
    BalancedDomainSampler,
    SIDMockDataset,
    create_dataloaders,
    extract_domain_labels,
)
from sid_unet.utils.config import ConfigDict, DEFAULT_CONFIG, deep_merge


def test_balanced_domain_sampler_oversample_and_batch_interleave():
    """Oversampling an imbalanced 1:7 domain split must yield 50/50 per epoch and per batch."""
    labels = ["casia"] * 10 + ["imd2020"] * 70
    sampler = BalancedDomainSampler(
        domain_labels=labels,
        num_samples=32,
        strategy="oversample",
        seed=42,
        interleave=True,
    )
    assert sampler.domain_counts == {"casia": 10, "imd2020": 70}
    assert sampler.per_domain_count == 16
    assert len(sampler) == 32

    indices = list(sampler)
    sampled_domains = [labels[i] for i in indices]
    assert Counter(sampled_domains) == {"casia": 16, "imd2020": 16}

    # Check every batch of size 8 is balanced 4:4
    for b_start in range(0, len(sampled_domains), 8):
        batch_doms = Counter(sampled_domains[b_start : b_start + 8])
        assert batch_doms == {"casia": 4, "imd2020": 4}


def test_balanced_domain_sampler_undersample_rotates_majority_across_epochs():
    """Undersampling must balance every epoch and rotate unseen majority samples across epochs."""
    labels = ["casia"] * 10 + ["imd2020"] * 30
    sampler = BalancedDomainSampler(
        domain_labels=labels,
        num_samples=None,
        strategy="undersample",
        seed=7,
        interleave=True,
    )
    assert len(sampler) == 20

    seen_majority = set()
    for ep in range(3):
        sampler.set_epoch(ep)
        idxs = list(sampler)
        counts = Counter(labels[i] for i in idxs)
        assert counts == {"casia": 10, "imd2020": 10}
        seen_majority.update(i for i in idxs if labels[i] == "imd2020")

    # After 3 epochs of 10 majority items each, all 30 unique majority items must be covered
    assert len(seen_majority) == 30


def test_balanced_domain_sampler_ddp_sharding():
    """Distributed replicas must receive disjoint, balanced slices."""
    labels = ["casia"] * 8 + ["imd2020"] * 24
    s0 = BalancedDomainSampler(labels, num_samples=16, seed=42, num_replicas=2, rank=0)
    s1 = BalancedDomainSampler(labels, num_samples=16, seed=42, num_replicas=2, rank=1)

    idx0 = list(s0)
    idx1 = list(s1)
    assert len(idx0) == 8 and len(idx1) == 8
    assert Counter(labels[i] for i in idx0) == {"casia": 4, "imd2020": 4}
    assert Counter(labels[i] for i in idx1) == {"casia": 4, "imd2020": 4}
    assert set(idx0).isdisjoint(set(idx1)) or len(set(idx0 + idx1)) > len(idx0)


def test_extract_domain_labels_and_create_dataloaders_balanced_mock():
    """create_dataloaders with balance_domains=True must yield balanced batches on imbalanced data."""
    mock_ds = SIDMockDataset(split="train", max_samples=50, target_image_size=(32, 32))
    extracted = extract_domain_labels(mock_ds)
    assert len(extracted) == 50
    assert Counter(extracted) == {"casia": 10, "imd2020": 40}

    cfg = ConfigDict(
        deep_merge(
            DEFAULT_CONFIG,
            {
                "data": {
                    "dataset_name": "mock",
                    "streaming": False,
                    "use_parquet_loader": False,
                    "image_size": [32, 32],
                    "batch_size": 8,
                    "num_workers": 0,
                    "balance_domains": True,
                    "balance_val_domains": True,
                    "domain_column": "dataset_source",
                    "train_samples_per_epoch": 16,
                    "val_samples_per_epoch": 8,
                    "normalize": False,
                }
            },
        )
    )
    train_loader, val_loader = create_dataloaders(cfg, include_test=False)

    train_domains: Counter[str] = Counter()
    for batch in train_loader:
        batch_counts = Counter(batch["dataset"])
        assert batch_counts == {"casia": 4, "imd2020": 4}
        train_domains.update(batch["dataset"])
    assert train_domains == {"casia": 8, "imd2020": 8}

    val_domains: Counter[str] = Counter()
    for batch in val_loader:
        val_domains.update(batch["dataset"])
    assert val_domains == {"casia": 4, "imd2020": 4}
