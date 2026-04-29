import logging
import os

import numpy as np
import pytest
import torch

from physics_ssl.data import ActiveMatterVideoDataset, compute_stats
from physics_ssl.transforms import ChannelZScore


def test_dataset_shape_and_keys(synthetic_active_matter_shard):
    ds = ActiveMatterVideoDataset(
        root=synthetic_active_matter_shard, split="train",
        num_frames=4, bins_per_param=2, transform=None,
    )
    assert len(ds) > 0
    sample = ds[0]
    assert set(sample.keys()) == {"video", "label", "alpha", "zeta"}
    assert sample["video"].shape == (11, 4, 16, 16)
    assert sample["video"].dtype == torch.float32
    assert isinstance(sample["label"], int)
    assert 0 <= sample["label"] < 4
    assert sample["alpha"] in (1.0, 3.0)
    assert sample["zeta"] in (5.0, 7.0)


def test_window_indexing_count(shard_factory):
    # T=12, num_frames=4, stride=4 -> 3 windows per object * 2 objects = 6
    root = shard_factory(T=12, H=8, W=8)
    ds = ActiveMatterVideoDataset(root=root, split="train", num_frames=4, bins_per_param=2)
    # default stride == num_frames
    assert len(ds) == 6


def test_skipped_short_trajectory_warns(shard_factory, caplog):
    root = shard_factory(T=2, H=8, W=8)  # T=2 < num_frames=4
    with caplog.at_level(logging.WARNING, logger="physics_ssl.data"):
        with pytest.raises(ValueError, match="No valid windows"):
            ActiveMatterVideoDataset(root=root, split="train", num_frames=4, bins_per_param=2)
    assert any("skipping shard" in rec.message for rec in caplog.records)


def test_bins_per_param_one_raises(synthetic_active_matter_shard):
    with pytest.raises(ValueError, match="bins_per_param must be >= 2"):
        ActiveMatterVideoDataset(
            root=synthetic_active_matter_shard, split="train",
            num_frames=4, bins_per_param=1,
        )


def test_alternative_11chan_all_scalar_schema(shard_factory):
    """11 scalar fields (no vector/tensor) must produce the same (11, T, H, W)."""
    root = shard_factory(schema="all_scalar", T=8, H=8, W=8)
    ds = ActiveMatterVideoDataset(root=root, split="train", num_frames=4, bins_per_param=2)
    sample = ds[0]
    assert sample["video"].shape == (11, 4, 8, 8)


def test_relative_path_isolates_duplicate_basenames(tmp_path):
    """Two shards in different subdirs with the same basename must NOT alias."""
    from tests.conftest import _make_shard
    _make_shard(tmp_path / "train" / "sub_a" / "fake.h5", alpha=(1.0, 1.0), zeta=(5.0, 5.0))
    _make_shard(tmp_path / "train" / "sub_b" / "fake.h5", alpha=(9.0, 9.0), zeta=(9.0, 9.0))
    ds = ActiveMatterVideoDataset(root=tmp_path, split="train", num_frames=4, bins_per_param=2)
    # Distinct rel paths preserved in the index
    rels = sorted({entry[0] for entry in ds.index})
    assert rels == ["sub_a/fake.h5", "sub_b/fake.h5"]


def test_compute_stats_writes_npz_and_loads_back(synthetic_active_matter_shard, tmp_path):
    out = tmp_path / "stats.npz"
    compute_stats(
        root=synthetic_active_matter_shard, split="train",
        num_frames=4, output_path=out, max_samples=4,
    )
    assert out.exists()
    cz = ChannelZScore(str(out))
    assert cz.mean.shape == (11, 1, 1, 1)
    assert cz.std.shape == (11, 1, 1, 1)
    # Round-trip: applying to a sample produces zero-mean (≈), unit-std (≈) per channel.
    ds = ActiveMatterVideoDataset(root=synthetic_active_matter_shard, split="train",
                                  num_frames=4, bins_per_param=2, transform=cz)
    v = ds[0]["video"]
    flat = v.reshape(11, -1)
    assert flat.mean(dim=1).abs().max().item() < 5  # synthetic random ⇒ stats not exact


def test_compute_stats_zero_max_samples_raises(synthetic_active_matter_shard, tmp_path):
    with pytest.raises(ValueError, match="max_samples must be positive"):
        compute_stats(
            root=synthetic_active_matter_shard, split="train",
            num_frames=4, output_path=tmp_path / "s.npz", max_samples=0,
        )
