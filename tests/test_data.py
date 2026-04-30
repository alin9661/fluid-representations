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
    assert set(sample.keys()) == {"video", "label", "alpha", "zeta", "alpha_raw", "zeta_raw"}
    assert sample["video"].shape == (11, 4, 16, 16)
    assert sample["video"].dtype == torch.float32
    assert isinstance(sample["label"], int)
    assert 0 <= sample["label"] < 4
    # Raw values match the conftest fixture's per-object α/ζ assignment.
    assert sample["alpha_raw"] in (1.0, 3.0)
    assert sample["zeta_raw"] in (5.0, 7.0)
    # Normalized: with α∈{1.0, 3.0} (mean=2, std=1) and ζ∈{5.0, 7.0} (mean=6, std=1),
    # std is well above the floor so normalized values are exactly ±1.
    assert min(abs(sample["alpha"] - 1.0), abs(sample["alpha"] + 1.0)) < 1e-6
    assert min(abs(sample["zeta"] - 1.0), abs(sample["zeta"] + 1.0)) < 1e-6


def test_alpha_zeta_zscore_normalization(synthetic_active_matter_shard):
    """All normalized α/ζ across the dataset have mean ≈ 0 and std ≈ 1.

    With the conftest fixture (α=(1.0, 3.0), ζ=(5.0, 7.0), 4 windows per obj × 2
    objs = 8 windows), train mean is exactly (2, 6) and population std is (1, 1),
    so normalized values are ±1 and span the dataset symmetrically.
    """
    ds = ActiveMatterVideoDataset(
        root=synthetic_active_matter_shard, split="train",
        num_frames=4, bins_per_param=2, transform=None,
    )
    alphas = np.array([ds[i]["alpha"] for i in range(len(ds))])
    zetas = np.array([ds[i]["zeta"] for i in range(len(ds))])
    assert abs(alphas.mean()) < 1e-4
    assert abs(alphas.std() - 1.0) < 1e-3
    assert abs(zetas.mean()) < 1e-4
    assert abs(zetas.std() - 1.0) < 1e-3


def test_param_stats_property_and_val_uses_train_stats(shard_factory):
    """`val_ds(param_stats=train_ds.param_stats)` normalizes val against train
    statistics, not its own — preventing probe-metric drift between splits.
    """
    # Train: α∈{1.0, 3.0} ⇒ mean=2.0, std=1.0.
    root = shard_factory(
        subdir="train", alpha=(1.0, 3.0), zeta=(5.0, 7.0), T=8, H=8, W=8,
    )
    train_ds = ActiveMatterVideoDataset(
        root=root, split="train", num_frames=4, bins_per_param=2,
    )
    stats = train_ds.param_stats
    assert set(stats.keys()) == {"alpha_mean", "alpha_std", "zeta_mean", "zeta_std"}
    assert stats["alpha_mean"] == pytest.approx(2.0, abs=1e-5)
    assert stats["alpha_std"] == pytest.approx(1.0, abs=1e-3)
    assert stats["zeta_mean"] == pytest.approx(6.0, abs=1e-5)
    assert stats["zeta_std"] == pytest.approx(1.0, abs=1e-3)

    # Val with a deliberately *different* α distribution so per-split stats
    # disagree with train stats: α∈{1.0, 5.0, 9.0} ⇒ mean=5.0, std≈3.27.
    # norm(1.0) ≈ -1.22 under val's own stats, but -1.0 under train stats.
    shard_factory(
        subdir="valid", n_objs=3, alpha=(1.0, 5.0, 9.0), zeta=(5.0, 7.0, 9.0),
        T=8, H=8, W=8,
    )
    val_no_stats = ActiveMatterVideoDataset(
        root=root, split="valid", num_frames=4, bins_per_param=2,
    )
    val_with_stats = ActiveMatterVideoDataset(
        root=root, split="valid", num_frames=4, bins_per_param=2,
        param_stats=train_ds.param_stats,
    )

    def alpha_at_raw_1(ds):
        for i in range(len(ds)):
            s = ds[i]
            if s["alpha_raw"] == 1.0:
                return s["alpha"]
        raise AssertionError("no sample with alpha_raw=1.0")

    a_self = alpha_at_raw_1(val_no_stats)
    a_train = alpha_at_raw_1(val_with_stats)
    assert abs(a_self - a_train) > 0.1, (
        f"val with own stats ({a_self}) should differ from val with train stats ({a_train})"
    )
    assert a_train == pytest.approx(-1.0, abs=1e-4)


def test_constant_alpha_warns_and_floors_std(shard_factory, caplog):
    """Constant α (std=0) warns loudly and floors std at eps — not silent.

    Mirrors the convention in `compute_stats` (warns on near-zero channel
    variance). Without this guard, normalized α would be `(x - mean) / 1e-6`
    ⇒ ±1e6-magnitude regression targets that explode probe loss while the
    user blames the optimizer.
    """
    root = shard_factory(alpha=(2.0, 2.0), zeta=(5.0, 7.0), T=8, H=8, W=8)
    with caplog.at_level(logging.WARNING, logger="physics_ssl.data"):
        ds = ActiveMatterVideoDataset(
            root=root, split="train", num_frames=4, bins_per_param=2,
        )
    assert any("near-zero variance" in rec.message for rec in caplog.records)
    # Floored exactly at eps (max(0, eps), not 0+eps).
    assert ds.param_stats["alpha_std"] == pytest.approx(1e-6, abs=1e-12)
    # Normalized α stays bounded — `(2.0 - 2.0)/1e-6 = 0`, not blown up.
    sample = ds[0]
    assert sample["alpha"] == pytest.approx(0.0, abs=1e-9)
    assert sample["alpha_raw"] == 2.0


def test_param_stats_validation_rejects_bad_input(shard_factory):
    """`param_stats=` adoption validates keys, types, and positivity up front."""
    root = shard_factory(subdir="valid", alpha=(1.0, 3.0), zeta=(5.0, 7.0), T=8, H=8, W=8)
    base_kwargs = dict(root=root, split="valid", num_frames=4, bins_per_param=2)

    with pytest.raises(ValueError, match="missing keys"):
        ActiveMatterVideoDataset(**base_kwargs, param_stats={"alpha_mean": 0.0})

    bad_type = {"alpha_mean": "x", "alpha_std": 1.0, "zeta_mean": 0.0, "zeta_std": 1.0}
    with pytest.raises(ValueError, match="not numeric"):
        ActiveMatterVideoDataset(**base_kwargs, param_stats=bad_type)

    for bad_std in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="positive finite float"):
            ActiveMatterVideoDataset(
                **base_kwargs,
                param_stats={"alpha_mean": 0.0, "alpha_std": bad_std,
                             "zeta_mean": 0.0, "zeta_std": 1.0},
            )


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
