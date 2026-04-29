"""Pytest fixtures.

The `synthetic_active_matter_shard` fixture builds a tiny HDF5 file matching
The Well's Active Matter layout: 5 scalar + 1 vector(2) + 1 tensor(2,2) = 11
channels, plus `scalars/{alpha, zeta, L}`. It writes the shard to a temp dir
and yields the parent directory so a dataset can be constructed against it.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest


def _make_shard(
    path: Path,
    n_objs: int = 2,
    T: int = 16,
    H: int = 16,
    W: int = 16,
    *,
    alpha=(1.0, 3.0),
    zeta=(5.0, 7.0),
    schema: str = "default",
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        if schema == "default":
            # 5 scalar (5) + 1 vector(2) + 1 tensor(2,2)=4 -> 11
            t0 = f.create_group("t0_fields")
            for n in "abcde":
                t0.create_dataset(n, data=rng.standard_normal((n_objs, T, H, W)).astype("f4"))
            t1 = f.create_group("t1_fields")
            t1.create_dataset("vec", data=rng.standard_normal((n_objs, T, H, W, 2)).astype("f4"))
            t2 = f.create_group("t2_fields")
            t2.create_dataset("tens", data=rng.standard_normal((n_objs, T, H, W, 2, 2)).astype("f4"))
        elif schema == "all_scalar":
            t0 = f.create_group("t0_fields")
            for i in range(11):
                t0.create_dataset(f"s{i}", data=rng.standard_normal((n_objs, T, H, W)).astype("f4"))
        else:
            raise ValueError(f"unknown schema {schema!r}")

        sc = f.create_group("scalars")
        sc.create_dataset("alpha", data=np.asarray(alpha, dtype="f4"))
        sc.create_dataset("zeta", data=np.asarray(zeta, dtype="f4"))
        sc.create_dataset("L", data=np.asarray([8.0] * n_objs, dtype="f4"))


@pytest.fixture(scope="session")
def synthetic_active_matter_shard(tmp_path_factory):
    """Tmp dir containing `train/fake.h5` with the default 11-channel schema."""
    root = tmp_path_factory.mktemp("am_shard")
    _make_shard(root / "train" / "fake.h5")
    return root


@pytest.fixture()
def shard_factory(tmp_path):
    """Functional builder for tests that want non-default schemas / params."""
    def _build(**kwargs):
        sub = kwargs.pop("subdir", "train")
        out = tmp_path / sub / "fake.h5"
        _make_shard(out, **kwargs)
        return tmp_path
    return _build
