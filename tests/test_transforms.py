import logging

import numpy as np
import pytest
import torch

from physics_ssl.transforms import AddGaussianNoise, ChannelZScore


def test_channel_zscore_roundtrip(tmp_path):
    C = 4
    mean = np.array([0.0, 1.0, -2.0, 5.0], dtype="f4")
    std = np.array([1.0, 0.5, 2.0, 3.0], dtype="f4")
    stats = tmp_path / "s.npz"
    np.savez(stats, mean=mean, std=std)
    cz = ChannelZScore(str(stats))
    # input crafted so the normalization output is exactly zero
    x = torch.from_numpy(mean).view(C, 1, 1, 1).expand(C, 2, 3, 3).float()
    out = cz(x)
    assert torch.allclose(out, torch.zeros_like(out))


def test_channel_zscore_warns_on_zero_std(tmp_path, caplog):
    """Constant channel must surface a logged warning, not be silently rescaled."""
    mean = np.array([0.0, 0.0], dtype="f4")
    std = np.array([1.0, 0.0], dtype="f4")  # second channel constant
    stats = tmp_path / "s.npz"
    np.savez(stats, mean=mean, std=std)
    with caplog.at_level(logging.WARNING, logger="physics_ssl.transforms"):
        ChannelZScore(str(stats))
    assert any("std < " in rec.message and "[1]" in rec.message for rec in caplog.records), \
        f"expected warning naming channel 1, got: {[r.message for r in caplog.records]}"


def test_add_gaussian_noise_passthrough_when_zero():
    f = AddGaussianNoise(0.0)
    x = torch.randn(2, 3, 4)
    assert torch.equal(f(x), x)


def test_add_gaussian_noise_changes_input():
    torch.manual_seed(0)
    f = AddGaussianNoise(0.1)
    x = torch.zeros(1024)
    y = f(x)
    assert not torch.equal(y, x)
    assert pytest.approx(y.std().item(), abs=0.02) == 0.1
