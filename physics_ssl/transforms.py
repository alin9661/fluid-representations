import numpy as np
import torch


class ChannelZScore:
    """Per-channel z-score normalization for (C, T, H, W) tensors.

    Stats live in an .npz file with two (C,) arrays: 'mean' and 'std'. The std
    array is clamped to a small floor to avoid divide-by-zero on constant
    channels (e.g., a boundary-condition channel with the same value everywhere).
    """

    def __init__(self, stats_path: str, eps: float = 1e-6):
        stats = np.load(stats_path)
        mean = torch.from_numpy(stats["mean"]).float().view(-1, 1, 1, 1)
        std = torch.from_numpy(stats["std"]).float().clamp_min(eps).view(-1, 1, 1, 1)
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class AddGaussianNoise:
    """Optional per-sample Gaussian noise; mirrors `physics_jepa/data.py:noise_std`."""

    def __init__(self, std: float = 0.0):
        self.std = float(std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.std <= 0.0:
            return x
        return x + torch.randn_like(x) * self.std
