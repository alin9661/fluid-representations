import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ChannelZScore:
    """Per-channel z-score normalization for (C, T, H, W) tensors.

    Stats live in an .npz file with two (C,) arrays: 'mean' and 'std'. The std
    array is clamped to a small floor to avoid divide-by-zero on constant
    channels; we also log a warning naming those channels so a future debugger
    can find them. (`np.load` returns an ``NpzFile`` lazy archive; we close it
    explicitly so DataLoader workers don't leak file descriptors when many
    transforms are constructed.)
    """

    def __init__(self, stats_path: str, eps: float = 1e-6):
        with np.load(stats_path) as stats:
            mean_np = np.asarray(stats["mean"], dtype=np.float32)
            std_np = np.asarray(stats["std"], dtype=np.float32)
        near_zero = np.where(std_np < eps)[0]
        if near_zero.size:
            logger.warning(
                "ChannelZScore: %d channels have std < %g (constant or "
                "near-constant). std will be floored at eps for these: %s",
                near_zero.size, eps, near_zero.tolist(),
            )
        self.mean = torch.from_numpy(mean_np).view(-1, 1, 1, 1)
        self.std = torch.from_numpy(np.maximum(std_np, eps)).view(-1, 1, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class AddGaussianNoise:
    """Optional per-sample Gaussian noise.

    Mirrors the noise-injection step in
    `remy9926/physical-representation-learning/physics_jepa/data.py:269-274`.
    """

    def __init__(self, std: float = 0.0):
        self.std = float(std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.std <= 0.0:
            return x
        return x + torch.randn_like(x) * self.std
