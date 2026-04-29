"""Temporal-AR JEPA on physics video data, on top of stable-pretraining.

Submodules import their own deps lazily so importing one (e.g. `model`) does
not require the others' deps (e.g. lightning, stable_pretraining).
"""

import importlib
from typing import TYPE_CHECKING, Any

__all__ = [
    "ActiveMatterVideoDataset",
    "ChannelZScore",
    "AddGaussianNoise",
    "SIGReg",
    "ARPredictor",
    "MLP",
    "Embedder",
    "build_per_frame_encoder",
    "tjepa_forward",
    "build_tjepa_module",
    "RegressionProbe",
]

_LAZY_MAP = {
    "ActiveMatterVideoDataset": "physics_ssl.data",
    "ChannelZScore": "physics_ssl.transforms",
    "AddGaussianNoise": "physics_ssl.transforms",
    "SIGReg": "physics_ssl.model",
    "ARPredictor": "physics_ssl.model",
    "MLP": "physics_ssl.model",
    "Embedder": "physics_ssl.model",
    "build_per_frame_encoder": "physics_ssl.encoder",
    "tjepa_forward": "physics_ssl.forward",
    "build_tjepa_module": "physics_ssl.forward",
    "RegressionProbe": "physics_ssl.callbacks",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_MAP:
        module = importlib.import_module(_LAZY_MAP[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'physics_ssl' has no attribute {name!r}")


if TYPE_CHECKING:
    from .callbacks import RegressionProbe
    from .data import ActiveMatterVideoDataset
    from .encoder import build_per_frame_encoder
    from .forward import build_tjepa_module, tjepa_forward
    from .model import MLP, ARPredictor, Embedder, SIGReg
    from .transforms import AddGaussianNoise, ChannelZScore
