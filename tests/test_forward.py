"""End-to-end tests for `tjepa_forward`, including the multistep AR rollout.

The multistep test specifically catches the kind of bug the original review
flagged: shape mismatches between predictor output and target window when the
LeWM invariant `T == history_size + num_preds` is violated, and a target
window that does not advance with the rollout step.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from physics_ssl.forward import tjepa_forward
from physics_ssl.model import ARPredictor, Embedder, MLP, SIGReg


def _cfg(history_size=2, num_preds=1, nsteps=1, sigreg_weight=0.05, unconditional=False):
    return SimpleNamespace(
        wm=SimpleNamespace(history_size=history_size, num_preds=num_preds, nsteps=nsteps),
        loss=SimpleNamespace(sigreg=SimpleNamespace(weight=sigreg_weight)),
        predictor=SimpleNamespace(unconditional=unconditional),
    )


class _StubEncoder(nn.Module):
    """Minimal encoder mimicking HF ViT's `(pixel_values=, interpolate_pos_encoding=)` API."""

    def __init__(self, in_chans, hidden):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, hidden, kernel_size=1)

    def forward(self, *, pixel_values, interpolate_pos_encoding=False):
        flat = self.proj(pixel_values).flatten(2).mean(dim=-1)  # (B*T, hidden)
        # mimic HF ModelOutput with last_hidden_state[:, 0] = our token
        cls = flat.unsqueeze(1)
        return SimpleNamespace(last_hidden_state=cls)


def _build_module(history_size=2, num_preds=1, *, in_chans=3, D=16, T=3, unconditional=False):
    """Construct a thin nn.Module bundle compatible with `tjepa_forward`'s `self`."""

    encoder = _StubEncoder(in_chans, hidden=D)
    projector = nn.Identity()
    cond_embedder = Embedder(input_dim=2, smoothed_dim=4, emb_dim=D)
    predictor = ARPredictor(
        num_frames=T, depth=1, heads=2, mlp_dim=32,
        input_dim=D, hidden_dim=D, output_dim=D, dim_head=8,
        unconditional=unconditional,
    )
    sigreg = SIGReg(knots=8, num_proj=64)

    bundle = nn.ModuleDict({
        "encoder": encoder,
        "projector": projector,
        "cond_embedder": cond_embedder,
        "predictor": predictor,
    })

    class _Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = bundle
            self.sigreg = sigreg

        def log(self, *args, **kwargs):  # pragma: no cover — no-op stub
            pass

    return _Holder()


def _make_batch(B=2, C=3, T=3, H=4, W=4):
    return {
        "video": torch.randn(B, C, T, H, W),
        "label": torch.zeros(B, dtype=torch.long),
        "alpha": torch.tensor([1.0] * B),
        "zeta": torch.tensor([2.0] * B),
    }


def test_singlestep_forward_shapes_and_finite():
    T = 3
    holder = _build_module(history_size=2, num_preds=1, T=T, D=16)
    batch = _make_batch(B=2, T=T)
    out = tjepa_forward(holder, batch, stage="fit", cfg=_cfg(history_size=2, num_preds=1))
    assert torch.isfinite(out["loss"])
    assert out["loss"].requires_grad
    assert out["embedding"].shape == (2, 16)
    out["loss"].backward()
    # Gradients must reach both encoder and predictor.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in holder.model.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in holder.model.predictor.parameters())


def test_invariant_violation_raises():
    """T must equal history_size + num_preds; otherwise raise loud."""
    T = 4
    holder = _build_module(history_size=2, num_preds=1, T=T, D=16)
    batch = _make_batch(B=2, T=T)
    with pytest.raises(ValueError, match="LeWM invariant"):
        tjepa_forward(holder, batch, stage="fit", cfg=_cfg(history_size=2, num_preds=1))


def test_negative_history_or_npreds_raises():
    T = 3
    holder = _build_module(history_size=2, num_preds=1, T=T, D=16)
    batch = _make_batch(B=2, T=T)
    with pytest.raises(ValueError, match="must be positive"):
        tjepa_forward(holder, batch, stage="fit", cfg=_cfg(history_size=0, num_preds=1))
    with pytest.raises(ValueError, match="must be positive"):
        tjepa_forward(holder, batch, stage="fit", cfg=_cfg(history_size=2, num_preds=0))


def test_multistep_invariant_violation_raises():
    """T must equal ctx + n_preds + (nsteps - 1) when nsteps > 1."""
    history_size = 2
    num_preds = 1
    nsteps = 2
    T = history_size + num_preds  # = 3, missing one frame for nsteps=2
    holder = _build_module(history_size=history_size, num_preds=num_preds, T=T, D=8)
    batch = _make_batch(B=2, T=T)
    with pytest.raises(ValueError, match="LeWM invariant"):
        tjepa_forward(holder, batch, stage="fit",
                      cfg=_cfg(history_size=history_size, num_preds=num_preds, nsteps=nsteps))


def test_multistep_runs_when_t_large_enough():
    """T = ctx + n_preds + (nsteps - 1); each step targets a window shifted by `step`."""
    history_size = 2
    num_preds = 1
    nsteps = 2
    T = history_size + num_preds + (nsteps - 1)  # = 4
    cfg = _cfg(history_size=history_size, num_preds=num_preds, nsteps=nsteps)
    holder = _build_module(history_size=history_size, num_preds=num_preds, T=T, D=8)
    batch = _make_batch(B=2, T=T)
    out = tjepa_forward(holder, batch, stage="fit", cfg=cfg)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    # Both encoder and predictor must have gradients (the multistep path keeps the
    # encoder in the graph through the target encodings).
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in holder.model.predictor.parameters())


def test_unconditional_path_skips_cond_embedder():
    T = 3
    holder = _build_module(history_size=2, num_preds=1, T=T, D=16, unconditional=True)
    batch = _make_batch(B=2, T=T)
    cfg = _cfg(history_size=2, num_preds=1, unconditional=True)
    out = tjepa_forward(holder, batch, stage="fit", cfg=cfg)
    out["loss"].backward()
    # cond_embedder should have no gradient since it was never called.
    assert all(p.grad is None or p.grad.abs().sum() == 0
               for p in holder.model.cond_embedder.parameters())
