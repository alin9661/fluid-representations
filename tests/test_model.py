"""Pure-torch tests for `physics_ssl.model` — no Lightning / stable_pretraining deps."""

import pytest
import torch

from physics_ssl.model import ARPredictor, Embedder, MLP, SIGReg


def test_sigreg_signal_random_vs_collapsed():
    """SIGReg should fire on collapsed embeddings and stay quiet on isotropic noise."""
    sigreg = SIGReg()
    torch.manual_seed(0)
    val_random = sigreg(torch.randn(8, 64, 32))
    val_collapsed = sigreg(torch.zeros(8, 64, 32))
    assert torch.isfinite(val_random) and torch.isfinite(val_collapsed)
    # Collapsed batch should produce a strictly larger penalty.
    assert val_collapsed > val_random
    # And a meaningful margin (not within numerical noise).
    assert (val_collapsed - val_random).item() > 1.0


def test_arpredictor_conditional_shape_contract():
    pred = ARPredictor(
        num_frames=8, depth=2, heads=4, mlp_dim=128,
        input_dim=32, hidden_dim=32, output_dim=32, dim_head=8,
    )
    x = torch.randn(2, 3, 32)
    c = torch.randn(2, 3, 32)
    out = pred(x, c)
    assert out.shape == (2, 3, 32)


def test_arpredictor_unconditional_ignores_cond_arg():
    pred = ARPredictor(
        num_frames=8, depth=2, heads=4, mlp_dim=128,
        input_dim=32, hidden_dim=32, output_dim=32, dim_head=8,
        unconditional=True,
    )
    x = torch.randn(2, 3, 32)
    out_with = pred(x, torch.randn(2, 3, 32))
    out_without = pred(x, None)
    assert out_with.shape == out_without.shape == (2, 3, 32)


def test_arpredictor_conditional_rejects_none_cond():
    """A conditional ARPredictor should not silently behave as unconditional
    when given c=None — the user passing None almost certainly means a config bug."""
    pred = ARPredictor(
        num_frames=8, depth=2, heads=4, mlp_dim=128,
        input_dim=32, hidden_dim=32, output_dim=32, dim_head=8,
    )
    with pytest.raises((TypeError, AttributeError)):
        pred(torch.randn(2, 3, 32), None)


def test_embedder_low_dim_input():
    emb = Embedder(input_dim=2, smoothed_dim=10, emb_dim=32)
    out = emb(torch.randn(4, 5, 2))
    assert out.shape == (4, 5, 32)


def test_mlp_projector_with_batchnorm():
    proj = MLP(input_dim=192, output_dim=192, hidden_dim=512, norm_fn=torch.nn.BatchNorm1d)
    out = proj(torch.randn(8, 192))
    assert out.shape == (8, 192)
