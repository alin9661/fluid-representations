"""tjepa_forward — temporal-AR JEPA forward.

Adapted from `le-wm/train.py:18-46` (`lejepa_forward`). Differences:

* Input is `(B, C=11, T, H, W)` physics video, not `(B, T, C, H, W)` pixels.
* Conditioning vector is constant per video — `[alpha, zeta]` broadcast across
  T — not a per-step action sequence.
* Surfaces a pooled `embedding` and continuous `alpha`, `zeta` scalars in the
  output dict so probe callbacks (`OnlineKNN`, `OnlineProbe`,
  `RegressionProbe`) can read them via `get_data_from_batch_or_outputs`.

The optional multistep unroll (`cfg.wm.nsteps > 1`) is a sliding-window
autoregressive rollout in the spirit of `le-wm/jepa.py:61-110`: at each step
the predictor emits one new embedding, the oldest history step is dropped, the
new prediction is appended, and the target window is shifted by one. This
shares the exposure-bias mitigation goal of EB-JEPA's parallel-unroll
(`eb_jepa/eb_jepa/jepa.py:142-157`) but is structurally different — EB-JEPA
prepends GT context each iter; we slide.
"""

from functools import partial

import torch


def _encode_per_frame(self, video: torch.Tensor) -> torch.Tensor:
    """(B, C, T, H, W) -> (B, T, D)."""
    B, C, T, H, W = video.shape
    frames = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    out = self.model.encoder(pixel_values=frames, interpolate_pos_encoding=True)
    cls = out.last_hidden_state[:, 0]
    proj = self.model.projector(cls)
    return proj.view(B, T, -1)


def _resolve_unconditional(cfg) -> bool:
    """Read the unconditional flag from either `cfg.predictor.unconditional`
    (training config layout) or `cfg.predictor_unconditional` (legacy flat).
    """
    pred_cfg = getattr(cfg, "predictor", None)
    if pred_cfg is not None and hasattr(pred_cfg, "unconditional"):
        return bool(pred_cfg.unconditional)
    return bool(getattr(cfg, "predictor_unconditional", False))


def tjepa_forward(self, batch, stage, cfg=None):
    """Temporal-AR JEPA forward.

    Args:
        self: the `spt.Module` (gives access to `model.encoder`,
            `model.projector`, `model.predictor`, `model.cond_embedder`, and
            `sigreg`).
        batch: dict with keys `video`, `label`, `alpha`, `zeta` from
            `ActiveMatterVideoDataset`.
        stage: 'fit' | 'validate'.
        cfg: dict-like with `wm.history_size`, `wm.num_preds`, `wm.nsteps`,
            `loss.sigreg.weight`, and `predictor.unconditional`.

    Invariant:
        ``num_frames == history_size + num_preds + (nsteps - 1)``. With the
        default ``nsteps == 1`` this collapses to LeWM's
        ``T = ctx_len + n_preds`` so the predictor output length and the
        target window length agree exactly. Each additional multistep
        iteration needs exactly one more future frame so the rolled target
        window stays in-bounds. We assert at runtime to fail loud on
        misconfiguration.
    """
    if cfg is None:
        raise ValueError("tjepa_forward requires `cfg` (use functools.partial).")

    video = batch["video"]                                     # (B, C, T, H, W)
    B = video.size(0)
    emb = _encode_per_frame(self, video)                       # (B, T, D)
    T = emb.size(1)

    ctx_len = int(cfg.wm.history_size)
    n_preds = int(cfg.wm.num_preds)
    nsteps = int(getattr(cfg.wm, "nsteps", 1))
    lambd = float(cfg.loss.sigreg.weight)

    if ctx_len <= 0 or n_preds <= 0 or nsteps <= 0:
        raise ValueError(
            f"history_size, num_preds, and nsteps must be positive; got "
            f"history_size={ctx_len}, num_preds={n_preds}, nsteps={nsteps}"
        )
    expected_T = ctx_len + n_preds + (nsteps - 1)
    if expected_T != T:
        raise ValueError(
            f"LeWM invariant violated: history_size + num_preds + (nsteps - 1) "
            f"= {expected_T} must equal video length T = {T}. "
            f"Adjust cfg.data.num_frames or cfg.wm.{{history_size,num_preds,nsteps}}."
        )

    if _resolve_unconditional(cfg):
        cond = None
    else:
        scalars = torch.stack(
            [batch["alpha"].float(), batch["zeta"].float()], dim=-1
        )                                                       # (B, 2)
        cond = self.model.cond_embedder(scalars.unsqueeze(1).expand(B, T, 2))  # (B, T, D)

    ctx_emb = emb[:, :ctx_len]
    ctx_cond = None if cond is None else cond[:, :ctx_len]

    if nsteps <= 1:
        # Single-step LeWM: predict (B, ctx_len, D), compare to emb shifted by n_preds.
        pred_emb = self.model.predictor(ctx_emb, ctx_cond)
        tgt_emb = emb[:, n_preds : n_preds + ctx_len].detach()
        # Lengths now match by invariant; assert in case the predictor changed shape.
        assert pred_emb.shape == tgt_emb.shape, (pred_emb.shape, tgt_emb.shape)
        pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    else:
        # Sliding-window AR rollout: at step k the target is shifted by (n_preds + k).
        # Each iter compares the predictor's output to GT future embeddings starting
        # from frame (n_preds + k); the worst-case last step targets emb[:, n_preds+nsteps-1:].
        losses = []
        cur = ctx_emb
        cur_cond = ctx_cond
        for step in range(nsteps):
            pred_emb = self.model.predictor(cur, cur_cond)
            tgt_window = emb[:, n_preds + step : n_preds + step + ctx_len].detach()
            assert pred_emb.shape == tgt_window.shape, (pred_emb.shape, tgt_window.shape, step)
            losses.append((pred_emb - tgt_window).pow(2).mean())
            # Refeed: drop oldest history step, append the model's last prediction.
            cur = torch.cat([cur[:, 1:], pred_emb[:, -1:]], dim=1)
            if cur_cond is not None:
                cur_cond = torch.cat([cur_cond[:, 1:], cur_cond[:, -1:]], dim=1)
        pred_loss = torch.stack(losses).mean()

    sigreg_loss = self.sigreg(emb.transpose(0, 1))             # (T, B, D)
    loss = pred_loss + lambd * sigreg_loss

    embedding = emb.mean(dim=1).detach()

    self.log(f"{stage}/pred_loss", pred_loss, on_step=True, on_epoch=True, sync_dist=True)
    self.log(f"{stage}/sigreg_loss", sigreg_loss, on_step=True, on_epoch=True, sync_dist=True)
    self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, sync_dist=True)

    return {
        "loss": loss,
        "embedding": embedding,
        "label": batch["label"].long(),
        "alpha": batch["alpha"].float(),
        "zeta": batch["zeta"].float(),
    }


def build_tjepa_module(*, encoder, projector, cond_embedder, predictor, sigreg, cfg, optim):
    """Bundle the four learnable modules into a single `spt.Module`.

    All four submodules are registered exclusively under `self.model.<name>`
    via the `ModuleDict` bundle so the optimizer scope `modules: "model"`
    covers them without duplicate registration via `**kwargs`. `tjepa_forward`
    accesses them as `self.model.encoder`, `self.model.projector`, etc.
    """
    import stable_pretraining as spt
    from torch import nn

    bundle = nn.ModuleDict(
        {
            "encoder": encoder,
            "projector": projector,
            "cond_embedder": cond_embedder,
            "predictor": predictor,
        }
    )
    return spt.Module(
        model=bundle,
        sigreg=sigreg,
        forward=partial(tjepa_forward, cfg=cfg),
        optim=optim,
    )
