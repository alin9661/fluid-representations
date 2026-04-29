"""tjepa_forward — temporal-AR JEPA forward.

Adapted from `le-wm/train.py:18-46` (`lejepa_forward`). Differences:

* Input is `(B, C=11, T, H, W)` physics video, not `(B, T, C, H, W)` pixels.
* Conditioning vector is constant per video — `[alpha, zeta]` broadcast across
  T — not a per-step action sequence.
* Surfaces a pooled `embedding` and continuous `alpha`, `zeta` scalars in the
  output dict so probe callbacks (`OnlineKNN`, `OnlineProbe`,
  `RegressionProbe`) can read them via `get_data_from_batch_or_outputs`.

The optional EB-JEPA-style multistep parallel unroll (`eb_jepa/jepa.py:142-157`)
is gated by `cfg.wm.nsteps > 1`; default is single-step (`nsteps=1`).
"""

from functools import partial

import torch
import torch.nn.functional as F


def _encode_per_frame(self, video: torch.Tensor) -> torch.Tensor:
    """(B, C, T, H, W) -> (B, T, D)."""
    B, C, T, H, W = video.shape
    frames = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    out = self.encoder(pixel_values=frames, interpolate_pos_encoding=True)
    cls = out.last_hidden_state[:, 0]
    proj = self.projector(cls)
    return proj.view(B, T, -1)


def tjepa_forward(self, batch, stage, cfg=None):
    """Temporal-AR JEPA forward.

    Args:
        self: the `spt.Module` (gives access to encoder, projector, predictor,
            cond_embedder, sigreg).
        batch: dict with keys `video`, `label`, `alpha`, `zeta` from
            `ActiveMatterVideoDataset`.
        stage: 'fit' | 'validate'.
        cfg: a small dict-like with `wm.history_size`, `wm.num_preds`,
            `wm.nsteps`, `loss.sigreg.weight`, and an optional
            `predictor_unconditional` flag.
    """
    if cfg is None:
        raise ValueError("tjepa_forward requires `cfg` (use functools.partial).")

    video = batch["video"]                                     # (B, C, T, H, W)
    B = video.size(0)
    emb = _encode_per_frame(self, video)                       # (B, T, D)
    T = emb.size(1)

    # Per-video conditioning [alpha, zeta] broadcast across T frames.
    if getattr(cfg, "predictor_unconditional", False):
        cond = None
    else:
        scalars = torch.stack(
            [batch["alpha"].float(), batch["zeta"].float()], dim=-1
        )                                                       # (B, 2)
        cond = self.cond_embedder(scalars.unsqueeze(1).expand(B, T, 2))  # (B, T, D)

    ctx_len = int(cfg.wm.history_size)
    n_preds = int(cfg.wm.num_preds)
    nsteps = int(getattr(cfg.wm, "nsteps", 1))
    lambd = float(cfg.loss.sigreg.weight)

    # Single-step (LeWM) path: predict frames [n_preds:] from history [:ctx_len].
    ctx_emb = emb[:, :ctx_len]
    ctx_cond = None if cond is None else cond[:, :ctx_len]
    tgt_emb = emb[:, n_preds:].detach()

    if nsteps <= 1:
        pred_emb = self.predictor(ctx_emb, ctx_cond)
        # Align lengths: predictor returns (B, ctx_len, D); target is (B, T-n_preds, D).
        # LeWM convention: shift comparison so we predict the next-step embedding.
        L = min(pred_emb.size(1), tgt_emb.size(1))
        pred_loss = (pred_emb[:, :L] - tgt_emb[:, :L]).pow(2).mean()
    else:
        # EB-JEPA parallel unroll: refeed GT history each step, average loss.
        pred_loss = 0.0
        cur = ctx_emb
        cur_cond = ctx_cond
        for _ in range(nsteps):
            pred_emb = self.predictor(cur, cur_cond)
            # Align lengths to target window for this iter.
            L = min(pred_emb.size(1), tgt_emb.size(1))
            pred_loss = pred_loss + (pred_emb[:, :L] - tgt_emb[:, :L]).pow(2).mean() / nsteps
            # Refeed: drop oldest history step, append the model's last prediction.
            cur = torch.cat([cur[:, 1:], pred_emb[:, -1:]], dim=1)
            if cur_cond is not None:
                cur_cond = torch.cat([cur_cond[:, 1:], cur_cond[:, -1:]], dim=1)

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
    """Convenience constructor that bundles modules into a `stable_pretraining.Module`."""
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
        encoder=encoder,
        projector=projector,
        cond_embedder=cond_embedder,
        predictor=predictor,
        sigreg=sigreg,
        forward=partial(tjepa_forward, cfg=cfg),
        optim=optim,
    )
