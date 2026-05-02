"""Evaluate the best T-JEPA Active Matter checkpoint on the held-out test split.

Usage:
    export THE_WELL_DATA_DIR=/path/to/the_well/datasets
    python test.py                                              # auto-discover best ckpt
    python test.py test.checkpoint_path=checkpoints/<run>/best/last.ckpt
"""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
import torchmetrics
from omegaconf import OmegaConf

from physics_ssl.callbacks import RegressionProbe
from physics_ssl.data import ActiveMatterVideoDataset
from physics_ssl.encoder import build_per_frame_encoder, encoder_hidden_size
from physics_ssl.forward import build_tjepa_module
from physics_ssl.model import ARPredictor, Embedder, MLP, SIGReg
from physics_ssl.transforms import ChannelZScore


def _resolve_checkpoint(cfg) -> Path:
    """Pick `cfg.test.checkpoint_path` if given, else the freshest .ckpt under best/."""
    if cfg.test.checkpoint_path:
        ckpt = Path(hydra.utils.to_absolute_path(cfg.test.checkpoint_path))
        if not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        return ckpt
    best_dir = Path(hydra.utils.to_absolute_path(cfg.checkpoint.dirpath)) / "best"
    candidates = list(best_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(
            f"no .ckpt files under {best_dir}; pass test.checkpoint_path=... explicitly"
        )
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


@hydra.main(version_base=None, config_path="configs", config_name="tjepa_active_matter")
def run(cfg):
    pl.seed_everything(cfg.seed, workers=True)

    # ---------------- Data ----------------
    transform = ChannelZScore(stats_path=cfg.data.stats_path) if Path(cfg.data.stats_path).exists() else None
    if transform is None:
        print(f"[warn] stats file {cfg.data.stats_path} missing; running without channel z-score")

    test_set = ActiveMatterVideoDataset(
        root=cfg.data.root,
        split="test",
        num_frames=cfg.data.num_frames,
        transform=transform,
        bins_per_param=cfg.data.bins_per_param,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, drop_last=False, **cfg.loader,
    )
    data_module = spt.data.DataModule(test=test_loader)

    # ---------------- Modules (must mirror train.py exactly) ----------------
    encoder = build_per_frame_encoder(
        in_chans=cfg.in_chans,
        image_size=cfg.img_size,
        patch_size=cfg.patch_size,
        scale=cfg.encoder_scale,
    )
    hidden_dim = encoder_hidden_size(encoder)
    embed_dim = int(cfg.wm.embed_dim)

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )
    cond_embedder = Embedder(input_dim=2, smoothed_dim=10, emb_dim=embed_dim)
    predictor = ARPredictor(
        num_frames=cfg.data.num_frames,
        input_dim=embed_dim,
        hidden_dim=embed_dim,
        output_dim=embed_dim,
        depth=cfg.predictor.depth,
        heads=cfg.predictor.heads,
        mlp_dim=cfg.predictor.mlp_dim,
        dim_head=cfg.predictor.dim_head,
        dropout=cfg.predictor.dropout,
        emb_dropout=cfg.predictor.emb_dropout,
        unconditional=cfg.predictor.unconditional,
    )
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs)

    optim = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": cfg.scheduler.type},
            "interval": "epoch",
        },
    }

    module = build_tjepa_module(
        encoder=encoder,
        projector=projector,
        cond_embedder=cond_embedder,
        predictor=predictor,
        sigreg=sigreg,
        cfg=cfg,
        optim=optim,
    )

    # ---------------- Probes (same set as train.py so state_dict keys match) ----------------
    callbacks = []
    n_classes = int(cfg.data.bins_per_param) ** 2

    if cfg.probes.knn.enabled:
        from stable_pretraining.callbacks import OnlineKNN
        callbacks.append(OnlineKNN(
            module=module,
            name="knn_probe",
            input="embedding",
            target="label",
            queue_length=cfg.probes.knn.queue_length,
            k=cfg.probes.knn.k,
            metrics={"acc": torchmetrics.classification.MulticlassAccuracy(num_classes=n_classes)},
        ))
    if cfg.probes.linear.enabled:
        from stable_pretraining.callbacks import OnlineProbe
        callbacks.append(OnlineProbe(
            module=module,
            name="linear_probe",
            input="embedding",
            target="label",
            probe=torch.nn.Linear(embed_dim, n_classes),
            loss=torch.nn.CrossEntropyLoss(),
            metrics={"acc": torchmetrics.classification.MulticlassAccuracy(num_classes=n_classes)},
        ))
    if cfg.probes.regression.enabled:
        callbacks.append(RegressionProbe(
            module=module,
            name="regprobe",
            input="embedding",
            targets=tuple(cfg.probes.regression.targets),
            in_features=embed_dim,
        ))

    # ---------------- Logger ----------------
    logger = None
    if cfg.wandb.enabled:
        try:
            from lightning.pytorch.loggers import WandbLogger
        except ImportError as e:
            raise ImportError(
                "wandb is an optional dependency. Install with `pip install wandb` "
                "or set `wandb.enabled=false`."
            ) from e
        wandb_cfg = OmegaConf.to_container(cfg.wandb.config, resolve=True)
        if cfg.test.wandb_run_suffix and wandb_cfg.get("name"):
            wandb_cfg["name"] = f"{wandb_cfg['name']}{cfg.test.wandb_run_suffix}"
        logger = WandbLogger(**wandb_cfg)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    # ---------------- Checkpoint resolution ----------------
    ckpt_path = _resolve_checkpoint(cfg)
    print(f"[test] loading checkpoint: {ckpt_path}")

    # ---------------- Trainer (test-only — strip fit-only kwargs) ----------------
    trainer_cfg = dict(cfg.trainer)
    trainer_cfg.pop("max_epochs", None)
    trainer_cfg.pop("limit_train_batches", None)
    trainer_cfg.pop("limit_val_batches", None)

    trainer = pl.Trainer(
        **trainer_cfg,
        callbacks=callbacks,
        logger=logger,
    )
    trainer.test(module, datamodule=data_module, ckpt_path=str(ckpt_path))


if __name__ == "__main__":
    run()
