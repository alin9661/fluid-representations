"""Train temporal-AR JEPA on Active Matter via stable-pretraining.

Usage:
    export THE_WELL_DATA_DIR=/path/to/the_well/datasets
    python -m physics_ssl.data --compute-stats --split train  # one-time
    python train.py                                           # full run
    python train.py trainer.max_epochs=1 trainer.limit_train_batches=2  # smoke
"""

from __future__ import annotations

import os
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


@hydra.main(version_base=None, config_path="configs", config_name="tjepa_active_matter")
def run(cfg):
    pl.seed_everything(cfg.seed, workers=True)

    # ---------------- Data ----------------
    transform = ChannelZScore(stats_path=cfg.data.stats_path) if Path(cfg.data.stats_path).exists() else None
    if transform is None:
        print(f"[warn] stats file {cfg.data.stats_path} missing; running without channel z-score")

    train_set = ActiveMatterVideoDataset(
        root=cfg.data.root,
        split="train",
        num_frames=cfg.data.num_frames,
        transform=transform,
        bins_per_param=cfg.data.bins_per_param,
    )
    val_set = ActiveMatterVideoDataset(
        root=cfg.data.root,
        split="valid",
        num_frames=cfg.data.num_frames,
        transform=transform,
        bins_per_param=cfg.data.bins_per_param,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, shuffle=True, drop_last=True, **cfg.loader,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, shuffle=False, drop_last=False, **cfg.loader,
    )
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)

    # ---------------- Modules ----------------
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

    # ---------------- Probes ----------------
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

    # ---------------- Trainer ----------------
    logger = None
    if cfg.wandb.enabled:
        try:
            from lightning.pytorch.loggers import WandbLogger
        except ImportError as e:
            raise ImportError(
                "wandb is an optional dependency. Install with `pip install wandb` "
                "or set `wandb.enabled=false`."
            ) from e
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
    )
    manager = spt.Manager(trainer=trainer, module=module, data=data_module)
    manager()


if __name__ == "__main__":
    run()
