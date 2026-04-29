# fluid-representations

Self-supervised representation learning for fluid / active-matter physics
simulations from [The Well](https://github.com/PolymathicAI/the_well).

The first feature lands a **temporal-autoregressive JEPA** (`tjepa`) on the
Active Matter dataset, built on top of
[`stable-pretraining`](https://github.com/galileo-lab/stable-pretraining), with
`OnlineKNN` / `OnlineProbe` plus a custom `RegressionProbe` tracking the
quality of `(alpha, zeta)` recovery during pretraining. The design is anchored
in the `le-wm` and `eb_jepa` reference codebases; full design notes live in
[`doc/`](doc/).

## Quick start

```bash
pip install -e ../stable-pretraining   # if not already installed
pip install -e .

export THE_WELL_DATA_DIR=/path/to/the_well_data/datasets

# one-time per-channel z-score cache
python -m physics_ssl.data --compute-stats --split train

# smoke test
python train.py trainer.max_epochs=1 trainer.limit_train_batches=2 trainer.limit_val_batches=1

# real run
python train.py
```

## Layout

```
fluid-representations/
├── doc/                                    # design notes
│   ├── 01_vjepa_masked_block_plan.md       # superseded — masked-block V-JEPA
│   └── 02_temporal_ar_jepa_plan.md         # active plan
├── configs/
│   └── tjepa_active_matter.yaml            # Hydra entry config
├── physics_ssl/
│   ├── data.py                             # ActiveMatterVideoDataset (HDF5)
│   ├── transforms.py                       # ChannelZScore, AddGaussianNoise
│   ├── model.py                            # SIGReg, ARPredictor, MLP, Embedder
│   ├── encoder.py                          # 11-channel ViT-tiny via vit_hf
│   ├── forward.py                          # tjepa_forward + multistep unroll
│   └── callbacks.py                        # RegressionProbe (continuous α/ζ)
└── train.py                                # Lightning + Hydra entry point
```

See [`doc/02_temporal_ar_jepa_plan.md`](doc/02_temporal_ar_jepa_plan.md) for
the full design and reference citations.
