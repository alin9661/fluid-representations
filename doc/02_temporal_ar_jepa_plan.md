# Plan: Temporal-autoregressive JEPA on Active Matter using `stable-pretraining` (LeWM / EB-JEPA pattern)

> **Status:** Active. Supersedes `01_vjepa_masked_block_plan.md`. Anchored in the `le-wm/` and `eb_jepa/` reference codebases.

## Context

The user wants to pretrain a representation model on the **Active Matter** subset of "The Well" using a JEPA-family objective and validate it with `stable-pretraining`'s `OnlineKNN` / `OnlineProbe` plus a custom regression probe over the continuous physical parameters `alpha` and `zeta`. The user named the method "V-JEPA" but asked that the implementation be anchored in two reference codebases — `le-wm` and `eb_jepa` — for correctness. Both references implement a **temporal-autoregressive JEPA**, not masked-block V-JEPA:

| Reference | Predicts | Encoder | Predictor | Anti-collapse | EMA target? |
|---|---|---|---|---|---|
| **LeWM** (`le-wm/`, on top of `stable-pretraining`) | next-frame embedding from history of 3 | per-frame 2D ViT (`vit_hf`, tiny, patch=14) | `ARPredictor` (transformer w/ AdaLN-zero conditioning, depth 6, dim 192) | `SIGReg(emb, weight=0.09)` | ❌ — single shared encoder |
| **EB-JEPA** (`eb_jepa/`) | next state from history (parallel + autoregressive unrolls) | per-frame `ResNet5` / `ImpalaEncoder` | `ResUNet` (CNN) or `RNNPredictor` (GRU) | `VC_IDM_Sim_Regularizer` (variance + covariance + temporal sim + IDM) | ❌ |
| **Original V-JEPA paper** | masked spatiotemporal blocks | Video ViT (3D tubelet) | small predictor with mask tokens | none (uses EMA) | ✅ |

**Implication.** Following the reference codebases — and matching `remy9926/physical-representation-learning`'s existing JEPA + `SigReg` design — the right method here is **temporal-autoregressive JEPA with SIGReg**, *not* the masked-block V-JEPA from the prior plan revision. This is also strictly easier to wire into `stable-pretraining`: LeWM (`le-wm/jepa.py`, `le-wm/module.py`, `le-wm/train.py:18-46`) is *already* built as a `stable_pretraining.Module` with `OnlineKNN`/`OnlineProbe` callbacks, so the work collapses to **adapting LeWM's pipeline from 2D RGB pixels to (C=11, T, H, W) physics volumes**.

The dataset is naturally `(C=11, T, H, W)`; per-frame encoding (a single 2D ViT applied to each timestep) sidesteps custom 3D tubelet embedding and a masked encoder entirely.

---

## Approach (recommended, anchored in `le-wm`)

**Pretraining objective.** Per-frame 2D ViT encodes each of `T` frames into a sequence of frame embeddings `(B, T, D)`. An autoregressive transformer predictor consumes the history and predicts future-frame embeddings. Loss = MSE between predicted future embeddings and the encoder's own future embeddings (detached) + SIGReg anti-collapse regularizer on the full embedding sequence. Optionally use **EB-JEPA-style parallel multistep unroll** (`eb_jepa/jepa.py:142-157`) so the predictor sees its own outputs during training (exposure-bias mitigation).

**Inputs.** `(C=11, T=8, H=256, W=256)` per sample. Treat the time axis as the sequence axis: `T=8` frame embeddings per video.

**Encoder.** `spt.backbone.utils.vit_hf("tiny", patch_size=16, image_size=256, pretrained=False, use_mask_token=False)` with **`in_chans=11`** (HF ViT supports `num_channels` via the config; if it doesn't accept 11 directly, replace `embeddings.patch_embeddings.projection` with `nn.Conv2d(11, hidden, 16, 16)` after init — same swap LeWM does *not* need on RGB but is one line for us). CLS-token output → `MLP(hidden→2048→192, BatchNorm1d)` projector, exactly as `le-wm/train.py:104-109`. Embedding dim `D=192`.

```python
# le-wm/train.py:82-88
encoder = spt.backbone.utils.vit_hf(
    cfg.encoder_scale,            # "tiny"
    patch_size=cfg.patch_size,    # 14 (we'll use 16 for 256 // 16 = 16)
    image_size=cfg.img_size,
    pretrained=False,
    use_mask_token=False,
)
```

```python
# le-wm/train.py:104-109 — keep verbatim
projector = MLP(input_dim=hidden_dim, output_dim=embed_dim,  # 192
                hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
```

**Predictor.** Reuse `le-wm/module.py:244-286` `ARPredictor` (depth 6, dim 192, learnable temporal positional embeddings, AdaLN-zero conditioning). Active Matter has no per-timestep actions; we replace the action stream with a **constant per-video conditioning vector** `c = [alpha, zeta]` broadcast across time and embedded by `Embedder` (`le-wm/module.py:189-214`). This gives the predictor parameter-conditional dynamics for free, matching the physics: `alpha`, `zeta` set the active-matter regime.

If we want to ablate conditioning, also expose a `predictor_no_cond=True` flag that drops AdaLN-zero and runs vanilla self-attention transformer blocks — clean apples-to-apples to a non-conditional baseline.

**Anti-collapse regularizer.** `SIGReg` from `le-wm/module.py:10-36` with weight `0.09` (LeWM default at `le-wm/config/train/lewm.yaml:61`). This matches `remy9926/physical-representation-learning`'s `SigReg` loss family — keeping methodological continuity.

**No EMA target encoder.** Both references omit it; we follow.

**Multistep unroll (optional, EB-JEPA-style).** Switch on with `nsteps=4`. From `eb_jepa/jepa.py:142-157`:

```python
if unroll_mode == "parallel":
    predicted_states = state
    for _ in range(nsteps):
        predicted_states = self.predictor(predicted_states, actions_encoded)[:, :, :-1]
        predicted_states = torch.cat(
            (state[:, :, :context_length], predicted_states), dim=2
        )
        if compute_loss:
            ploss += self.predcost(state, predicted_states) / nsteps
```

We adopt the same averaged-multistep loss; default `nsteps=1` (single-step) for the first run, raise to 4 once stable.

**Loss.** `loss = pred_loss + 0.09 * sigreg_loss`, where `pred_loss = ((pred - emb.detach()) ** 2).mean()` (LeWM convention) and SIGReg is `module.SIGReg(emb.transpose(0, 1))`. Targets are detached (LeWM does this implicitly at the loss; we make it explicit since `remy9926` did so explicitly and it removes ambiguity).

**Probes.** During each forward, surface a pooled embedding `embedding = emb.mean(dim=1)` (mean over `T`) plus continuous `alpha`/`zeta` and a 16-bin discretized `label`. Then attach:
- `OnlineKNN` (queue 20000, k=10, on `label`)
- `OnlineProbe` linear (`Linear(192, 16)`, on `label`)
- Custom `RegressionProbe` (`Linear(192, 2) + MSE`, on `[alpha, zeta]`, logs `val/regprobe_{mse_alpha,mse_zeta,r2}`)

**Optimizer / schedule.** LeWM defaults: AdamW lr `5e-5`, weight_decay `1e-3`, `gradient_clip_val=1.0`, `LinearWarmupCosineAnnealingLR` (10% warmup). EB-JEPA uses separate clips for encoder vs. predictor (`grad_clip_enc=2.0`, `grad_clip_pred=2.0`, `eb_jepa/main.py:334-341`); we keep one global clip first and split only if instability shows up.

---

## Files to create (all new — no edits to upstream `stable-pretraining`, `le-wm`, `eb_jepa`, or `remy9926`)

```
stable-pretraining-physics/
├── configs/
│   └── tjepa_active_matter.yaml      # Hydra entry point
├── physics_ssl/
│   ├── __init__.py
│   ├── data.py                        # ActiveMatterVideoDataset (returns volumes)
│   ├── transforms.py                  # ChannelZScore, AddGaussianNoise
│   ├── model.py                       # ARPredictor wrapper, SIGReg, MLP — copied from le-wm
│   ├── encoder.py                     # build_per_frame_encoder() (vit_hf + 11-channel patch)
│   ├── forward.py                     # tjepa_forward (= adapted lejepa_forward)
│   └── callbacks.py                   # RegressionProbe
└── train.py                           # Optional Python entry point
```

### 1. `physics_ssl/data.py` — `ActiveMatterVideoDataset`

Adapted from `WellDatasetForJEPA` (`remy9926/physical-representation-learning/physics_jepa/data.py:17-241`). Reuse — by copy — the HDF5 shard discovery + `t0_fields`/`t1_fields`/`t2_fields` parsing (lines 117-150) and `alpha`/`zeta` extraction (line 113). Output:

```python
def __getitem__(self, idx):
    volume = self._load_window(idx)     # (C=11, T=8, H, W) float32
    volume = self.transform(volume)     # channel z-score
    a, z = self._scalar_params(idx)
    return {
        "video": volume,                 # (C, T, H, W)
        "label": self._bin(a, z),        # 16-bin int (4×4 grid in α-ζ space)
        "alpha": float(a),
        "zeta":  float(z),
    }
```

Path: `${THE_WELL_DATA_DIR}/active_matter/data/{train,valid}/`. No spatial flips/rotations.

### 2. `physics_ssl/transforms.py`

`ChannelZScore(mean, std)` (`(11,)`-shaped stats from a one-time `.npz` cache) and `AddGaussianNoise(std)` (mirrors `physics_jepa/data.py:233-278`). `python -m physics_ssl.data --compute-stats --split train` writes the cache.

### 3. `physics_ssl/model.py` — `ARPredictor`, `SIGReg`, `MLP`, `Embedder`

Direct copy from `le-wm/module.py` (lines 10-36 for `SIGReg`, 88-111 for `ConditionalBlock` w/ AdaLN-zero, 189-214 for `Embedder`, 244-286 for `ARPredictor`). Two changes:

- `Embedder.__init__` accepts `input_dim=2` (for `[alpha, zeta]`) instead of action_dim.
- `ARPredictor.forward(x, c=None)` short-circuits AdaLN modulation when `c is None` — gives us the conditioning ablation toggle.

### 4. `physics_ssl/encoder.py` — per-frame 2D ViT for 11 channels

```python
def build_per_frame_encoder(in_chans=11, image_size=256, patch_size=16, scale="tiny"):
    enc = spt.backbone.utils.vit_hf(
        scale, patch_size=patch_size, image_size=image_size,
        pretrained=False, use_mask_token=False,
    )
    # HF ViT: replace the patch projection with one accepting 11 channels.
    proj = enc.embeddings.patch_embeddings.projection
    new_proj = nn.Conv2d(in_chans, proj.out_channels,
                         kernel_size=proj.kernel_size,
                         stride=proj.stride)
    enc.embeddings.patch_embeddings.projection = new_proj
    enc.config.num_channels = in_chans
    return enc
```

### 5. `physics_ssl/forward.py` — `tjepa_forward` (the adapted `lejepa_forward`)

Derived line-for-line from `le-wm/train.py:18-46`, with `(B, T, C, H, W)` reshape glue and SIGReg unchanged:

```python
def tjepa_forward(self, batch, stage, cfg):
    video = batch["video"]                                # (B, C=11, T, H, W)
    B, C, T, H, W = video.shape
    frames = video.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)

    # Per-frame encode
    hf = self.encoder(pixel_values=frames).last_hidden_state[:, 0]   # (B*T, hidden)
    proj = self.projector(hf).view(B, T, -1)                          # (B, T, D=192)

    # Conditioning vector: broadcast [alpha, zeta] across time
    c_scalar = torch.stack([batch["alpha"], batch["zeta"]], dim=-1)   # (B, 2)
    c = c_scalar.unsqueeze(1).expand(B, T, 2)                         # (B, T, 2)
    c = self.cond_embedder(c)                                         # (B, T, D)

    ctx_len = cfg.wm.history_size                                     # default 3
    n_preds = cfg.wm.num_preds                                        # default 1
    ctx_emb, ctx_act = proj[:, :ctx_len], c[:, :ctx_len]
    tgt_emb = proj[:, n_preds:].detach()
    pred_emb = self.predictor(ctx_emb, ctx_act)                       # (B, T-1, D) shape per LeWM

    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    sigreg_loss = self.sigreg(proj.transpose(0, 1))                   # SIGReg expects (T, B, D)
    loss = pred_loss + cfg.lam_sigreg * sigreg_loss                   # 0.09 default

    # Pooled eval embedding for probes
    embedding = proj.mean(dim=1).detach()

    self.log(f"{stage}/pred_loss",   pred_loss,   on_step=True, on_epoch=True, sync_dist=True)
    self.log(f"{stage}/sigreg_loss", sigreg_loss, on_step=True, on_epoch=True, sync_dist=True)
    self.log(f"{stage}/loss",        loss,        on_step=True, on_epoch=True, sync_dist=True)

    return {
        "loss": loss, "embedding": embedding,
        "label": batch["label"].long(),
        "alpha": batch["alpha"].float(),
        "zeta":  batch["zeta"].float(),
    }
```

A `multistep=True` flag wraps the prediction in EB-JEPA's parallel-unroll loop (`eb_jepa/jepa.py:142-157`); off by default.

### 6. `physics_ssl/callbacks.py` — `RegressionProbe`

Copy `stable_pretraining/callbacks/probe.py`'s skeleton; swap probe head to `nn.Linear(192, 2)`, swap loss to `nn.MSELoss`. Read `batch["alpha"]`, `batch["zeta"]`, stack as the regression target. Log `val/regprobe_mse_alpha`, `val/regprobe_mse_zeta`, `val/regprobe_r2` (compute R² against batch variance of the targets). Pattern to mirror exactly: lines that drive `OnlineProbe`'s `train_batch_end` / `val_batch_end` hooks — the only deltas are the head dim, the loss, and the target tensor.

### 7. `configs/tjepa_active_matter.yaml`

```yaml
defaults: [_self_]

cfg:
  wm: { history_size: 3, num_preds: 1 }
  lam_sigreg: 0.09

data:
  train:
    dataset:
      _target_: physics_ssl.data.ActiveMatterVideoDataset
      root: ${oc.env:THE_WELL_DATA_DIR}/active_matter/data/train
      num_frames: 8
      transform:
        _target_: physics_ssl.transforms.ChannelZScore
        stats_path: cache/active_matter_stats.npz
    batch_size: 16
    num_workers: 8
  val: { _target_: ..., split: valid, batch_size: 16 }

module:
  encoder:
    _target_: physics_ssl.encoder.build_per_frame_encoder
    in_chans: 11
    image_size: 256
    patch_size: 16
    scale: tiny
  projector:
    _target_: physics_ssl.model.MLP
    input_dim: 192        # ViT-tiny hidden
    output_dim: 192
    hidden_dim: 2048
    norm_fn: { _target_: torch.nn.BatchNorm1d, _partial_: true }
  cond_embedder:
    _target_: physics_ssl.model.Embedder
    input_dim: 2          # [alpha, zeta]
    output_dim: 192
  predictor:
    _target_: physics_ssl.model.ARPredictor
    input_dim: 192
    depth: 6
    num_heads: 6
    num_frames: 8
  sigreg:
    _target_: physics_ssl.model.SIGReg
  forward: physics_ssl.forward.tjepa_forward

callbacks:
  - _target_: stable_pretraining.callbacks.OnlineKNN
    name: knn_probe
    input: embedding
    target: label
    queue_length: 20000
    k: 10
    metrics: [{ _target_: torchmetrics.classification.MulticlassAccuracy, num_classes: 16 }]
  - _target_: stable_pretraining.callbacks.OnlineProbe
    name: linear_probe
    input: embedding
    target: label
    probe: { _target_: torch.nn.Linear, in_features: 192, out_features: 16 }
    metrics: [{ _target_: torchmetrics.classification.MulticlassAccuracy, num_classes: 16 }]
  - _target_: physics_ssl.callbacks.RegressionProbe
    input: embedding
    targets: [alpha, zeta]
    in_features: 192

optimizer:
  type: AdamW
  lr: 5e-5
  weight_decay: 1e-3
scheduler:
  _target_: stable_pretraining.scheduler.LinearWarmupCosineAnnealingLR
  warmup_ratio: 0.1
trainer:
  max_epochs: 200
  precision: bf16-mixed
  accelerator: gpu
  devices: 1
  gradient_clip_val: 1.0
```

The structure parrots `le-wm/config/train/lewm.yaml` (LR `5e-5`, wd `1e-3`, gradient clip `1.0`, history `3`, predict `1`, SIGReg weight `0.09`) for hyperparameter continuity.

---

## Critical reused functions / files

| What to reuse | Where it lives |
|---|---|
| HDF5 discovery, `t0/t1/t2_fields` parsing, `alpha`/`zeta` extraction, GPU noise pattern | `remy9926/physical-representation-learning/physics_jepa/data.py:101-278` |
| **`ARPredictor`** (depth 6, dim 192, AdaLN-zero conditioning, learnable temporal pos-embed) | `le-wm/module.py:244-286` |
| **`ConditionalBlock`** with AdaLN-zero modulation | `le-wm/module.py:88-111` |
| **`Embedder`** for conditioning vector (`Conv1d` smoothing + MLP) | `le-wm/module.py:189-214` |
| **`SIGReg`** anti-collapse regularizer (`λ=0.09`) | `le-wm/module.py:10-36` |
| **`lejepa_forward`** (template for `tjepa_forward`) | `le-wm/train.py:18-46` |
| `MLP` projector (Linear→BatchNorm1d→Linear→…) | `le-wm/train.py:104-109` |
| Hyperparameter defaults (lr 5e-5, wd 1e-3, clip 1.0, warmup 10%) | `le-wm/config/train/lewm.yaml`, `le-wm/train.py:130` |
| **EB-JEPA parallel multistep unroll** (optional `nsteps>1`) | `eb_jepa/jepa.py:142-157` |
| **EB-JEPA `CosineWithWarmup`** (alternative scheduler) | `eb_jepa/schedulers.py:4-38` |
| **EB-JEPA separate enc/predictor grad clipping** (optional, if instability) | `eb_jepa/examples/ac_video_jepa/main.py:334-341` |
| `vit_hf` 2D ViT loader | `stable-pretraining/stable_pretraining/backbone/utils.py` |
| `OnlineKNN`, `OnlineProbe` (template for `RegressionProbe`) | `stable-pretraining/stable_pretraining/callbacks/{knn,probe}.py` |
| `Manager` orchestrator | `stable-pretraining/stable_pretraining/manager.py` |

No upstream edits in any repo.

---

## Why this design (vs. masked-block V-JEPA from the previous draft)

- **The references actually implement this.** Both LeWM and EB-JEPA are temporal-AR JEPAs without masking or EMA. Anchoring in the references means we can copy structure (forward, predictor, regularizer, optimizer) almost verbatim.
- **LeWM is already on `stable-pretraining`** — we inherit its `OnlineKNN`/`OnlineProbe` integration without re-deriving it.
- **Methodological continuity with `remy9926`.** That project already uses VICReg + `SigReg` losses on JEPA; LeWM's SIGReg is the exact same regularizer family. Results between the two will be directly comparable.
- **No 3D tubelet, no masked encoder, no 3D mask sampler, no `TeacherStudentWrapper` plumbing.** All of those are removed from the build surface; the entire encoder chain is `vit_hf` + a one-line `Conv2d` swap to accept 11 channels.
- **Conditional-by-default** on `(alpha, zeta)`. The predictor learns parameter-conditional dynamics, which is the core scientific question for active matter. We retain a clean toggle (`predictor_no_cond=True`) for the unconditional ablation.

---

## Verification

1. **Env:** `export THE_WELL_DATA_DIR=…/the_well_data/datasets`.
2. **Stats cache:** `python -m physics_ssl.data --compute-stats --split train` → `cache/active_matter_stats.npz`.
3. **Encoder sanity:** load `build_per_frame_encoder(in_chans=11)`, push `(2, 11, 256, 256)`, assert CLS shape `(2, 192)` and projector output `(2, 192)`.
4. **Predictor sanity:** instantiate `ARPredictor(input_dim=192, depth=6, num_frames=8)`, push `(B=2, T=3, D=192)` ctx + `(2, 3, 192)` cond → assert output `(2, 3, 192)`. Also test `predictor_no_cond=True` path.
5. **SIGReg sanity:** `SIGReg()(torch.randn(8, 32, 192))` returns a non-NaN scalar; with all-equal embeddings the value is large (collapse penalty fires).
6. **1-step dry run:**
   ```bash
   spt configs/tjepa_active_matter.yaml \
       trainer.max_epochs=1 trainer.limit_train_batches=2 trainer.limit_val_batches=1
   ```
   Confirm `train/pred_loss`, `train/sigreg_loss`, `train/loss` all log; no shape mismatches.
7. **Probe sanity:** after first val epoch `val/knn_probe_acc`, `val/linear_probe_acc`, `val/regprobe_mse_alpha`, `val/regprobe_mse_zeta`, `val/regprobe_r2` all appear.
8. **Real run:** 200 epochs single-GPU. Success signal: `val/regprobe_r2` rises monotonically and ends meaningfully > 0; `val/knn_probe_acc` clears the `1/16=6.25%` random baseline by a wide margin.
9. **Comparison:** rerun the same probe protocol on a frozen `remy9926/physical-representation-learning` JEPA encoder for an apples-to-apples representation-quality number.

If `vit_hf` rejects `num_channels=11` at construction (HF AutoConfig sometimes enforces 3), the workaround is the explicit `Conv2d` swap shown in `encoder.py` — already the default path, no extra branching needed.

---

## Out of scope (deliberately)

- **Masked-block V-JEPA / 3D tubelet Video ViT** — defer. Neither reference uses it; revisit only if temporal-AR underperforms `remy9926` after head-to-head probing. See `01_vjepa_masked_block_plan.md` for that alternative design.
- **EMA target encoder (`TeacherStudentWrapper`)** — not used by LeWM/EB-JEPA; SIGReg replaces it.
- VICReg / SimCLR / DINO baselines — separate experiment.
- Multi-dataset training (Shear Flow, Rayleigh-Bénard) — Active Matter only.
- Refactoring `remy9926` into `stable-pretraining`.
