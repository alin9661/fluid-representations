# Plan (superseded): V-JEPA pretraining on Active Matter using `stable-pretraining`

> **Status:** Superseded by `02_temporal_ar_jepa_plan.md`. This document is preserved for reference. The masked-block V-JEPA design below was set aside in favor of a temporal-autoregressive JEPA after a closer read of the `le-wm` and `eb_jepa` reference codebases — both implement temporal-AR JEPAs without masking or EMA target encoders. This plan is still a viable path if the temporal-AR variant underperforms.

## Context

The user wants to pretrain a representation model on the **Active Matter** subset of "The Well" using **V-JEPA** (Video Joint-Embedding Predictive Architecture) with a **Video ViT** backbone, validated by `stable-pretraining`'s `OnlineKNN` / `OnlineProbe` plus a custom regression probe for the continuous physical parameters `alpha` and `zeta`. This replaces an earlier VICReg + temporal-frame-pair plan: instead of producing two augmented 2D frames and contrasting them, V-JEPA operates on the full spatiotemporal volume `(C=11, T, H, W)`, masks spatiotemporal tubelets, and predicts target embeddings from a context encoder using an EMA-target encoder — exactly the JEPA family that `remy9926` already explores, but in the framework that gives us probing + Lightning + Hydra for free.

`stable-pretraining` already has most of the V-JEPA primitives, just not the video-specific glue:

| V-JEPA component | Available in `stable-pretraining`? | Where |
|---|---|---|
| Image-JEPA reference (`IJEPA`, `IJEPAOutput`, `ijepa_forward`) | ✅ | `benchmarks/imagenet100/ijepa-vit-base.py:23-37` (forward); the `IJEPA` class is in `stable_pretraining` (used in that benchmark) |
| EMA target encoder (`TeacherStudentWrapper` + `TeacherStudentCallback`) | ✅ | `stable_pretraining/backbone/utils.py:336-493`, `callbacks/teacher_student.py:11-143` |
| Token-level masked encoder (`MaskedEncoder` for timm ViTs) | ✅ | `stable_pretraining/backbone/vit.py:229-468` |
| Predictor block (`FlexibleTransformer`, supports `sincos_3d` positional embeddings + learnable `[MASK]` token) | ✅ | `stable_pretraining/backbone/vit.py:1129-1671` (config example at 1182-1193) |
| 2D multi-block mask generator (`multi_block_mask`) | ✅ (2D only) | `stable_pretraining/data/masking.py:81-142` |
| **3D / spatiotemporal mask generator** | ❌ — extend the 2D one | (new) |
| **3D tubelet patch embedding** | ❌ — `from_timm` is 2D-only | (new) |
| **`vjepa_forward`** | ❌ | (new) |
| `OnlineKNN`, `OnlineProbe` | ✅ (classification only) | `stable_pretraining/callbacks/{knn,probe}.py` |
| Regression probing | ❌ | (new, custom callback) |

So the V-JEPA build is concentrated in **one new package** — three small new modules + a custom forward + a custom callback — with **no edits to upstream `stable-pretraining`** and **no edits to `remy9926`** (we only borrow its HDF5 reading pattern).

---

## Approach (recommended)

**Pretraining objective:** V-JEPA — smooth-L1 between **predicted** target-block embeddings (from the predictor consuming context tokens + target-block positions) and **target** embeddings produced by the EMA teacher on the same volume, gathered at the target-block token positions.

**Inputs.** Active Matter volume `(C=11, T, H, W)` per sample, normalized per channel. Use `T=8` and `H=W=256`. Tubelet patch size `(t=2, h=16, w=16)` → `4 × 16 × 16 = 1024` tokens per video.

**Backbone (Video ViT).** Take `timm.vit_small_patch16_224` and swap its 2D `patch_embed` for a custom `PatchEmbed3D` (a single `Conv3d(in=11, out=384, kernel=(2,16,16), stride=(2,16,16))`) plus 3D `sincos` positional embeddings. Everything else (attention blocks, norm) is reused untouched, which keeps it compatible with `MaskedEncoder` (since `MaskedEncoder` only requires `patch_embed`, `blocks`, `cls_token`, `reg_token`).

**Target encoder.** Wrap the encoder with `TeacherStudentWrapper(base_ema=0.994, final_ema=1.0)`. Use `forward_student(x, mask=ctx_mask)` for context and `torch.no_grad(): forward_teacher(x)` for all tokens — gather target-block tokens from the teacher output.

**Predictor.** A small `FlexibleTransformer` (depth 6, dim 384 → 192 internal, heads 6, `self_attn=True, cross_attn=False, add_mask_token=True, pos_embed="sincos_3d", grid_size=(4,16,16)`). Per target block: feed `[ctx_tokens; mask_tokens_at_target_positions]` and read predictions at the mask-token positions. This matches the IJEPA predictor config at `vit.py:1182-1193`.

**Masking.** New `multi_block_mask_3d(grid_size=(T_p, H_p, W_p), context_scale=(0.85, 1.0), target_scale=(0.15, 0.20), n_targets=4)` modeled on `multi_block_mask` in `stable_pretraining/data/masking.py:81-142`. Sample one large context block (`~85%` of tokens, contiguous in T×H×W) and 4 small target blocks; targets and context are made disjoint.

**Loss.** Smooth-L1 averaged over target tokens, then over target blocks (V-JEPA paper convention).

**Probes.** During validation, compute a pooled embedding from the **teacher** on the unmasked volume (mean over all 1024 tokens → `R^384`) and surface it as `embedding`. Then:
- `OnlineKNN` and `OnlineProbe` on a 16-bin discretization of `(alpha, zeta)` (sanity signal).
- Custom `RegressionProbe` callback predicting continuous `[alpha, zeta]` with MSE / R².

**Trainer.** Lightning, `bf16-mixed`, single-GPU first; scale to multi-GPU later via `pl.Trainer(devices=N, strategy="ddp")`.

---

## Files to create (all new — no edits to upstream)

```
stable-pretraining-physics/
├── configs/
│   └── vjepa_active_matter.yaml
├── physics_ssl/
│   ├── __init__.py
│   ├── data.py                  # ActiveMatterVolumeDataset
│   ├── transforms.py            # ChannelZScore, AddGaussianNoise
│   ├── masking.py               # multi_block_mask_3d
│   ├── models/
│   │   ├── __init__.py
│   │   ├── patch_embed_3d.py    # PatchEmbed3D + build_video_vit()
│   │   └── predictor.py         # build_vjepa_predictor() (FlexibleTransformer)
│   ├── forwards.py              # vjepa_forward
│   └── callbacks.py             # RegressionProbe, EvalEmbedding
└── train.py                     # Optional Python entry point
```

### 1. `physics_ssl/data.py` — `ActiveMatterVolumeDataset`

Adapt `WellDatasetForJEPA` (`remy9926/physical-representation-learning/physics_jepa/data.py:18-333`) but emit a **single volume**, not a view pair:

```python
def __getitem__(self, idx):
    volume = self._load_window(idx)            # (C=11, T=8, H, W), float32
    volume = self.transform(volume)             # channel z-score
    alpha, zeta = self._scalar_params(idx)
    return {
        "video": volume,                        # NOT "image"
        "label": self._bin(alpha, zeta),        # int, 16 bins
        "alpha": alpha, "zeta": zeta,           # floats for RegressionProbe
    }
```

Reuse — by copy, not by import — the HDF5 shard discovery (`physics_jepa/data.py:103-134`), field-stacking schema (`136-171`), and `alpha`/`zeta` extraction (`130-132`). Path: `${THE_WELL_DATA_DIR}/active_matter/data/{train,valid}/`. No spatial flips/rotations.

### 2. `physics_ssl/transforms.py`

`ChannelZScore(mean, std)` (`(11,)`-shaped stats loaded from a one-time `.npz` cache) and `AddGaussianNoise(std)` mirroring the GPU noise step at `physics_jepa/data.py:269-274`. A `--compute-stats` CLI on `data.py` walks the train split once and writes the cache.

### 3. `physics_ssl/masking.py` — `multi_block_mask_3d`

Generalize `stable_pretraining/data/masking.py:multi_block_mask` from `(H, W)` to `(T, H, W)`:

```python
def multi_block_mask_3d(
    grid_size: tuple[int, int, int],          # (T_p, H_p, W_p) e.g. (4,16,16)
    context_scale: tuple[float, float] = (0.85, 1.0),
    target_scale:  tuple[float, float] = (0.15, 0.20),
    target_aspect: tuple[float, float] = (0.75, 1.5),
    n_targets: int = 4,
    generator: torch.Generator | None = None,
) -> tuple[torch.BoolTensor, list[torch.BoolTensor]]:
    """Returns (context_mask, [target_masks]); each is a flat [T_p*H_p*W_p] bool.
    Context covers ~context_scale of all tokens; each target is a contiguous
    spatiotemporal block; context is set to (context_block AND NOT any target)."""
```

Implementation outline: sample a contiguous `(t0:t1, h0:h1, w0:w1)` block per scale, flatten the 3D mask, exclude target tokens from the context.

### 4. `physics_ssl/models/patch_embed_3d.py` — Video ViT builder

```python
class PatchEmbed3D(nn.Module):
    def __init__(self, in_chans, embed_dim, tubelet=(2,16,16)):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=tubelet, stride=tubelet)
        self.tubelet = tubelet
    def forward(self, x):                       # (B,C,T,H,W) -> (B,N,D)
        x = self.proj(x)                        # (B,D,T_p,H_p,W_p)
        return x.flatten(2).transpose(1, 2)

def build_video_vit(model_name="vit_small_patch16_224", in_chans=11,
                    img_size=256, num_frames=8, tubelet=(2,16,16)):
    vit = timm.create_model(model_name, pretrained=False, num_classes=0,
                            img_size=img_size, in_chans=in_chans)
    embed_dim = vit.embed_dim                   # 384 for ViT-S
    vit.patch_embed = PatchEmbed3D(in_chans, embed_dim, tubelet)
    grid = (num_frames // tubelet[0], img_size // tubelet[1], img_size // tubelet[2])
    n_tokens = grid[0] * grid[1] * grid[2]
    vit.pos_embed = nn.Parameter(
        sincos_3d_pos_embed(embed_dim, grid).unsqueeze(0), requires_grad=False)
    vit.num_patches = n_tokens
    vit._grid_size = grid
    return vit
```

After this swap, `MaskedEncoder(vit)` works because the masked encoder only needs `patch_embed`, `blocks`, optional `cls_token`/`reg_token`.

### 5. `physics_ssl/models/predictor.py` — V-JEPA predictor

```python
def build_vjepa_predictor(encoder_dim=384, predictor_dim=192,
                          depth=6, heads=6, grid_size=(4,16,16)):
    return FlexibleTransformer(
        dim=predictor_dim, depth=depth, num_heads=heads,
        self_attn=True, cross_attn=False,
        add_mask_token=True, pos_embed="sincos_3d", grid_size=grid_size,
        in_proj=nn.Linear(encoder_dim, predictor_dim),
        out_proj=nn.Linear(predictor_dim, encoder_dim),
    )
```

Matches the IJEPA predictor config at `vit.py:1182-1193`; we only swap `sincos_2d` → `sincos_3d` and bump grid to 3D.

### 6. `physics_ssl/forwards.py` — `vjepa_forward`

```python
def vjepa_forward(self, batch, stage):
    x = batch["video"]                                     # (B,C,T,H,W)
    grid = self.encoder.module._grid_size                  # (T_p,H_p,W_p)
    ctx_mask, tgt_masks = multi_block_mask_3d(grid, ...)
    ctx_mask = ctx_mask.to(x.device)
    tgt_masks = [m.to(x.device) for m in tgt_masks]

    ctx_out = self.encoder.forward_student(x, mask=ctx_mask)
    with torch.no_grad():
        tgt_all = self.encoder.forward_teacher(x).encoded
        tgt_all = F.layer_norm(tgt_all, (tgt_all.size(-1),))   # V-JEPA stop-grad LN

    losses = []
    for tgt_mask in tgt_masks:
        tgt_tokens = tgt_all[:, tgt_mask, :]
        pred = self.predictor(ctx_out.encoded,
                              ctx_ids=ctx_out.ids_keep,
                              tgt_ids=tgt_mask.nonzero(as_tuple=False).squeeze(-1))
        losses.append(F.smooth_l1_loss(pred, tgt_tokens))
    loss = torch.stack(losses).mean()

    with torch.no_grad():
        emb = tgt_all.mean(dim=1).detach()

    self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
    return {
        "loss": loss, "embedding": emb,
        "label": batch["label"].long(),
        "alpha": batch["alpha"].float(), "zeta": batch["zeta"].float(),
    }
```

Pattern mirrors `ijepa_forward` (`benchmarks/imagenet100/ijepa-vit-base.py:23-37`); the only V-JEPA-specific change is the 3D mask, the multi-target loop, and the smooth-L1 with stop-grad LayerNorm on targets.

### 7. `physics_ssl/callbacks.py`

- `RegressionProbe`: copy `stable_pretraining/callbacks/probe.py`'s structure but use `nn.Linear(384, 2)` + `nn.MSELoss`; logs `val/regprobe_mse_alpha`, `val/regprobe_mse_zeta`, `val/regprobe_r2`.
- `EvalEmbedding`: a no-op shim if `vjepa_forward` already surfaces `embedding`. Keep only if multi-layer probing is desired later.

### 8. `configs/vjepa_active_matter.yaml`

```yaml
defaults: [_self_]

data:
  train:
    dataset:
      _target_: physics_ssl.data.ActiveMatterVolumeDataset
      root: ${oc.env:THE_WELL_DATA_DIR}/active_matter/data/train
      num_frames: 8
      transform: { _target_: physics_ssl.transforms.ChannelZScore,
                   stats_path: cache/active_matter_stats.npz }
    batch_size: 16
    num_workers: 8
  val: { _target_: ..., split: valid, batch_size: 16 }

module:
  encoder:
    _target_: stable_pretraining.backbone.utils.TeacherStudentWrapper
    module:
      _target_: stable_pretraining.backbone.vit.MaskedEncoder
      backbone:
        _target_: physics_ssl.models.patch_embed_3d.build_video_vit
        in_chans: 11
        img_size: 256
        num_frames: 8
        tubelet: [2, 16, 16]
    base_ema_coefficient: 0.994
    final_ema_coefficient: 1.0
    warm_init: true
  predictor:
    _target_: physics_ssl.models.predictor.build_vjepa_predictor
    encoder_dim: 384
    predictor_dim: 192
    depth: 6
    heads: 6
    grid_size: [4, 16, 16]
  forward: physics_ssl.forwards.vjepa_forward

callbacks:
  - _target_: stable_pretraining.callbacks.TeacherStudentCallback
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
    probe: { _target_: torch.nn.Linear, in_features: 384, out_features: 16 }
    metrics: [{ _target_: torchmetrics.classification.MulticlassAccuracy, num_classes: 16 }]
  - _target_: physics_ssl.callbacks.RegressionProbe
    input: embedding
    targets: [alpha, zeta]
    in_features: 384

trainer:
  max_epochs: 200
  precision: bf16-mixed
  accelerator: gpu
  devices: 1
  gradient_clip_val: 1.0
```

---

## Critical reused functions / files

| What to reuse | Where it lives |
|---|---|
| HDF5 shard discovery (`_build_index`), field-stacking schema (`_build_global_field_schema`), frame sampling | `remy9926/physical-representation-learning/physics_jepa/data.py:103-134, 136-171, 194-280` |
| `t0_fields`/`t1_fields`/`t2_fields` parsing | same file, lines 117-150 |
| `alpha`/`zeta` extraction | `physics_jepa/data.py:130-132`, `finetuner.py` regression targets |
| Channel normalization + GPU noise pattern | `physics_jepa/data.py:269-274` |
| `MaskedEncoder` (timm-ViT-aware token masking) | `stable-pretraining/stable_pretraining/backbone/vit.py:229-468` |
| `FlexibleTransformer` (predictor with `add_mask_token=True`, `sincos_3d`) | `stable-pretraining/stable_pretraining/backbone/vit.py:1129-1671`, IJEPA config at 1182-1193 |
| `TeacherStudentWrapper` + `TeacherStudentCallback` (EMA target encoder) | `backbone/utils.py:336-493`, `callbacks/teacher_student.py:11-143` |
| 2D multi-block mask (template to extend to 3D) | `stable_pretraining/data/masking.py:81-142` |
| `IJEPA` / `ijepa_forward` (template to copy structure from) | `benchmarks/imagenet100/ijepa-vit-base.py:23-37` |
| `OnlineKNN`, `OnlineProbe`, probe template | `stable_pretraining/callbacks/{knn,probe}.py` |
| Hydra config layout | `stable-pretraining/examples/simclr_cifar10_config.yaml` |

---

## Why V-JEPA fits this dataset

- **Spatiotemporal volumes are V-JEPA's native input** — Active Matter's `(C, T, H, W)` matches the assumption directly.
- **No augmentations needed** — JEPA's pretext is masking, not invariance to color jitter / crops, which sidesteps the open question of "what augmentation preserves physics."
- **Continuous with `remy9926`'s research direction** — that project is already a JEPA variant; using V-JEPA here lets us compare *how* the prediction is structured (image-vs-future in `remy9926`, masked-blocks in V-JEPA) on identical evaluation.
- **All probing comes for free** — the framework-level `OnlineKNN` / `OnlineProbe` plus our `RegressionProbe` give us per-epoch representation-quality numbers without any post-hoc fine-tuning loop.

---

## Verification

1. **Env:** `export THE_WELL_DATA_DIR=…/the_well_data/datasets`.
2. **Stats cache:** `python -m physics_ssl.data --compute-stats --split train` → writes `cache/active_matter_stats.npz`.
3. **Mask sanity:** `python -c "from physics_ssl.masking import multi_block_mask_3d as m; ctx, tgts = m((4,16,16)); print(ctx.shape, ctx.float().mean(), [t.float().mean() for t in tgts])"` — verify context covers ~0.85 of tokens, each target ~0.15-0.20, targets disjoint.
4. **Backbone sanity:** load `build_video_vit(...)`, push a dummy `(2, 11, 8, 256, 256)` through, assert output shape `(2, 1024, 384)`.
5. **1-step dry run:**
   ```bash
   spt configs/vjepa_active_matter.yaml \
       trainer.max_epochs=1 trainer.limit_train_batches=2 trainer.limit_val_batches=1
   ```
   Confirm `train/loss` logs, no shape mismatches, EMA updates fire (look for `teacher_student/ema_coefficient`).
6. **Probe sanity:** after the first val epoch confirm `val/knn_probe_acc`, `val/linear_probe_acc`, `val/regprobe_mse_alpha`, `val/regprobe_mse_zeta`, `val/regprobe_r2` all appear.
7. **Real run:** 200 epochs single-GPU. Success signal: `val/regprobe_r2` improves monotonically and ends meaningfully > 0. Compare to a frozen `remy9926` JEPA encoder run through the same probing protocol.

If `MaskedEncoder` rejects the swapped `PatchEmbed3D` (it inspects `patch_embed.proj.weight` shape in some paths), fall back to a thin `Video ViT` class that **mimics** the timm ViT attribute layout (`patch_embed`, `blocks`, `cls_token=None`, `reg_token=None`, `pos_embed`, `norm`) — `MaskedEncoder` only uses these.

---

## Out of scope (deliberately)

- VICReg / SimCLR / DINO comparison runs — defer until V-JEPA has a baseline.
- Refactoring `remy9926` into `stable-pretraining`.
- Multi-dataset training (Shear Flow, Rayleigh-Bénard) — Active Matter only.
- Pretrained Video ViT initialization (e.g., `facebook/vjepa2-*`) — start from scratch; the input modality (11 physics channels) doesn't align with RGB-pretrained weights anyway.

---

## Reason this plan was superseded

After surveying `le-wm/` and `eb_jepa/`, both reference codebases implement **temporal-autoregressive JEPA** (predict next-frame embeddings from a history of past frames, no masking, no EMA target, anti-collapse via SIGReg or VC+IDM regularizers). LeWM is already built on `stable-pretraining`, with `OnlineKNN`/`OnlineProbe` integration in place. To match the references' correctness and to avoid building a 3D tubelet patch embed, a 3D mask sampler, and EMA plumbing, the design pivoted to temporal-AR JEPA. See `02_temporal_ar_jepa_plan.md`.
