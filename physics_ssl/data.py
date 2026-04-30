"""Active Matter (The Well) dataset adapted for temporal-AR JEPA.

Adapted from ``remy9926/physical-representation-learning/physics_jepa/data.py``
(``WellDatasetForJEPA``, lines 18-333). The structural changes:

* Returns a single contiguous video volume ``(C=11, T, H, W)`` per sample, not
  a context/target pair, since the temporal-AR JEPA forward operates on the
  whole sequence and slices its own history/target windows internally.
* Surfaces continuous physical params ``alpha``, ``zeta`` (named explicitly
  via the HDF5 ``scalars`` group) plus a discretized ``label`` so the
  classification probes (``OnlineKNN``, ``OnlineProbe``) have a target.
* Provides a ``--compute-stats`` CLI that walks the train split once and
  writes per-channel mean/std to a ``.npz`` file used by ``ChannelZScore``.

Reused logic copied (not imported) from ``physics_jepa/data.py``:

* HDF5 shard discovery: ``_build_index`` (lines 103-134).
* Field schema parsing for ``t0/t1/t2_fields``: ``_build_global_field_schema``
  (lines 136-171).
* ``alpha``/``zeta`` scalar extraction pattern: lines 130-132.
* Optional Gaussian noise injection step: lines 269-274 (factored into
  ``physics_ssl/transforms.py:AddGaussianNoise`` here).

Active Matter conventions (from remy9926):

* HDF5 layout: ``t0_fields/{name}``, ``t1_fields/{name}``, ``t2_fields/{name}``
  plus ``scalars/{alpha,zeta,L}``. ``L`` is constant across the dataset and
  is ignored.
* Total channels = 11 (after flattening the per-field component dims).
"""

from __future__ import annotations

import argparse
import logging
import os
import weakref
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# 4x4 grid of (alpha, zeta) -> 16 bins for classification probes. The exact
# physical ranges for Active Matter in The Well: alpha in {-1, 0, 1, 3, 5},
# zeta in {1, 3, 5, 7, 9, 11, 13, 15, 17}. We z-score then quantile-bin.
_DEFAULT_BINS = 4


class ActiveMatterVideoDataset(Dataset):
    """Active Matter HDF5 dataset emitting `{video, label, alpha, zeta}` dicts.

    Args:
        root: Directory containing `train/`, `valid/` subdirs, e.g.
            `${THE_WELL_DATA_DIR}/active_matter/data`. If `split` is given as a
            subdir name, it is appended.
        split: 'train' or 'valid' (or 'val' aliased to 'valid').
        num_frames: Length of the video volume per sample (T).
        stride: Temporal stride between consecutive sample windows. Defaults to
            `num_frames` (non-overlapping windows).
        resolution: Optional `(H, W)` resize.
        transform: Callable applied to the `(C, T, H, W)` volume.
        bins_per_param: Number of bins per parameter for the classification
            label (so total classes = bins_per_param ** 2).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        num_frames: int = 8,
        stride: Optional[int] = None,
        resolution: Optional[tuple[int, int]] = None,
        transform=None,
        bins_per_param: int = _DEFAULT_BINS,
        param_stats: Optional[dict] = None,
        max_open_files: int = 6,
        rdcc_nbytes: int = 512 * 1024 ** 2,
        rdcc_nslots: int = 1_000_003,
        rdcc_w0: float = 0.75,
    ):
        if split == "val":
            split = "valid"
        root = Path(root)
        # Allow either `…/active_matter/data` or `…/active_matter/data/train`.
        if (root / split).is_dir():
            self.split_dir = root / split
        else:
            self.split_dir = root
        self.split = split
        self.num_frames = int(num_frames)
        assert self.num_frames > 0
        self.stride = stride if stride is not None else self.num_frames
        self.resolution = tuple(resolution) if resolution is not None else None
        self.transform = transform
        self.bins_per_param = int(bins_per_param)
        # Read by `_compute_label_edges`: if non-None, adopt these stats instead
        # of computing per-split. Used to share train-set α/ζ z-score stats with
        # val so the regression probe head is evaluated against identically
        # scaled targets.
        self._provided_param_stats = param_stats

        if self.bins_per_param < 2:
            raise ValueError(
                f"bins_per_param must be >= 2 (got {bins_per_param}); "
                "with a single bin every sample collapses to label 0 and the "
                "classification probes are meaningless."
            )

        self._open: OrderedDict[str, tuple[h5py.File, dict]] | None = None
        self._max_open_files = int(max_open_files)
        self._rdcc = (int(rdcc_nbytes), int(rdcc_nslots), float(rdcc_w0))

        self.index, self._scalars_per_file = self._build_index()
        if not self.index:
            raise ValueError(
                f"No valid windows in {self.split_dir}. "
                f"Need trajectories with at least num_frames={self.num_frames} timesteps."
            )
        self._build_field_schema(self.split_dir / self.index[0][0])
        self._compute_label_edges()
        print(
            f"[ActiveMatterVideoDataset] split={split} files={len(self._scalars_per_file)} "
            f"windows={len(self.index)} channels={self._C_total} HxW={self._spatial_shape}",
            flush=True,
        )

    # -------- Indexing --------

    def _build_index(self):
        """Scan HDF5 shards once at construction; return (index, scalars_per_file).

        index entries: (rel_path, obj_idx, t0) where rel_path is the shard's
        path relative to ``self.split_dir`` (preserving any subdirectory
        structure so two shards with the same basename do not alias). The
        scalars dict is keyed by the same relative path.
        """
        idx = []
        scalars: dict[str, dict[str, np.ndarray]] = {}
        F = self.num_frames
        paths = sorted(
            list(self.split_dir.rglob("*.h5")) + list(self.split_dir.rglob("*.hdf5"))
        )
        skipped = 0
        for path in paths:
            rel_path = path.relative_to(self.split_dir).as_posix()
            with h5py.File(path, "r") as f:
                first_field = f["t0_fields"][list(f["t0_fields"].keys())[0]]
                T = int(first_field.shape[1])
                n_objs = int(first_field.shape[0])
                max_t0 = T - F
                if max_t0 < 0:
                    skipped += 1
                    logger.warning(
                        "skipping shard %s: T=%d < num_frames=%d",
                        rel_path, T, F,
                    )
                    continue
                for obj_id in range(n_objs):
                    for t0 in range(0, max_t0 + 1, self.stride):
                        idx.append((rel_path, obj_id, t0))
                # Scalars: pull alpha and zeta by name, ignore L (constant).
                if "scalars" in f:
                    s_group = f["scalars"]
                    file_scalars = {}
                    for key in ("alpha", "zeta"):
                        if key in s_group:
                            arr = np.asarray(s_group[key][()]).reshape(-1)
                            if arr.size == 1:
                                arr = np.broadcast_to(arr, (n_objs,)).copy()
                            file_scalars[key] = arr.astype(np.float32)
                    if "alpha" not in file_scalars or "zeta" not in file_scalars:
                        # Fallback: take first two non-L scalars in declared order.
                        non_L = [k for k in s_group.keys() if k != "L"]
                        if len(non_L) >= 2:
                            file_scalars.setdefault(
                                "alpha",
                                np.broadcast_to(
                                    np.asarray(s_group[non_L[0]][()]).reshape(-1),
                                    (n_objs,),
                                ).astype(np.float32),
                            )
                            file_scalars.setdefault(
                                "zeta",
                                np.broadcast_to(
                                    np.asarray(s_group[non_L[1]][()]).reshape(-1),
                                    (n_objs,),
                                ).astype(np.float32),
                            )
                    scalars[rel_path] = file_scalars
                else:
                    scalars[rel_path] = {
                        "alpha": np.zeros(n_objs, dtype=np.float32),
                        "zeta": np.zeros(n_objs, dtype=np.float32),
                    }
        if skipped:
            logger.warning(
                "skipped %d shards with T < num_frames; %d windows kept",
                skipped, len(idx),
            )
        return idx, scalars

    def _build_field_schema(self, sample_path: Path):
        field_paths, d_sizes, comp_shapes = [], [], []
        with h5py.File(sample_path, "r") as f:
            for group in ("t0_fields", "t1_fields", "t2_fields"):
                if group not in f:
                    continue
                for name, ds in f[group].items():
                    if not isinstance(ds, h5py.Dataset):
                        continue
                    comp = tuple(ds.shape[4:])  # () scalar, (2,) vector, (2,2) tensor
                    d_sizes.append(int(np.prod(comp) or 1))
                    comp_shapes.append(comp)
                    field_paths.append(f"{group}/{name}")
            if not field_paths:
                raise RuntimeError(f"No fields under {sample_path}")
            _, _, H, W = f[field_paths[0]].shape  # (n_objs, T, H, W, [comp])
            dtype = f[field_paths[0]].dtype

        d_sizes = np.asarray(d_sizes, dtype=np.int64)
        chan_offsets = np.concatenate(([0], np.cumsum(d_sizes)))
        self._field_paths = tuple(field_paths)
        self._d_sizes = d_sizes
        self._comp_shapes = comp_shapes
        self._chan_offsets = chan_offsets
        self._C_total = int(chan_offsets[-1])
        self._spatial_shape = (int(H), int(W))
        self._dtype = dtype

    def _compute_label_edges(self):
        """Quantile bin edges + z-score stats for alpha/zeta over this split.

        Using quantiles for the bin edges avoids one-bin-collapsed cases when
        the parameter grid is highly non-uniform.

        Z-score stats: if `param_stats` was supplied at construction (val using
        train stats), those are adopted instead of computing per-split — so the
        regression probe head trained on train z-scores is evaluated against
        identically-scaled val z-scores.
        """
        all_a = np.concatenate([s["alpha"] for s in self._scalars_per_file.values()])
        all_z = np.concatenate([s["zeta"] for s in self._scalars_per_file.values()])
        n = self.bins_per_param
        qs = np.linspace(0, 1, n + 1)[1:-1]
        self._alpha_edges = np.quantile(all_a, qs).astype(np.float32) if qs.size else np.array([], dtype=np.float32)
        self._zeta_edges = np.quantile(all_z, qs).astype(np.float32) if qs.size else np.array([], dtype=np.float32)

        if self._provided_param_stats is not None:
            s = self._provided_param_stats
            self._alpha_mean = float(s["alpha_mean"])
            self._alpha_std = float(s["alpha_std"])
            self._zeta_mean = float(s["zeta_mean"])
            self._zeta_std = float(s["zeta_std"])
        else:
            eps = 1e-6
            self._alpha_mean = float(all_a.mean())
            self._alpha_std = float(all_a.std() + eps)
            self._zeta_mean = float(all_z.mean())
            self._zeta_std = float(all_z.std() + eps)

    @property
    def param_stats(self) -> dict:
        """Mean/std for α and ζ — pass to the val dataset to share train stats."""
        return {
            "alpha_mean": self._alpha_mean,
            "alpha_std": self._alpha_std,
            "zeta_mean": self._zeta_mean,
            "zeta_std": self._zeta_std,
        }

    def _bin(self, a: float, z: float) -> int:
        a_bin = int(np.searchsorted(self._alpha_edges, a))
        z_bin = int(np.searchsorted(self._zeta_edges, z))
        return a_bin * self.bins_per_param + z_bin

    # -------- Worker-local file LRU --------

    def _ensure_worker_state(self):
        if self._open is None:
            self._open = OrderedDict()
            weakref.finalize(self, self._close_all)

    def _close_all(self):
        if self._open:
            for f, _ in self._open.values():
                try:
                    f.close()
                except Exception:
                    pass
            self._open.clear()

    def _open_file(self, file_id: str):
        self._ensure_worker_state()
        if file_id in self._open:
            f, st = self._open.pop(file_id)
            self._open[file_id] = (f, st)
            return f, st
        while len(self._open) >= self._max_open_files:
            _, (old_f, _) = self._open.popitem(last=False)
            try:
                old_f.close()
            except Exception:
                pass
        # Try SWMR (faster, parallel-reader-safe) first; many archived datasets
        # are not authored in SWMR mode and will raise OSError on open. Fall
        # back to a plain read-only handle in that case.
        common_kwargs = dict(
            mode="r",
            rdcc_nbytes=self._rdcc[0],
            rdcc_nslots=self._rdcc[1],
            rdcc_w0=self._rdcc[2],
        )
        path = self.split_dir / file_id
        try:
            f = h5py.File(path, libver="latest", swmr=True, **common_kwargs)
        except (OSError, ValueError):
            f = h5py.File(path, **common_kwargs)
        st: dict = {}
        self._open[file_id] = (f, st)
        return f, st

    def _get_ds(self, f, state, path):
        ds_cache = state.setdefault("ds_cache", {})
        if path in ds_cache:
            return ds_cache[path]
        ds = f[path]
        try:
            ds.id.set_chunk_cache(self._rdcc[1], self._rdcc[0], self._rdcc[2])
        except Exception:
            pass
        ds_cache[path] = ds
        return ds

    # -------- Dataset API --------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        file_id, obj_id, t0 = self.index[i]
        F = self.num_frames
        f, state = self._open_file(file_id)
        H, W = self._spatial_shape
        C = self._C_total

        out = np.empty((F, H, W, C), dtype=self._dtype, order="C")
        sel_prefix = (obj_id, slice(t0, t0 + F), slice(None), slice(None))
        tmp_cache = state.setdefault("buf_cache", {})

        c0 = 0
        for path, dsize, comp_shape in zip(self._field_paths, self._d_sizes, self._comp_shapes):
            c1 = c0 + int(dsize)
            ds = self._get_ds(f, state, path)
            need_shape = (F, H, W) + comp_shape
            buf = tmp_cache.get(comp_shape)
            if buf is None or buf.shape != need_shape or buf.dtype != self._dtype:
                buf = np.empty(need_shape, dtype=self._dtype, order="C")
                tmp_cache[comp_shape] = buf
            sel = sel_prefix + (slice(None),) * len(comp_shape)
            ds.read_direct(buf, source_sel=sel)
            out[..., c0:c1] = buf.reshape(F, H, W, c1 - c0)
            c0 = c1

        # (F, H, W, C) -> (C, T, H, W)
        video = torch.from_numpy(out).permute(3, 0, 1, 2).contiguous().float()

        if self.resolution is not None and tuple(video.shape[-2:]) != self.resolution:
            video = torch.nn.functional.interpolate(
                video, size=self.resolution, mode="bilinear", align_corners=False
            )

        if self.transform is not None:
            video = self.transform(video)

        scalars = self._scalars_per_file[file_id]
        alpha_raw = float(scalars["alpha"][obj_id])
        zeta_raw = float(scalars["zeta"][obj_id])
        alpha = (alpha_raw - self._alpha_mean) / self._alpha_std
        zeta = (zeta_raw - self._zeta_mean) / self._zeta_std

        return {
            "video": video,
            "label": self._bin(alpha_raw, zeta_raw),
            "alpha": alpha,
            "zeta": zeta,
            "alpha_raw": alpha_raw,
            "zeta_raw": zeta_raw,
        }

    def __getstate__(self):
        st = self.__dict__.copy()
        st["_open"] = None
        return st


# -------- Stats CLI --------


def compute_stats(root: str | Path, split: str, num_frames: int, output_path: str | Path,
                  max_samples: Optional[int] = 256):
    """Walk a (subset of) the dataset and write per-channel mean/std to `.npz`.

    `max_samples` caps the sample count to keep the cache step cheap; defaults
    to a few hundred windows which is plenty for stable statistics on physics
    fields with smooth distributions.
    """
    dataset = ActiveMatterVideoDataset(
        root=root, split=split, num_frames=num_frames, transform=None
    )
    if len(dataset) == 0:
        raise ValueError(f"No windows found for split={split!r} under {root}")
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    if n <= 0:
        raise ValueError(
            f"max_samples must be positive (got {max_samples}); cannot compute stats over zero windows."
        )
    indices = np.linspace(0, len(dataset) - 1, num=n, dtype=int)

    sum_, sumsq, count = None, None, 0
    for idx in indices:
        sample = dataset[int(idx)]
        v = sample["video"]  # (C, T, H, W)
        v_flat = v.reshape(v.shape[0], -1).double()
        if sum_ is None:
            sum_ = v_flat.sum(dim=1)
            sumsq = (v_flat ** 2).sum(dim=1)
        else:
            sum_ += v_flat.sum(dim=1)
            sumsq += (v_flat ** 2).sum(dim=1)
        count += v_flat.shape[1]

    mean = (sum_ / count).numpy().astype(np.float32)
    var = (sumsq / count).numpy().astype(np.float32) - mean ** 2
    std = np.sqrt(np.clip(var, 0, None)).astype(np.float32)

    # Surface zero-variance channels: an always-constant channel is almost
    # always a data-integrity bug (e.g. a missing field, an identically-zero
    # tensor component) and silently rescaling it by 1/eps would mask the bug.
    near_zero = np.where(std < 1e-6)[0]
    if near_zero.size:
        logger.warning(
            "channels with near-zero variance (std < 1e-6) — likely a constant "
            "or missing field; ChannelZScore will floor std at eps for these: %s",
            near_zero.tolist(),
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, mean=mean, std=std)
    print(f"Saved stats to {output_path}: mean shape={mean.shape}, std shape={std.shape}")


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute-stats", action="store_true", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--root", default=None,
                        help="Defaults to $THE_WELL_DATA_DIR/active_matter/data")
    parser.add_argument("--out", default="cache/active_matter_stats.npz")
    parser.add_argument("--max-samples", type=int, default=256)
    args = parser.parse_args()

    root = args.root
    if root is None:
        base = os.environ.get("THE_WELL_DATA_DIR")
        if not base:
            raise SystemExit("Set THE_WELL_DATA_DIR or pass --root")
        root = str(Path(base) / "active_matter" / "data")
    compute_stats(root, args.split, args.num_frames, args.out, args.max_samples)


if __name__ == "__main__":
    _main()
