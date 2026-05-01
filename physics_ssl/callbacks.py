"""RegressionProbe callback for continuous physical-parameter probing.

Skeleton modeled on `stable_pretraining/callbacks/probe.py:OnlineProbe`. The
deltas vs. `OnlineProbe`:

* Reads two scalar batch keys (`alpha`, `zeta` by default), stacks them into a
  `(B, 2)` regression target.
* Uses `nn.MSELoss` instead of CrossEntropy.
* Logs per-dim MSE plus a streaming R^2 against the running batch variance.
"""

from __future__ import annotations

import types
from functools import partial
from typing import Optional, Sequence, Union

import torch
import torchmetrics
from lightning.pytorch import LightningModule
from torch import nn

from stable_pretraining.callbacks.utils import TrainableCallback, log_header
from stable_pretraining.utils import detach_tensors, get_data_from_batch_or_outputs


class RegressionProbe(TrainableCallback):
    """Lightweight regression probe for continuous targets like `[alpha, zeta]`.

    Args:
        module: the `spt.Module` being probed (mirrors `OnlineProbe.module`).
        name: unique name for this probe (used as the metric prefix).
        input: batch / outputs key holding the embedding tensor `(B, D)`.
        targets: list of batch keys to stack column-wise into the regression
            target. Default `("alpha", "zeta")`.
        in_features: input dim for the probe head.
        loss: regression loss (defaults to MSE).
        optimizer / scheduler: same convention as `OnlineProbe`.
    """

    def __init__(
        self,
        module: LightningModule,
        name: str = "regprobe",
        input: str = "embedding",
        targets: Sequence[str] = ("alpha", "zeta"),
        in_features: int = 192,
        loss: Optional[callable] = None,
        optimizer: Optional[Union[str, dict, partial, torch.optim.Optimizer]] = None,
        scheduler: Optional[
            Union[str, dict, partial, torch.optim.lr_scheduler.LRScheduler]
        ] = None,
        accumulate_grad_batches: int = 1,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: str = "norm",
        verbose: Optional[bool] = None,
    ):
        from stable_pretraining.callbacks.utils import resolve_verbose

        self.input = input
        self.targets = tuple(targets)
        self.in_features = int(in_features)
        self.loss = loss if loss is not None else nn.MSELoss()
        self.verbose = resolve_verbose(verbose)
        self._probe_config = nn.Linear(self.in_features, len(self.targets))

        super().__init__(
            module=module,
            name=name,
            optimizer=optimizer,
            scheduler=scheduler,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

        # Per-dim MSE + R^2.
        self.metrics = {
            "mse": torchmetrics.MeanSquaredError(),
            "r2": torchmetrics.R2Score(num_outputs=len(self.targets), multioutput="uniform_average"),
        }

        log_header("RegressionProbe")
        from loguru import logger as logging
        logging.info(f"  name: {self.name}")
        logging.info(f"  input: {input}")
        logging.info(f"  targets: {self.targets}")
        logging.info(f"  in_features: {self.in_features}")
        self.wrap_forward(pl_module=module)

    # ------------------------------------------------------------------
    # Mirrors OnlineProbe.configure_model — but the head was already built.

    def configure_model(self, pl_module: LightningModule) -> nn.Module:
        return self._probe_config

    # ------------------------------------------------------------------

    def _stack_targets(self, batch: dict, outputs: dict) -> torch.Tensor:
        cols = []
        for k in self.targets:
            t = get_data_from_batch_or_outputs(k, batch, outputs, caller_name=self.name)
            if t is None:
                raise ValueError(f"{self.name}: missing target key '{k}' in batch / outputs")
            cols.append(t.float().reshape(-1))
        return torch.stack(cols, dim=-1)  # (B, len(targets))

    def wrap_forward(self, pl_module: LightningModule):
        fn = pl_module.forward

        def new_forward(self_module, batch, stage, callback=self, fn=fn):
            outputs = fn(batch, stage)

            x = get_data_from_batch_or_outputs(
                callback.input, batch, outputs, caller_name=callback.name
            )
            if x is None:
                raise ValueError(f"{callback.name}: missing input key '{callback.input}'")
            y = callback._stack_targets(batch, outputs)

            preds = callback.module(detach_tensors(x))
            y = detach_tensors(y)

            outputs[f"{callback.name}_preds"] = preds

            scalar_logs, metric_logs = {}, {}
            if stage == "fit":
                loss = callback.loss(preds, y)
                outputs["loss"] = outputs.get("loss", 0) + loss
                scalar_logs[f"train/{callback.name}_loss"] = loss.item()

                # Per-dim MSE for logging.
                with torch.no_grad():
                    per_dim = (preds - y).pow(2).mean(dim=0)
                    for k, v in zip(callback.targets, per_dim.detach().cpu().tolist()):
                        scalar_logs[f"train/{callback.name}_mse_{k}"] = float(v)

                my_metrics = self_module.callbacks_metrics[callback.name]["_train"]
                for metric_name, metric in my_metrics.items():
                    metric.update(preds.detach(), y)
                    metric_logs[f"train/{callback.name}_{metric_name}"] = metric

            elif stage == "validate":
                with torch.no_grad():
                    per_dim = (preds - y).pow(2).mean(dim=0)
                    for k, v in zip(callback.targets, per_dim.detach().cpu().tolist()):
                        scalar_logs[f"eval/{callback.name}_mse_{k}"] = float(v)

                my_metrics = pl_module.callbacks_metrics[callback.name]["_val"]
                for metric_name, metric in my_metrics.items():
                    metric(preds.detach(), y)
                    metric_logs[f"eval/{callback.name}_{metric_name}"] = metric

            elif stage == "test":
                # spt's metric registry only allocates `_train`/`_val` slots, so
                # we emit per-dim and overall MSE as plain scalars and let
                # Lightning's on_epoch reduction average across the test pass.
                with torch.no_grad():
                    per_dim = (preds - y).pow(2).mean(dim=0)
                    for k, v in zip(callback.targets, per_dim.detach().cpu().tolist()):
                        scalar_logs[f"test/{callback.name}_mse_{k}"] = float(v)
                    scalar_logs[f"test/{callback.name}_mse"] = (preds - y).pow(2).mean().item()

            if scalar_logs:
                self_module.log_dict(scalar_logs, on_step=True, on_epoch=True, sync_dist=True)
            if metric_logs:
                self_module.log_dict(metric_logs, on_step=True, on_epoch=True, sync_dist=False)

            return outputs

        pl_module.forward = types.MethodType(new_forward, pl_module)
