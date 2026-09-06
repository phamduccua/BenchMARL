#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

import torch

from benchmarl.optimizers.common import OptimizerConfig


class AdamCosine(torch.optim.Adam):
    """Adam with a cosine-annealed learning rate, decayed inside ``step()``.

    This is a **control branch**, not part of the paper. ``pc`` / ``pcvi`` /
    ``adaptive`` all shrink their effective step as training proceeds, so
    "pcvi beats adam" is confounded with "a decaying step beats a constant one".
    This branch removes that confound: same Adam, same tuned ``lr``, only the
    schedule added.

    The decay is applied here rather than through a ``torch.optim.lr_scheduler``
    because BenchMARL's training loop never calls a scheduler — the optimizer is
    the only object the loop touches per update.

    The schedule is the standard one, over the whole run rather than per
    collection round::

        lr_t = eta_min + (lr_0 - eta_min) * (1 + cos(pi * t / total_steps)) / 2

    Args:
        params (iterable): parameters to optimize
        total_steps (int): number of ``step()`` calls over which to anneal. After
            this many steps the learning rate stays at ``eta_min``.
        eta_min (float): the floor of the schedule
        **kwargs: passed to :class:`torch.optim.Adam`
    """

    def __init__(self, params, total_steps: int, eta_min: float = 0.0, **kwargs):
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}")
        super().__init__(params, **kwargs)
        self.total_steps = total_steps
        self.eta_min = eta_min
        self._step_count_cosine = 0
        # Captured before the first decay so that a restored optimizer anneals
        # from the configured peak, not from wherever it happened to be.
        for group in self.param_groups:
            group.setdefault("initial_lr", group["lr"])

    @torch.no_grad()
    def step(self, closure=None):
        t = min(self._step_count_cosine, self.total_steps)
        cos_factor = (1.0 + math.cos(math.pi * t / self.total_steps)) / 2.0
        for group in self.param_groups:
            group["lr"] = self.eta_min + (group["initial_lr"] - self.eta_min) * cos_factor
        self._step_count_cosine += 1
        return super().step(closure)

    def state_dict(self):
        state = super().state_dict()
        state["_step_count_cosine"] = self._step_count_cosine
        return state

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        self._step_count_cosine = state_dict.pop("_step_count_cosine", 0)
        super().load_state_dict(state_dict)


@dataclass
class AdamCosineConfig(OptimizerConfig):
    """Configuration dataclass for :class:`AdamCosine`.

    Control branch: Adam with a cosine-annealed learning rate. See
    :class:`AdamCosine` for why this branch exists.

    Args:
        lr (float, optional): the peak learning rate. ``null`` reads
            ``experiment.lr``.
        eps (float, optional): ``null`` reads ``experiment.adam_eps``.
        eta_min (float): the learning rate the schedule decays to.
        total_steps (int, optional): number of optimizer steps to anneal over.
            ``null`` derives it from the experiment config as
            ``max_n_iters * n_minibatch_iters * ceil(batch / minibatch)``.
        extra_kwargs (dict, optional): ``null`` reads
            ``experiment.adam_extra_kwargs``.
    """

    lr: Optional[float] = None
    eps: Optional[float] = None
    eta_min: float = 0.0
    total_steps: Optional[int] = None
    extra_kwargs: Optional[Dict[str, Any]] = None

    @staticmethod
    def associated_class() -> Type[torch.optim.Optimizer]:
        return AdamCosine

    def _total_steps(self, experiment_config) -> int:
        if self.total_steps is not None:
            return self.total_steps
        # Derived for the ON-POLICY path: every experiment this branch controls
        # for (IPPO / MAPPO) is on-policy. Set `total_steps` explicitly to use it
        # with an off-policy algorithm.
        on_policy = True
        if experiment_config.max_n_iters is None and experiment_config.max_n_frames is None:
            raise ValueError(
                "AdamCosineConfig needs to know the length of the run to anneal over. "
                "Set experiment.max_n_iters or experiment.max_n_frames, or set "
                "optimizer.total_steps explicitly."
            )
        n_iters = experiment_config.get_max_n_iters(on_policy)
        updates_per_round = experiment_config.n_optimizer_steps(on_policy) * -(
            -experiment_config.train_batch_size(on_policy)
            // experiment_config.train_minibatch_size(on_policy)
        )
        return n_iters * updates_per_round

    def _optimizer_kwargs(self, experiment_config) -> Dict[str, Any]:
        extra_kwargs = self.extra_kwargs
        if extra_kwargs is None:
            extra_kwargs = experiment_config.adam_extra_kwargs
        return {
            "lr": self.lr if self.lr is not None else experiment_config.lr,
            "eps": self.eps if self.eps is not None else experiment_config.adam_eps,
            "eta_min": self.eta_min,
            "total_steps": self._total_steps(experiment_config),
            **extra_kwargs,
        }
