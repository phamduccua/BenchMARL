#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Any, Dict, Optional, Type

import torch

from benchmarl.optimizers.common import OptimizerConfig


@dataclass
class AdamConfig(OptimizerConfig):
    """Configuration dataclass for :class:`torch.optim.Adam`.

    This is the default optimizer, and it reproduces exactly the behaviour
    BenchMARL had before optimizers became configurable: every field defaults to
    ``null``, which means "read it from the experiment config"
    (``lr``, ``adam_eps`` and ``adam_extra_kwargs`` of
    ``benchmarl/conf/experiment/base_experiment.yaml``).

    Args:
        lr (float, optional): overrides ``experiment.lr``
        eps (float, optional): overrides ``experiment.adam_eps``
        extra_kwargs (dict, optional): overrides ``experiment.adam_extra_kwargs``
    """

    lr: Optional[float] = MISSING
    eps: Optional[float] = MISSING
    extra_kwargs: Optional[Dict[str, Any]] = MISSING

    @staticmethod
    def associated_class() -> Type[torch.optim.Optimizer]:
        return torch.optim.Adam

    def _optimizer_kwargs(self, experiment_config) -> Dict[str, Any]:
        extra_kwargs = self.extra_kwargs
        if extra_kwargs is None:
            extra_kwargs = experiment_config.adam_extra_kwargs
        return {
            "lr": self.lr if self.lr is not None else experiment_config.lr,
            "eps": self.eps if self.eps is not None else experiment_config.adam_eps,
            **extra_kwargs,
        }
