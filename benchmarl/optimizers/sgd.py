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
class SgdConfig(OptimizerConfig):
    """Configuration dataclass for :class:`torch.optim.SGD`.

    This exists as the **control branch** for the projection-contraction
    optimizers. ``pc``, ``pcvi`` and ``extragradient`` are plain gradient methods:
    they drop Adam's per-coordinate normalisation entirely. Comparing them only
    against Adam therefore confounds two things -- the update rule, and the
    presence of a preconditioner. SGD isolates the second.

    Args:
        lr (float, optional): overrides ``experiment.lr``
        momentum (float): SGD momentum. 0.0 keeps it a pure gradient method,
            which is what makes it the right control.
        extra_kwargs (dict, optional): further kwargs for :class:`torch.optim.SGD`
    """

    lr: Optional[float] = MISSING
    momentum: float = MISSING
    extra_kwargs: Optional[Dict[str, Any]] = MISSING

    @staticmethod
    def associated_class() -> Type[torch.optim.Optimizer]:
        return torch.optim.SGD

    def _optimizer_kwargs(self, experiment_config) -> Dict[str, Any]:
        extra_kwargs = self.extra_kwargs
        if extra_kwargs is None:
            extra_kwargs = {}
        return {
            "lr": self.lr if self.lr is not None else experiment_config.lr,
            "momentum": self.momentum,
            **extra_kwargs,
        }
