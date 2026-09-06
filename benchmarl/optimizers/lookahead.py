#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Lookahead: k steps forward, 1 step back.

    Michael R. Zhang, James Lucas, Geoffrey Hinton, Jimmy Ba,
    "Lookahead Optimizer: k steps forward, 1 step back", NeurIPS 2019.
    https://arxiv.org/abs/1907.08610

Two sets of weights are kept: slow ``phi`` and fast ``theta``. The fast weights
run ``k`` steps of an inner optimizer, then::

    phi   <- phi + alpha (theta - phi)
    theta <- phi

so ``phi`` is an exponential moving average over the endpoints of the fast
trajectory. That filters the variance of stochastic gradients without having to
lower the learning rate.

**This is not the look-ahead of extragradient.** Same word, different mechanism:

===================  ==========================  ===========================
                     looks ahead by              uses it by
===================  ==========================  ===========================
extragradient/pcvi   one gradient step, to v_k   taking the gradient AT v_k
Lookahead            k optimizer steps           interpolating phi towards it
===================  ==========================  ===========================

``v_k`` is never an iterate; ``theta_{t,k}`` is one, it just gets pulled back.

This is a *wrapper*: it accepts any other optimizer in the registry as its inner
optimizer, including the two-gradient ones (``pcvi``, ``pc``, ``extragradient``,
``adaptive``), whose ``probe``/``restore``/``apply`` protocol it forwards.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, MISSING
from typing import Any, Dict, List, Optional, Type

import torch

from benchmarl.optimizers.common import OptimizerConfig


class Lookahead(torch.optim.Optimizer):
    """Lookahead (Zhang et al., 2019) wrapped around an inner optimizer.

    Args:
        params (iterable): the parameters to optimize. Must form a single param
            group; the inner optimizer has to be built on the **same** objects.
        inner (torch.optim.Optimizer): the already-built inner optimizer.
        k (int): sync period, in inner steps. Zhang et al. use 5 or 10.

            BenchMARL runs ``n_optimizer_steps * n_minibatches`` updates per
            collection round (675 by default) and then collects with the live
            policy. If ``k`` does not divide that, data is collected with *fast*
            weights mid-cycle, whereas the paper treats ``phi`` as the model.
            Values of ``k`` dividing 675: 5, 9, 15, 27, 45.
        alpha (float): the interpolation coefficient, in ``(0, 1]``. ``alpha = 1``
            makes Lookahead a no-op.
        sync_at_round_end (bool): force a sync at the end of every collection
            round, so data is always collected with the slow weights. **Not in
            the paper**: a partial cycle then gets the same ``alpha`` as a full
            one.
    """

    def __init__(
        self,
        params,
        inner: torch.optim.Optimizer,
        k: int = 5,
        alpha: float = 0.5,
        sync_at_round_end: bool = False,
    ):
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"k has to be a positive integer, got {k}")
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha has to be in (0, 1], got {alpha}")
        if alpha == 1:
            warnings.warn(
                "Lookahead with alpha=1 is a no-op: phi <- phi + 1*(theta - phi) "
                "is just phi <- theta. The inner optimizer runs unchanged."
            )

        super().__init__(
            params,
            defaults={"k": k, "alpha": alpha, "sync_at_round_end": sync_at_round_end},
        )
        if len(self.param_groups) != 1:
            raise ValueError("Lookahead needs exactly one param group")

        self._inner = inner
        self.k = k
        self.alpha = alpha
        self.sync_at_round_end = sync_at_round_end

        self._slow: List[torch.Tensor] = [
            prm.detach().clone() for prm in self._params
        ]
        self._since_sync = 0
        self.n_syncs = 0

    @property
    def _params(self) -> List[torch.Tensor]:
        return self.param_groups[0]["params"]

    @property
    def inner(self) -> torch.optim.Optimizer:
        """The wrapped optimizer."""
        return self._inner

    # -----------------------------------------------------------------
    @torch.no_grad()
    def _sync(self) -> bool:
        """``phi <- phi + alpha (theta - phi)``, then ``theta <- phi``."""
        torch._foreach_add_(
            self._slow,
            torch._foreach_sub(self._params, self._slow),
            alpha=self.alpha,
        )
        for prm, slow in zip(self._params, self._slow):
            prm.data.copy_(slow)
        self._since_sync = 0
        self.n_syncs += 1
        # The pull-back is not an optimizer step, so any inner optimizer that
        # estimates something from consecutive iterates has to forget its
        # history (see Adaptive with lipschitz_from="iterates").
        if hasattr(self._inner, "invalidate_iterate_history"):
            self._inner.invalidate_iterate_history()
        return True

    def _maybe_sync(self) -> bool:
        self._since_sync += 1
        if self._since_sync < self.k:
            return False
        return self._sync()

    def sync_now(self) -> bool:
        """Force a sync, if ``sync_at_round_end``. Called by the experiment."""
        if self.sync_at_round_end and self._since_sync:
            return self._sync()
        return False

    # -----------------------------------------------------------------
    # single-gradient path
    # -----------------------------------------------------------------
    def step(self, closure=None) -> Dict[str, float]:
        info = self._inner.step(closure)
        synced = self._maybe_sync()
        return self._info(info, synced)

    # -----------------------------------------------------------------
    # two-gradient path: forward the protocol, sync AFTER the inner update
    # -----------------------------------------------------------------
    def probe(self):  # noqa: D102
        self._inner.probe()

    def restore(self):  # noqa: D102
        self._inner.restore()

    def apply(self) -> Dict[str, float]:  # noqa: D102
        info = self._inner.apply()  # the inner optimizer moves theta first
        synced = self._maybe_sync()  # only then does Lookahead pull it back
        return self._info(info, synced)

    def reset_lambda(self):  # noqa: D102
        if hasattr(self._inner, "reset_lambda"):
            self._inner.reset_lambda()

    def _info(self, inner_info, synced: bool) -> Dict[str, float]:
        info = dict(inner_info) if isinstance(inner_info, dict) else {}
        info["lookahead_synced"] = float(synced)
        info["lookahead_since_sync"] = float(self._since_sync)
        return info

    # -----------------------------------------------------------------
    def zero_grad(self, set_to_none: bool = True):  # noqa: D102
        super().zero_grad(set_to_none=set_to_none)
        self._inner.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:  # noqa: D102
        return {
            "outer": super().state_dict(),
            "inner": self._inner.state_dict(),
            "slow": [t.detach().clone() for t in self._slow],
            "since_sync": self._since_sync,
            "n_syncs": self.n_syncs,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):  # noqa: D102
        self._since_sync = state_dict["since_sync"]
        self.n_syncs = state_dict["n_syncs"]
        for slow, saved in zip(self._slow, state_dict["slow"]):
            slow.copy_(saved)
        self._inner.load_state_dict(state_dict["inner"])
        super().load_state_dict(state_dict["outer"])


@dataclass
class LookaheadConfig(OptimizerConfig):
    """Configuration dataclass for :class:`~benchmarl.optimizers.Lookahead`.

    Args:
        inner (str): name of the inner optimizer in
            ``benchmarl.optimizers.optimizer_config_registry``. Anything but
            ``"lookahead"``.
        inner_overrides (dict): fields to override on the inner config after it
            has been loaded from its own yaml.
        k (int): sync period.
        alpha (float): interpolation coefficient.
        sync_at_round_end (bool): **not in the paper**, see :class:`Lookahead`.
    """

    inner: str = MISSING
    inner_overrides: Optional[Dict[str, Any]] = MISSING
    k: int = MISSING
    alpha: float = MISSING
    sync_at_round_end: bool = MISSING

    def inner_config(self) -> OptimizerConfig:
        """The inner optimizer's config, loaded from its yaml and overridden."""
        # Imported here, not at module level: benchmarl.optimizers imports this
        # module, so a top-level import would be circular.
        from benchmarl.optimizers import optimizer_config_registry

        if self.inner == "lookahead":
            raise ValueError("Lookahead cannot wrap itself")
        if self.inner not in optimizer_config_registry:
            raise ValueError(
                f"Unknown inner optimizer {self.inner!r}. "
                f"Available: {sorted(optimizer_config_registry)}"
            )
        config = optimizer_config_registry[self.inner].get_from_yaml()
        for key, value in (self.inner_overrides or {}).items():
            if not hasattr(config, key):
                raise ValueError(
                    f"{type(config).__name__} has no field {key!r} to override"
                )
            setattr(config, key, value)
        return config

    @staticmethod
    def associated_class() -> Type[torch.optim.Optimizer]:
        return Lookahead

    # Both of these DELEGATE: whether a second backward pass is needed, and
    # whether the gradient is treated as an operator, are properties of the
    # inner optimizer, not of the wrapper.
    def requires_two_gradient_evals(self) -> bool:
        return self.inner_config().requires_two_gradient_evals()

    def uses_gradient_as_operator(self) -> bool:
        return self.inner_config().uses_gradient_as_operator()

    def get_optimizer(self, params, experiment_config) -> torch.optim.Optimizer:
        # The inner optimizer must be built on the very same parameter objects.
        params = list(params)
        inner = self.inner_config().get_optimizer(params, experiment_config)
        return Lookahead(
            params,
            inner=inner,
            k=self.k,
            alpha=self.alpha,
            sync_at_round_end=self.sync_at_round_end,
        )
