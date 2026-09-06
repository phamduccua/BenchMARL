#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Adam driven by the adaptive step of Algorithm 1
(https://doi.org/10.1002/mma.11132).

This keeps IPPO's original update machinery and replaces only the constant
learning rate::

    theta_{k+1} = theta_k + Adam_{lr = lr_scale * lambda_k}(grad L_CLIP(theta_k))

with ``lambda_k`` produced by Step 3 of Algorithm 1. Steps 4-6 (``d_k``,
``beta_k``) are **not** used; ``v_k`` exists only to estimate ``lambda`` and is
never an iterate. See :mod:`benchmarl.optimizers.pcvi` for the variant that uses
the whole algorithm.

Step 3 thus becomes a *data-driven learning-rate schedule*: ``lambda_k`` is
non-increasing, so the learning rate can only decay, and it decays exactly when
the local Lipschitz constant of the gradient says the current step is too long.

Two things to read before using this (both documented at length in
``KE_HOACH_IMPLEMENT_ADAPTIVE.md``):

* ``lambda`` and Adam's ``lr`` do not have the same units. ``lambda ~ p / L`` is
  a step for the *raw* gradient, while Adam steps by ``lr * m/sqrt(v)``, whose
  magnitude is ~1 per coordinate whatever the gradient scale. Measured on IPPO +
  vmas/balance + MLP, ``lambda`` settles at ~2.1e-3 against a tuned Adam ``lr``
  of 5e-5, i.e. ~42x larger. ``lr_scale`` exists to reconcile the two.
* ``lambda_0`` has to be **above** ``p / L`` or the mechanism never fires and
  this is just IPPO at twice the cost.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, MISSING
from typing import Any, Dict, List, Optional, Type

import torch

from benchmarl.optimizers._lambda import (
    adaptive_lambda,
    global_dot,
    global_norm,
    to_float64,
)
from benchmarl.optimizers.common import OptimizerConfig

LIPSCHITZ_FROM = ("probe", "iterates")


class Adaptive(torch.optim.Optimizer):
    """Adam whose learning rate is Step 3 of Algorithm 1.

    One update, with ``F = grad L`` (so ``F = -grad L_CLIP``, the sign works out
    on its own -- see ``KE_HOACH_IMPLEMENT_IPPO.md`` section 2):

    1. ``F(u_k)`` from the caller's ``backward()``
    2. ``v_k = u_k - lambda_k F(u_k)`` -- a look-ahead, never an iterate
    3. ``F(v_k)`` from a second ``backward()`` on the **same** minibatch
    4. ``lambda_{k+1}`` from Step 3
    5. parameters back to ``u_k`` **and ``.grad`` back to ``F(u_k)``**
    6. ``Adam.step()`` with ``lr = lr_scale * lambda_k``

    Step 5 is the trap: after the probe, ``.grad`` holds ``F(v_k)``, but the
    update has to be Adam on the gradient **at** ``u_k``. Forgetting to restore
    it silently turns this into extragradient-with-Adam, with no error raised.
    ``test_equals_adam_when_lambda_frozen`` is there to catch exactly that.

    Args:
        params (iterable): the parameters to optimize. Must form a single param
            group, since ``lambda_k`` is a global scalar over the whole space.
        lambda_0 (float): the initial step of Step 1. ``lambda_k`` is
            non-increasing, so this bounds the learning rate for the whole run.
            It must be **above** ``p / L_local`` for the adaptation to do
            anything at all.
        p (float): the constant in ``(0, 1)`` of Step 1 (``theta`` in the paper,
            renamed to avoid the clash with the network parameters).
        eps_denominator (float): below this, ``||F(u_k) - F(v_k)||`` counts as 0.
        lr_scale (float): ``lr = lr_scale * lambda_k``. **Not in the paper**;
            1.0 is faithful. Use it to keep the *shape* of the schedule Step 3
            produces while putting it on a scale Adam can use: run once with 1.0,
            read the ``lambda`` it settles on, then set
            ``lr_scale = wanted_lr / lambda_settle``.
        lambda_min (float): floor for ``lambda_k``. **Not in the paper**; 0.0 is
            faithful. Setting ``lambda_min == lambda_0`` freezes the schedule,
            which is how the equivalence test above pins Adam.
        lipschitz_from (str): where the Lipschitz estimate comes from.

            * ``"probe"`` (default, faithful): from ``u_k`` and the look-ahead
              ``v_k``. Costs **two** gradients per update, for a step that only
              uses one of them.
            * ``"iterates"``: from ``theta_k`` and ``theta_{k-1}``. **One**
              gradient per update, i.e. the same cost as plain Adam, which makes
              the comparison against Adam exactly fair. **Not in the paper**:
              ``theta_k - theta_{k-1}`` is a step already taken through Adam, not
              ``-lambda_k F(u_k)``, so it estimates the Lipschitz constant along
              a different direction.
        reset_lambda_per_batch (bool): reset ``lambda_k`` to ``lambda_0`` at the
            start of every collection round. **Not in the paper**; the objective
            does change at every round, so the estimate carried over from the
            previous one is not obviously meaningful.
        eps (float): Adam's ``eps``.
        extra_kwargs (dict): further kwargs for :class:`torch.optim.Adam`.
    """

    def __init__(
        self,
        params,
        lambda_0: float,
        p: float,
        eps_denominator: float = 1e-12,
        lr_scale: float = 1.0,
        lambda_min: float = 0.0,
        lipschitz_from: str = "probe",
        reset_lambda_per_batch: bool = False,
        eps: float = 1e-8,
        extra_kwargs: Optional[Dict[str, Any]] = None,
        lambda_growth: float = 1.0,
        min_probe_rel: float = 0.0,
        float64_stats: bool = False,
    ):
        if not lambda_0 > 0:
            raise ValueError(f"lambda_0 has to be > 0, got {lambda_0}")
        if not 0 < p < 1:
            raise ValueError(f"p has to be in (0, 1), got {p}")
        if not lr_scale > 0:
            raise ValueError(f"lr_scale has to be > 0, got {lr_scale}")
        if lambda_min < 0:
            raise ValueError(f"lambda_min has to be >= 0, got {lambda_min}")
        if lambda_growth < 1.0:
            raise ValueError(
                f"lambda_growth has to be >= 1.0, got {lambda_growth}. 1.0 is the "
                f"paper's non-increasing lambda; above it, lambda may recover."
            )
        if min_probe_rel < 0:
            raise ValueError(f"min_probe_rel has to be >= 0, got {min_probe_rel}")
        if lambda_min > lambda_0:
            raise ValueError(
                f"lambda_min ({lambda_min}) has to be <= lambda_0 ({lambda_0})"
            )
        if lipschitz_from not in LIPSCHITZ_FROM:
            raise ValueError(
                f"lipschitz_from has to be one of {LIPSCHITZ_FROM}, "
                f"got {lipschitz_from!r}"
            )

        super().__init__(
            params,
            defaults={
                "lambda_0": lambda_0,
                "p": p,
                "eps_denominator": eps_denominator,
                "lr_scale": lr_scale,
                "lambda_min": lambda_min,
                "lambda_growth": lambda_growth,
                "min_probe_rel": min_probe_rel,
                "float64_stats": float64_stats,
                "lipschitz_from": lipschitz_from,
                "reset_lambda_per_batch": reset_lambda_per_batch,
            },
        )
        if len(self.param_groups) != 1:
            raise ValueError(
                "Adaptive needs exactly one param group: lambda_k is a global "
                "quantity over the whole space and cannot be split per tensor."
            )

        self.lambda_0 = lambda_0
        self.p = p
        self.eps_denominator = eps_denominator
        self.lr_scale = lr_scale
        self.lambda_min = lambda_min
        self.lambda_growth = lambda_growth
        self.min_probe_rel = min_probe_rel
        self.float64_stats = float64_stats
        # how often the probe was too small to carry information
        self.n_probe_too_small = 0
        self.lipschitz_from = lipschitz_from
        self.reset_lambda_per_batch = reset_lambda_per_batch

        self.lambda_k = lambda_0
        self.n_steps = 0

        # The inner Adam MUST be built on the very same parameter objects, not on
        # copies, or its step() would update different tensors.
        self._adam = torch.optim.Adam(
            self._params,
            lr=lr_scale * lambda_0,
            eps=eps,
            **(extra_kwargs if extra_kwargs is not None else {}),
        )

        # filled between probe() and apply()
        self._u: Optional[List[torch.Tensor]] = None
        self._f_u: Optional[List[torch.Tensor]] = None
        self._probe_lambda: float = lambda_0
        # "iterates" mode: theta_{k-1} and F(theta_{k-1})
        self._prev_params: Optional[List[torch.Tensor]] = None
        self._prev_grads: Optional[List[torch.Tensor]] = None

    @property
    def _params(self) -> List[torch.Tensor]:
        return self.param_groups[0]["params"]

    def _gather_grads(self) -> List[torch.Tensor]:
        return [
            (
                prm.grad.detach().clone()
                if prm.grad is not None
                else torch.zeros_like(prm)
            )
            for prm in self._params
        ]

    def invalidate_iterate_history(self):
        """Forgets ``theta_{k-1}`` and ``F(theta_{k-1})``.

        Called by :class:`~benchmarl.optimizers.Lookahead` after it pulls the
        parameters back to the slow weights. That jump is not an optimizer step,
        so ``||theta_k - theta_{k-1}|| / ||F(theta_k) - F(theta_{k-1})||`` across
        it would not be a Lipschitz estimate at all, and would drag ``lambda``
        down for no reason.
        """
        self._prev_params = self._prev_grads = None

    def reset_lambda(self):
        """Sets ``lambda_k`` back to ``lambda_0``, if ``reset_lambda_per_batch``.

        Called by the experiment at the start of every collection round. **Not in
        the paper.**
        """
        if self.reset_lambda_per_batch:
            self.lambda_k = self.lambda_0
            self._prev_params = self._prev_grads = None

    # -----------------------------------------------------------------
    # the Adam step, shared by both modes
    # -----------------------------------------------------------------
    @torch.no_grad()
    def _adam_step(self, f_u: List[torch.Tensor], lam: float) -> float:
        """Restores ``.grad`` to ``F(u_k)`` and takes one Adam step at ``lr``.

        Uses ``lambda_k``, not ``lambda_{k+1}``: Step 6 of Algorithm 1 builds its
        update from ``lambda_k``, and staying consistent with that is the only
        way this variant lines up with PCVI step for step.
        """
        for prm, grad in zip(self._params, f_u):
            prm.grad = grad
        lr = self.lr_scale * lam
        for group in self._adam.param_groups:
            group["lr"] = lr
        self._adam.step()
        return lr

    # -----------------------------------------------------------------
    # "probe" mode: two gradients, driven by _optimizer_loop_two_point
    # -----------------------------------------------------------------
    @torch.no_grad()
    def probe(self):
        """Phase 1: read ``F(u_k)``, then move the parameters to ``v_k``."""
        params = self._params
        # The clone is mandatory: the next backward will overwrite .grad.
        self._u = [prm.detach().clone() for prm in params]
        self._f_u = self._gather_grads()
        self._probe_lambda = self.lambda_k
        torch._foreach_add_(params, self._f_u, alpha=-self._probe_lambda)

    @torch.no_grad()
    def restore(self):
        """Puts the parameters back to ``u_k`` after a :meth:`probe`."""
        if self._u is None:
            return
        for prm, u_i in zip(self._params, self._u):
            prm.data.copy_(u_i)

    @torch.no_grad()
    def _next_lambda(self, lam, delta, delta_grad):
        """Step 3, with the two guards that a real run showed it needs.

        ``delta`` and ``delta_grad`` are the vectors ``u_k - v_k`` and
        ``F(u_k) - F(v_k)``; the norms are taken here so both call sites -- the
        probe path and the "iterates" path -- get the same treatment.

        Measured on matrixgame/rock_paper_scissors: the faithful rule drove
        ``lambda`` from 1e-2 to 1.5e-10 and the branch stopped moving, because
        once the displacement is below the working precision ``delta_grad`` is
        rounding noise and the ratio keeps coming out small.
        """
        if self.float64_stats:
            delta, delta_grad = to_float64(delta), to_float64(delta_grad)
        norm_delta = global_norm(delta)
        norm_delta_grad = global_norm(delta_grad)

        if self.min_probe_rel > 0.0:
            norm_theta = global_norm(
                [prm.detach() for prm in self._params]
            )
            if norm_delta.item() < self.min_probe_rel * max(
                norm_theta.item(), 1e-30
            ):
                self.n_probe_too_small += 1
                return lam, norm_delta, norm_delta_grad

        return (
            adaptive_lambda(
                lambda_k=lam,
                norm_delta=norm_delta,
                norm_delta_grad=norm_delta_grad,
                p=self.p,
                eps_denominator=self.eps_denominator,
                lambda_min=self.lambda_min,
                growth=self.lambda_growth,
                lambda_max=self.lambda_0,
            ),
            norm_delta,
            norm_delta_grad,
        )

    def apply(self) -> Dict[str, float]:
        """Phase 2: read ``F(v_k)``, update ``lambda``, then take the Adam step."""
        if self._u is None:
            raise RuntimeError("Adaptive.apply() called without a matching probe()")
        lam = self._probe_lambda
        f_u = self._f_u
        f_v = self._gather_grads()
        self.restore()
        self._u = self._f_u = None

        norm_f_u = global_norm(f_u)
        # u_k - v_k = lambda_k F(u_k)
        lambda_next, _, _ = self._next_lambda(
            lam,
            torch._foreach_mul(f_u, lam),
            torch._foreach_sub(f_u, f_v),
        )

        lr = self._adam_step(f_u, lam)

        self.lambda_k = lambda_next
        self.n_steps += 1

        norm_f_v = global_norm(f_v)
        denom = (norm_f_u * norm_f_v).item()
        return {
            "adaptive_lambda": lam,
            "adaptive_lambda_next": lambda_next,
            "adaptive_lambda_ratio": lam / self.lambda_0,
            "adaptive_lr": lr,
            "adaptive_grad_corr": (
                global_dot(f_u, f_v).item() / denom if denom > 0 else 0.0
            ),
            "adaptive_grad_norm": norm_f_u.item(),
        }

    # -----------------------------------------------------------------
    # "iterates" mode: one gradient, driven by the ordinary _optimizer_loop
    # -----------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None) -> Dict[str, float]:
        """One update.

        In ``"iterates"`` mode this is the whole update and needs no closure: the
        Lipschitz estimate comes from ``theta_k`` and ``theta_{k-1}``, so a single
        gradient per step is enough.

        In ``"probe"`` mode this is :meth:`probe` + ``closure()`` + :meth:`apply`,
        and ``closure`` must recompute the loss and call ``backward()`` on the
        **same minibatch** that produced the current ``.grad``. Drawing a fresh
        minibatch would make ``||F(u_k) - F(v_k)||`` measure sampling noise rather
        than the local Lipschitz constant, and collapse ``lambda_k``.
        """
        if self.lipschitz_from == "probe":
            if closure is None:
                raise ValueError(
                    "Adaptive.step() with lipschitz_from='probe' requires a closure "
                    "that recomputes the loss and calls backward() on the same "
                    "minibatch, to evaluate F(v_k)."
                )
            self.probe()
            try:
                with torch.enable_grad():
                    closure()
            except Exception:
                self.restore()
                self._u = self._f_u = None
                raise
            return self.apply()

        # --- "iterates" ---
        lam = self.lambda_k
        params = self._params
        f_u = self._gather_grads()
        theta_k = [prm.detach().clone() for prm in params]

        if self._prev_params is None:
            # first step: nothing to compare against, lambda stays put
            lambda_next = lam
            norm_diff = torch.zeros(())
        else:
            delta = torch._foreach_sub(theta_k, self._prev_params)
            delta_grad = torch._foreach_sub(f_u, self._prev_grads)
            lambda_next, _, norm_diff = self._next_lambda(lam, delta, delta_grad)

        # saved before the step: they are the pair (theta_k, F(theta_k))
        self._prev_params, self._prev_grads = theta_k, f_u

        lr = self._adam_step(f_u, lam)

        self.lambda_k = lambda_next
        self.n_steps += 1
        return {
            "adaptive_lambda": lam,
            "adaptive_lambda_next": lambda_next,
            "adaptive_lambda_ratio": lam / self.lambda_0,
            "adaptive_lr": lr,
            "adaptive_grad_norm": global_norm(f_u).item(),
            "adaptive_grad_diff_norm": norm_diff.item(),
        }

    # -----------------------------------------------------------------
    def zero_grad(self, set_to_none: bool = True):  # noqa: D102
        super().zero_grad(set_to_none=set_to_none)
        self._adam.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:  # noqa: D102
        return {
            "outer": super().state_dict(),
            # Adam's moments matter as much as lambda: losing them on resume is
            # what the pre-existing BenchMARL checkpoint gap used to do.
            "adam": self._adam.state_dict(),
            "lambda_k": self.lambda_k,
            "n_steps": self.n_steps,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):  # noqa: D102
        self.lambda_k = state_dict["lambda_k"]
        self.n_steps = state_dict["n_steps"]
        self._adam.load_state_dict(state_dict["adam"])
        super().load_state_dict(state_dict["outer"])


@dataclass
class AdaptiveConfig(OptimizerConfig):
    """Configuration dataclass for :class:`~benchmarl.optimizers.Adaptive`.

    Note the fields that are **absent**: ``beta``, ``gamma``, ``use_identity_dk``
    and ``beta_k_min/max`` all belong to Steps 4-6, which this variant does not
    use. Their absence is the definition of the variant.

    There is no ``lr`` either: the learning rate is ``lr_scale * lambda_k``, so
    ``experiment.lr`` is ignored (with a warning).
    """

    lambda_0: float = MISSING
    p: float = MISSING
    eps_denominator: float = MISSING
    lr_scale: float = MISSING
    lambda_min: float = MISSING
    lipschitz_from: str = MISSING
    reset_lambda_per_batch: bool = MISSING
    eps: Optional[float] = MISSING
    extra_kwargs: Optional[Dict[str, Any]] = MISSING
    # Departures from the paper. Only these apply here: Adaptive uses Step 3 and
    # nothing else, so there is no beta_k to fall back on, and it keeps Adam, so
    # it already has the preconditioner pc/pcvi drop. Neutral by default.
    lambda_growth: float = 1.0
    min_probe_rel: float = 0.0
    float64_stats: bool = False

    @staticmethod
    def associated_class() -> Type[torch.optim.Optimizer]:
        return Adaptive

    def requires_two_gradient_evals(self) -> bool:
        """Only ``"probe"`` needs the second backward pass."""
        return self.lipschitz_from == "probe"

    def uses_gradient_as_operator(self) -> bool:
        return True

    def _optimizer_kwargs(self, experiment_config) -> Dict[str, Any]:
        kwargs = dict(self.__dict__)
        if kwargs.get("eps", None) is None:
            kwargs["eps"] = experiment_config.adam_eps
        if kwargs.get("extra_kwargs", None) is None:
            kwargs["extra_kwargs"] = experiment_config.adam_extra_kwargs
        if experiment_config.lr is not None:
            warnings.warn(
                f"experiment.lr ({experiment_config.lr}) is ignored by "
                f"{type(self).__name__}: the learning rate is lr_scale * lambda_k, "
                f"driven by Step 3 of Algorithm 1. Starting at "
                f"{self.lr_scale * self.lambda_0:g} and decaying from there."
            )
        return kwargs


@dataclass
class AdaptivePlusConfig(AdaptiveConfig):
    """``adaptive`` with the two departures that apply to Step 3.

    ``beta_fallback`` and ``precond`` are absent by construction: this variant
    has no Step 5 to produce a ``beta_k``, and it keeps Adam, so the metric is
    already Adam's.

    What remains is the mechanism that killed a real run: ``lambda`` is
    non-increasing in the paper, so one ``||F(u_k)-F(v_k)||`` dominated by
    rounding lowers it permanently. ``lambda_growth`` lets it recover, bounded by
    ``lambda_0``; ``min_probe_rel`` stops the estimate being taken at all when
    the displacement is below the working precision.
    """
