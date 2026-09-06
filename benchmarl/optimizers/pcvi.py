#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Projection-Contraction Method (PCVI), i.e. Algorithm 1 of

    Mai Thi Ngoc Ha, Nguyen Thi Trang, Truong Minh Tuyen,
    "New Explicit Algorithms for a Class of the Split Common Solution Problem",
    Mathematical Methods in the Applied Sciences, 2025.
    https://doi.org/10.1002/mma.11132

used as a parameter update rule for neural networks.

Symbol mapping (see ``KE_HOACH_IMPLEMENT_IPPO.md``)::

    u_k   -> the (flattened) parameters of one param group
    F(x)  -> the gradient of the loss at x, i.e. ``param.grad``
    lambda_k -> the adaptive learning rate
    p     -> the constant in (0, 1) of Step 1. The paper calls it ``theta``; it is
             renamed here to avoid the clash with the network parameters, which
             every RL reference (and BenchMARL) calls ``theta``.

Note that the ``Omega`` of the paper becomes the set of stationary points
``{theta : grad L(theta) = 0}``, and that the convergence result (Theorem 3.1)
requires ``F`` to be inverse strongly monotone (equivalently, by the
Baillon-Haddad theorem, ``L`` convex with Lipschitz gradient). Deep RL losses are
neither convex nor deterministic, so the theorem does **not** transfer: this is a
heuristic transplant of the update rule, and the diagnostics returned by
:meth:`Pcvi.step` are there to measure how badly the hypotheses are violated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, MISSING
from typing import Any, Dict, List, Optional, Type

import torch

from benchmarl.optimizers._lambda import adaptive_lambda, global_dot, global_norm
from benchmarl.optimizers.common import OptimizerConfig


class Pcvi(torch.optim.Optimizer):
    """Algorithm 1 of https://doi.org/10.1002/mma.11132 as a torch optimizer.

    One :meth:`step` performs, with ``F = grad L``:

    - Step 3: ``v_k = u_k - lambda_k F(u_k)`` and the update of ``lambda_k``
    - Step 4: ``d_k = u_k - v_k - lambda_k (F(u_k) - F(v_k))``
    - Step 5: ``beta_k = beta <u_k - v_k, d_k> / ||d_k||^2``
    - Step 6: ``u_{k+1} = u_k - beta_k d_k``

    ``F(v_k)`` is obtained by calling ``closure()``, which must recompute the loss
    and call ``backward()`` **on the same minibatch** that produced ``F(u_k)``.

    Args:
        params (iterable): the parameters to optimize. Must form a single param
            group, since ``lambda_k`` and ``beta_k`` are global scalars over ``H``.
        lambda_0 (float): the initial step ``lambda_0 = lambda > 0`` of Step 1.
            Since ``lambda_k`` is non-increasing, this is an upper bound on the
            learning rate for the whole run.
        beta (float): the ``beta`` in ``(0, 2)`` of Step 1.
        p (float): the constant in ``(0, 1)`` of Step 1 (called ``theta`` in the
            paper). Larger values let ``lambda_k`` stay larger.
        gamma (float): the ``gamma > 0`` of Step 1, used when ``||d_k|| == 0``.
            Note that it has **no numerical effect**: when ``d_k = 0`` the update
            ``u_{k+1} = u_k - gamma * 0`` leaves the iterate unchanged. It only
            appears in the lower bound of Remark 3.1 (iii). It is kept as a
            parameter for fidelity to the paper.
        lambda_min (float): floor for ``lambda_k``. **Not in the paper**; set it to
            ``0.0`` to be faithful. With stochastic gradients ``lambda_k`` can
            collapse to zero, which stalls training.
        beta_k_min (float): lower clamp for ``beta_k``. **Not in the paper**; set it
            to ``-inf`` to be faithful. With a non-convex loss
            ``<F(u_k), F(v_k)>`` can be negative, which turns Step 6 into an
            ascent step on the loss.
        beta_k_max (float): upper clamp for ``beta_k``. **Not in the paper**; set it
            to ``inf`` to be faithful.
        eps_denominator (float): the threshold below which ``||F(u_k) - F(v_k)||``
            is treated as zero in Step 3.
        use_identity_dk (bool): if ``True``, compute ``d_k`` as ``lambda_k F(v_k)``,
            which is algebraically identical to Step 4 (since
            ``u_k - v_k = lambda_k F(u_k)``) but saves one vector. If ``False``
            (default), Step 4 is applied literally.
        reset_lambda_per_batch (bool): if ``True``, :meth:`reset_lambda` sets
            ``lambda_k`` back to ``lambda_0``; the experiment calls it at the start
            of every collection round. **Not in the paper**.
        use_adaptive_lambda (bool): if ``False``, Step 3's update of ``lambda`` is
            skipped and ``lambda_k = lambda_0`` for the whole run, leaving only the
            projection-contraction machinery of Steps 4-6. This is the ``pc``
            variant; see ``KE_HOACH_IMPLEMENT_PC.md``.

            Two consequences of freezing ``lambda``:

            * the proof needs ``lambda_{k+1} ||F(u_k)-F(v_k)|| <= p ||u_k-v_k||``,
              which Step 3 guaranteed by construction. With a fixed ``lambda`` it
              holds iff ``lambda <= p / L``, so that becomes the user's problem.
            * inside that range it is *stronger*: the proof of Remark 3.1 (iii)
              needs ``1 - p (lambda_k/lambda_{k+1})^2 > 0``, and with a constant
              ``lambda`` the ratio is exactly 1 from the very first step, so
              ``beta_k >= min(gamma, beta/2)`` holds for **all** k, not just for
              ``k >= k0``.
        use_contraction (bool): if ``False``, Step 5 is skipped and ``beta_k = 1``,
            which reduces Step 6 to ``theta_{k+1} = theta_k - d_k =
            theta_k - lambda_k F(v_k)``: exactly Korpelevich's extragradient
            method (1976). See ``KE_HOACH_IMPLEMENT_EXTRAGRADIENT.md``.

            Measured on the paper's Example 5.1: extragradient is stable up to
            ``lambda L ~ 1`` and diverges past it, while the contraction lets the
            same method run at an effective step of ``1.197 / L`` and reach
            1e-8 in 26 steps against extragradient's 91. ``beta_k`` is therefore
            not a step multiplier -- simply scaling ``lambda`` by ``beta`` makes
            extragradient *worse* (183 steps).
    """

    def __init__(
        self,
        params,
        lambda_0: float,
        beta: float,
        p: float,
        gamma: float,
        lambda_min: float = 0.0,
        beta_k_min: float = -math.inf,
        beta_k_max: float = math.inf,
        eps_denominator: float = 1e-12,
        use_identity_dk: bool = False,
        reset_lambda_per_batch: bool = False,
        use_adaptive_lambda: bool = True,
        use_contraction: bool = True,
    ):
        if not lambda_0 > 0:
            raise ValueError(f"lambda_0 has to be > 0, got {lambda_0}")
        if not 0 < beta < 2:
            raise ValueError(f"beta has to be in (0, 2), got {beta}")
        if not 0 < p < 1:
            raise ValueError(f"p has to be in (0, 1), got {p}")
        if not gamma > 0:
            raise ValueError(f"gamma has to be > 0, got {gamma}")
        if lambda_min < 0:
            raise ValueError(f"lambda_min has to be >= 0, got {lambda_min}")
        if lambda_min > lambda_0:
            raise ValueError(
                f"lambda_min ({lambda_min}) has to be <= lambda_0 ({lambda_0})"
            )
        if beta_k_min > beta_k_max:
            raise ValueError(
                f"beta_k_min ({beta_k_min}) has to be <= beta_k_max ({beta_k_max})"
            )
        if not use_contraction and (beta != 1.0 or gamma != 1.0):
            raise ValueError(
                f"beta ({beta}) and gamma ({gamma}) belong to Step 5, which is "
                f"skipped when use_contraction=False (beta_k is pinned to 1). "
                f"Leave them at 1.0 so it is clear they play no part."
            )
        if not use_adaptive_lambda and lambda_min:
            raise ValueError(
                f"lambda_min ({lambda_min}) is meaningless with "
                f"use_adaptive_lambda=False: lambda never moves, so there is "
                f"nothing to put a floor under. Set it to 0.0."
            )

        super().__init__(
            params,
            defaults={
                "lambda_0": lambda_0,
                "beta": beta,
                "p": p,
                "gamma": gamma,
                "lambda_min": lambda_min,
                "beta_k_min": beta_k_min,
                "beta_k_max": beta_k_max,
                "eps_denominator": eps_denominator,
                "use_identity_dk": use_identity_dk,
                "reset_lambda_per_batch": reset_lambda_per_batch,
                "use_adaptive_lambda": use_adaptive_lambda,
                "use_contraction": use_contraction,
            },
        )
        if len(self.param_groups) != 1:
            raise ValueError(
                "Pcvi needs exactly one param group: lambda_k and beta_k are global "
                "quantities over the whole space H and cannot be split per tensor."
            )

        self.lambda_0 = lambda_0
        self.beta = beta
        self.p = p
        self.gamma = gamma
        self.lambda_min = lambda_min
        self.beta_k_min = beta_k_min
        self.beta_k_max = beta_k_max
        self.eps_denominator = eps_denominator
        self.use_identity_dk = use_identity_dk
        self.reset_lambda_per_batch = reset_lambda_per_batch
        self.use_adaptive_lambda = use_adaptive_lambda
        self.use_contraction = use_contraction

        self.lambda_k = lambda_0
        self.n_steps = 0
        # filled between probe() and apply()
        self._u: Optional[List[torch.Tensor]] = None
        self._f_u: Optional[List[torch.Tensor]] = None
        self._probe_lambda: float = lambda_0

    @property
    def _params(self) -> List[torch.Tensor]:
        return self.param_groups[0]["params"]

    def reset_lambda(self):
        """Sets ``lambda_k`` back to ``lambda_0``, if ``reset_lambda_per_batch``.

        Motivation: the objective changes at every collection round (a new batch,
        new advantages), so the Lipschitz estimate accumulated on the previous
        objective is not meaningful for the new one. **This is not in the paper.**
        """
        if self.reset_lambda_per_batch:
            self.lambda_k = self.lambda_0

    def _gather_grads(self) -> List[torch.Tensor]:
        return [
            (
                prm.grad.detach().clone()
                if prm.grad is not None
                else torch.zeros_like(prm)
            )
            for prm in self._params
        ]

    @torch.no_grad()
    def probe(self):
        """Phase 1 of a step: read ``F(u_k)``, then move the parameters to ``v_k``.

        The caller must have run ``backward()`` so that ``.grad`` holds ``F(u_k)``.
        After this call the parameters sit at the probe point, so a second
        ``forward``/``backward`` on the *same* minibatch yields ``F(v_k)``; then
        call :meth:`apply`.

        Split out of :meth:`step` so that several ``Pcvi`` instances (one per loss)
        can share a single second forward pass instead of one each.
        """
        params = self._params
        # The clone is mandatory: the next backward will overwrite .grad.
        self._u = [prm.detach().clone() for prm in params]
        self._f_u = self._gather_grads()
        self._probe_lambda = self.lambda_k
        # Step 3: v_k = u_k - lambda_k F(u_k)
        torch._foreach_add_(params, self._f_u, alpha=-self._probe_lambda)

    @torch.no_grad()
    def restore(self):
        """Puts the parameters back to ``u_k`` after a :meth:`probe`.

        Leaving them at ``v_k`` would silently corrupt the run, and any checkpoint
        taken afterwards.
        """
        if self._u is None:
            return
        for prm, u_i in zip(self._params, self._u):
            prm.data.copy_(u_i)

    @torch.no_grad()
    def apply(self) -> Dict[str, float]:
        """Phase 2 of a step: read ``F(v_k)`` from ``.grad`` and do Steps 3-6."""
        if self._u is None:
            raise RuntimeError("Pcvi.apply() called without a matching probe()")
        params = self._params
        lam = self._probe_lambda
        f_u, u = self._f_u, self._u
        f_v = self._gather_grads()
        self.restore()
        self._u = self._f_u = None
        del u

        # Step 3 (continued): lambda_{k+1}
        diff = torch._foreach_sub(f_u, f_v)  # F(u_k) - F(v_k)
        norm_diff = global_norm(diff)
        norm_f_u = global_norm(f_u)
        norm_uv = lam * norm_f_u  # ||u_k - v_k|| = lambda_k ||F(u_k)||
        if self.use_adaptive_lambda:
            lambda_next = adaptive_lambda(
                lambda_k=lam,
                norm_delta=norm_uv,
                norm_delta_grad=norm_diff,
                p=self.p,
                eps_denominator=self.eps_denominator,
                lambda_min=self.lambda_min,  # lambda_min is not in the paper
            )
        else:
            lambda_next = lam  # "pc": lambda is a constant

        # Step 4: d_k
        uv = torch._foreach_mul(f_u, lam)  # u_k - v_k
        if self.use_identity_dk:
            d = torch._foreach_mul(f_v, lam)  # d_k = lambda_k F(v_k)
        else:
            d = torch._foreach_sub(uv, torch._foreach_mul(diff, lam))

        # Step 5: beta_k
        norm_d = global_norm(d)
        # The paper branches on ``||d_k|| > 0``. In floating point, ``||d_k||`` can
        # be nonzero but at roundoff level, and then beta_k = 0/0 is pure noise
        # (though harmless: beta_k * d_k is still ~0). eps_denominator is the
        # floating-point rendering of "> 0".
        if not self.use_contraction:
            # extragradient: u_{k+1} = u_k - d_k = u_k - lambda_k F(v_k)
            beta_k_raw = 1.0
        elif norm_d.item() > self.eps_denominator:
            beta_k_raw = self.beta * global_dot(uv, d).item() / (norm_d**2).item()
        else:
            # d_k = 0, so u_{k+1} = u_k whatever gamma is (see the docstring).
            beta_k_raw = self.gamma
        if not math.isfinite(beta_k_raw):
            beta_k_raw = self.gamma
        beta_k = min(max(beta_k_raw, self.beta_k_min), self.beta_k_max)

        # Step 6: u_{k+1} = u_k - beta_k d_k
        torch._foreach_add_(params, d, alpha=-beta_k)

        # Step 7
        self.lambda_k = lambda_next
        self.n_steps += 1

        norm_f_v = global_norm(f_v)
        denom = (norm_f_u * norm_f_v).item()
        return {
            "pcvi_lambda": lam,
            "pcvi_lambda_next": lambda_next,
            "pcvi_beta_k": beta_k,
            "pcvi_beta_k_raw": beta_k_raw,
            "pcvi_beta_clamped": float(beta_k != beta_k_raw),
            "pcvi_effective_lr": beta_k * lam,
            "pcvi_grad_corr": (global_dot(f_u, f_v).item() / denom) if denom > 0 else 0.0,
            "pcvi_grad_norm": norm_f_u.item(),
            "pcvi_d_norm": norm_d.item(),
        }

    def step(self, closure=None) -> Dict[str, float]:
        """One full iteration of Algorithm 1.

        Args:
            closure (callable): recomputes the loss and calls ``backward()`` on
                the **same minibatch** that produced the current ``.grad``.
                Sampling a fresh minibatch here would make
                ``||F(u_k) - F(v_k)||`` measure sampling noise instead of the
                local Lipschitz constant, and collapse ``lambda_k``.
        """
        if closure is None:
            raise ValueError(
                "Pcvi.step() requires a closure that recomputes the loss and calls "
                "backward() on the same minibatch, to evaluate F(v_k)."
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

    def state_dict(self) -> Dict[str, Any]:  # noqa: D102
        state_dict = super().state_dict()
        # torch.optim.Optimizer.state_dict() only serializes self.state and
        # self.param_groups, so lambda_k would be lost on resume.
        state_dict["lambda_k"] = self.lambda_k
        state_dict["n_steps"] = self.n_steps
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):  # noqa: D102
        state_dict = dict(state_dict)
        self.lambda_k = state_dict.pop("lambda_k", self.lambda_0)
        self.n_steps = state_dict.pop("n_steps", 0)
        super().load_state_dict(state_dict)


@dataclass
class ExtragradientConfig(OptimizerConfig):
    """Korpelevich's extragradient method (1976).

    Algorithm 1 with ``beta_k = 1`` and a fixed ``lambda``::

        v_k       = theta_k - lambda F(theta_k)
        theta_k+1 = theta_k - lambda F(v_k)

    Note the fields that are **absent**: ``beta`` and ``gamma`` only appear in
    Step 5, and ``p`` and ``lambda_min`` only in Step 3. Their absence is the
    definition of the variant.

    ``lambda_0`` is the only real hyperparameter and has to be swept. The method
    is stable for ``lambda < 1/L`` and diverges past it, with no mechanism to
    recover -- see ``KE_HOACH_IMPLEMENT_EXTRAGRADIENT.md`` section 1.2.
    """

    lambda_0: float = MISSING
    eps_denominator: float = MISSING
    use_identity_dk: bool = MISSING

    @staticmethod
    def associated_class() -> Type[torch.optim.Optimizer]:
        return Pcvi

    def requires_two_gradient_evals(self) -> bool:
        return True

    def uses_gradient_as_operator(self) -> bool:
        return True

    def _optimizer_kwargs(self, experiment_config) -> Dict[str, Any]:
        return {
            **self.__dict__,
            "use_contraction": False,  # <- the definition of the variant
            "use_adaptive_lambda": False,
            # passed because Pcvi.__init__ requires them, unused here
            "beta": 1.0,
            "gamma": 1.0,
            "p": 0.5,
            "lambda_min": 0.0,
            "beta_k_min": -math.inf,
            "beta_k_max": math.inf,
            "reset_lambda_per_batch": False,
        }


@dataclass
class AdaptiveExtragradientConfig(OptimizerConfig):
    """Extragradient with Step 3's adaptive ``lambda``: ``beta_k = 1``, adaptive step.

    The fourth cell of the 2x2 grid
    ``{fixed, adaptive} lambda x {Step 5, beta_k = 1}``, whose other three cells
    are ``pc``, ``pcvi`` and ``extragradient``.
    """

    lambda_0: float = MISSING
    p: float = MISSING
    lambda_min: float = MISSING
    eps_denominator: float = MISSING
    use_identity_dk: bool = MISSING
    reset_lambda_per_batch: bool = MISSING

    @staticmethod
    def associated_class() -> Type[torch.optim.Optimizer]:
        return Pcvi

    def requires_two_gradient_evals(self) -> bool:
        return True

    def uses_gradient_as_operator(self) -> bool:
        return True

    def _optimizer_kwargs(self, experiment_config) -> Dict[str, Any]:
        return {
            **self.__dict__,
            "use_contraction": False,
            "use_adaptive_lambda": True,
            "beta": 1.0,
            "gamma": 1.0,
            "beta_k_min": -math.inf,
            "beta_k_max": math.inf,
        }


@dataclass
class PcConfig(OptimizerConfig):
    """Algorithm 1 with a fixed ``lambda``: only the projection-contraction part.

    Steps 4-6 are run exactly as in :class:`~benchmarl.optimizers.Pcvi`; Step 3's
    update of ``lambda`` is switched off. See ``KE_HOACH_IMPLEMENT_PC.md``.

    Note the fields that are **absent**: ``p`` only appears in the formula for
    ``lambda_{k+1}``, which is disabled, and ``lambda_min`` is a floor for a value
    that never moves. Their absence is the definition of the variant.

    ``lambda_0`` becomes the single most important hyperparameter and has to be
    swept: there is no mechanism left to find a step size on its own, and the
    convergence proof only applies while ``lambda_0 <= p / L``.
    """

    lambda_0: float = MISSING
    beta: float = MISSING
    gamma: float = MISSING
    beta_k_min: Optional[float] = MISSING
    beta_k_max: Optional[float] = MISSING
    eps_denominator: float = MISSING
    use_identity_dk: bool = MISSING
    reset_lambda_per_batch: bool = MISSING

    @staticmethod
    def associated_class() -> Type[torch.optim.Optimizer]:
        return Pcvi

    def requires_two_gradient_evals(self) -> bool:
        return True

    def uses_gradient_as_operator(self) -> bool:
        return True

    def _optimizer_kwargs(self, experiment_config) -> Dict[str, Any]:
        kwargs = dict(self.__dict__)
        if kwargs.get("beta_k_min", None) is None:
            kwargs["beta_k_min"] = -math.inf
        if kwargs.get("beta_k_max", None) is None:
            kwargs["beta_k_max"] = math.inf
        # the two knobs this variant does not expose
        kwargs["use_adaptive_lambda"] = False
        kwargs["use_contraction"] = True
        kwargs["lambda_min"] = 0.0
        kwargs["p"] = 0.5  # unused, but Pcvi.__init__ still range-checks it
        return kwargs


@dataclass
class PcviConfig(OptimizerConfig):
    """Configuration dataclass for :class:`~benchmarl.optimizers.Pcvi`."""

    lambda_0: float = MISSING
    beta: float = MISSING
    p: float = MISSING
    gamma: float = MISSING
    lambda_min: float = MISSING
    beta_k_min: Optional[float] = MISSING
    beta_k_max: Optional[float] = MISSING
    eps_denominator: float = MISSING
    use_identity_dk: bool = MISSING
    reset_lambda_per_batch: bool = MISSING
    use_adaptive_lambda: bool = MISSING
    use_contraction: bool = MISSING

    @staticmethod
    def associated_class() -> Type[torch.optim.Optimizer]:
        return Pcvi

    def requires_two_gradient_evals(self) -> bool:
        return True

    def uses_gradient_as_operator(self) -> bool:
        return True

    def _optimizer_kwargs(self, experiment_config) -> Dict[str, Any]:
        kwargs = dict(self.__dict__)
        # yaml has no way of spelling +-inf, so null means "no clamp"
        if kwargs.get("beta_k_min", None) is None:
            kwargs["beta_k_min"] = -math.inf
        if kwargs.get("beta_k_max", None) is None:
            kwargs["beta_k_max"] = math.inf
        return kwargs
