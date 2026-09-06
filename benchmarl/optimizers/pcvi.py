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

from benchmarl.optimizers._lambda import (
    adaptive_lambda,
    global_dot,
    global_dot_weighted,
    global_norm,
    global_norm_weighted,
    to_float64,
)
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
        lambda_growth: float = 1.0,
        min_probe_rel: float = 0.0,
        beta_fallback: Optional[float] = None,
        float64_stats: bool = False,
        precond: bool = False,
        precond_beta2: float = 0.999,
        precond_eps: float = 1e-8,
        precond_amsgrad: bool = True,
        precond_clamp: float = 1e3,
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
        if lambda_growth < 1.0:
            raise ValueError(
                f"lambda_growth has to be >= 1.0, got {lambda_growth}. 1.0 is the "
                f"paper's non-increasing lambda; above it, lambda may recover."
            )
        if not use_adaptive_lambda and lambda_growth != 1.0:
            raise ValueError(
                f"lambda_growth ({lambda_growth}) is meaningless with "
                f"use_adaptive_lambda=False: lambda never moves."
            )
        if min_probe_rel < 0:
            raise ValueError(f"min_probe_rel has to be >= 0, got {min_probe_rel}")
        if beta_fallback is not None and not 0 <= beta_fallback <= 2:
            raise ValueError(
                f"beta_fallback has to be in [0, 2] or None, got {beta_fallback}"
            )
        if precond and not 0 < precond_beta2 < 1:
            raise ValueError(
                f"precond_beta2 has to be in (0, 1), got {precond_beta2}"
            )
        if precond and precond_clamp <= 1:
            raise ValueError(
                f"precond_clamp has to be > 1, got {precond_clamp}"
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
                "lambda_growth": lambda_growth,
                "min_probe_rel": min_probe_rel,
                "beta_fallback": beta_fallback,
                "float64_stats": float64_stats,
                "precond": precond,
                "precond_beta2": precond_beta2,
                "precond_eps": precond_eps,
                "precond_amsgrad": precond_amsgrad,
                "precond_clamp": precond_clamp,
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
        self.lambda_growth = lambda_growth
        self.min_probe_rel = min_probe_rel
        self.beta_fallback = beta_fallback
        self.float64_stats = float64_stats
        self.precond = precond
        self.precond_beta2 = precond_beta2
        self.precond_eps = precond_eps
        self.precond_amsgrad = precond_amsgrad
        self.precond_clamp = precond_clamp

        self.lambda_k = lambda_0
        self.n_steps = 0
        # counters for the departures from the paper, so a run can report how
        # often each one actually fired instead of assuming it never did
        self.n_probe_too_small = 0
        self.n_beta_fallback = 0
        # diagonal preconditioner state, allocated lazily on the first probe
        self._p_diag: Optional[List[torch.Tensor]] = None
        self._v_ema: Optional[List[torch.Tensor]] = None
        self._v_max: Optional[List[torch.Tensor]] = None
        self._precond_steps = 0
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
        if self.precond:
            # Step 3 in the metric P: v_k = u_k - lambda_k P F(u_k). P is Adam's
            # diagonal, so the probe -- and every norm below -- lives in the
            # geometry the preconditioner induces. AMSGrad keeps v_hat monotone,
            # hence P convergent, which is what makes a fixed-metric reading of
            # the algorithm defensible at all.
            self._update_preconditioner(self._f_u)
            direction = torch._foreach_mul(self._f_u, self._p_diag)
        else:
            direction = self._f_u
        # Step 3: v_k = u_k - lambda_k F(u_k)
        torch._foreach_add_(params, direction, alpha=-self._probe_lambda)

    @torch.no_grad()
    def _update_preconditioner(self, grads: List[torch.Tensor]):
        """Adam's diagonal ``P = 1 / (sqrt(v_hat) + eps)``, clamped.

        Not in the paper. ``pc``/``pcvi`` drop Adam's per-coordinate scaling
        entirely, which is the handicap the ``sgd`` control exists to measure;
        this puts it back without changing Steps 4-6, by reading them in the
        ``P^-1`` metric instead of the Euclidean one.
        """
        if self._v_ema is None:
            self._v_ema = [torch.zeros_like(g) for g in grads]
            self._v_max = [torch.zeros_like(g) for g in grads]
            self._precond_steps = 0
        self._precond_steps += 1
        beta2 = self.precond_beta2
        bias = 1.0 - beta2**self._precond_steps
        p_diag = []
        for ema, vmax, g in zip(self._v_ema, self._v_max, grads):
            # Never feed the bias-corrected or max-ed value back into the EMA.
            ema.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            corrected = ema / bias
            if self.precond_amsgrad:
                torch.maximum(vmax, corrected, out=vmax)
                used = vmax
            else:
                used = corrected
            p_diag.append(
                (1.0 / (used.sqrt() + self.precond_eps)).clamp(
                    1.0 / self.precond_clamp, self.precond_clamp
                )
            )
        self._p_diag = p_diag

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

        # P = 1 everywhere when the preconditioner is off, so the expressions
        # below are the paper's verbatim in that case.
        p_diag = self._p_diag if self.precond else None
        weight = (  # the P^-1 metric; 1 means Euclidean
            [1.0 / pd for pd in p_diag] if p_diag is not None else None
        )

        # Step 3 (continued): lambda_{k+1}
        diff = torch._foreach_sub(f_u, f_v)  # F(u_k) - F(v_k)
        # u_k - v_k = lambda_k P F(u_k)
        uv = torch._foreach_mul(
            torch._foreach_mul(f_u, p_diag) if p_diag is not None else f_u, lam
        )
        if self.float64_stats:
            stat_uv, stat_diff = to_float64(uv), to_float64(diff)
            stat_w = to_float64(weight) if weight is not None else None
        else:
            stat_uv, stat_diff, stat_w = uv, diff, weight
        if stat_w is not None:
            norm_uv = global_norm_weighted(stat_uv, stat_w)
            # ||P dF||_{P^-1} = sqrt(sum P dF^2)
            norm_diff = global_norm_weighted(stat_diff, to_float64(p_diag)
                                             if self.float64_stats else p_diag)
        else:
            norm_uv = global_norm(stat_uv)
            norm_diff = global_norm(stat_diff)
        norm_f_u = global_norm(f_u)

        # The probe displacement has to be resolvable in the working precision:
        # below that, ||F(u)-F(v)|| is rounding noise rather than an operator
        # difference, the Lipschitz estimate is meaningless, and feeding it to
        # Step 3 drives lambda to zero and never recovers. Measured on
        # matrixgame/rock_paper_scissors: 1e-2 -> 1.5e-10 with the branch frozen.
        probe_too_small = False
        if self.min_probe_rel > 0.0:
            norm_theta = global_norm(params)
            probe_too_small = (
                norm_uv.item() < self.min_probe_rel * max(norm_theta.item(), 1e-30)
            )
            if probe_too_small:
                self.n_probe_too_small += 1

        if self.use_adaptive_lambda and not probe_too_small:
            lambda_next = adaptive_lambda(
                lambda_k=lam,
                norm_delta=norm_uv,
                norm_delta_grad=norm_diff,
                p=self.p,
                eps_denominator=self.eps_denominator,
                lambda_min=self.lambda_min,  # lambda_min is not in the paper
                growth=self.lambda_growth,  # nor is growth
                lambda_max=self.lambda_0,  # keep lambda_0 an upper bound
            )
        else:
            # "pc": lambda is a constant. Or: the probe was below the noise floor,
            # so lambda is held rather than updated from a meaningless ratio.
            lambda_next = lam

        # Step 4: d_k
        if self.use_identity_dk:
            # d_k = lambda_k P F(v_k)
            d = torch._foreach_mul(
                torch._foreach_mul(f_v, p_diag) if p_diag is not None else f_v, lam
            )
        else:
            scaled_diff = (
                torch._foreach_mul(diff, p_diag) if p_diag is not None else diff
            )
            d = torch._foreach_sub(uv, torch._foreach_mul(scaled_diff, lam))

        # Step 5: beta_k
        stat_d = to_float64(d) if self.float64_stats else d
        norm_d = (
            global_norm_weighted(stat_d, stat_w) if stat_w is not None
            else global_norm(stat_d)
        )
        # The paper branches on ``||d_k|| > 0``. In floating point, ``||d_k||`` can
        # be nonzero but at roundoff level, and then beta_k = 0/0 is pure noise
        # (though harmless: beta_k * d_k is still ~0). eps_denominator is the
        # floating-point rendering of "> 0".
        if not self.use_contraction:
            # extragradient: u_{k+1} = u_k - d_k = u_k - lambda_k F(v_k)
            beta_k_raw = 1.0
        elif norm_d.item() > self.eps_denominator:
            numerator = (
                global_dot_weighted(stat_uv, stat_d, stat_w) if stat_w is not None
                else global_dot(stat_uv, stat_d)
            )
            beta_k_raw = self.beta * numerator.item() / (norm_d**2).item()
        else:
            # d_k = 0, so u_{k+1} = u_k whatever gamma is (see the docstring).
            beta_k_raw = self.gamma
        if not math.isfinite(beta_k_raw):
            beta_k_raw = self.gamma
        beta_k = beta_k_raw
        used_fallback = False
        if self.beta_fallback is not None and beta_k_raw < 0.0:
            # beta_k < 0 means <u_k - v_k, d_k> < 0: the inverse-strong-monotonicity
            # the contraction rests on does not hold at this iterate, and the
            # faithful step would move AGAINST d_k. Falling back to beta_k = 1 is
            # a plain extragradient step, which needs no such assumption -- a
            # weaker method rather than a wrong one. How often it fires is itself
            # the measurement of how badly ISM is violated: see n_beta_fallback.
            beta_k = self.beta_fallback
            used_fallback = True
            self.n_beta_fallback += 1
        beta_k = min(max(beta_k, self.beta_k_min), self.beta_k_max)

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
            # The departures from the paper, each reported so a run can say how
            # often it actually mattered rather than leaving it to be assumed.
            "pcvi_probe_too_small": float(probe_too_small),
            "pcvi_beta_fallback": float(used_fallback),
            "pcvi_lambda_grew": float(lambda_next > lam),
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
    # Departures from the paper. Only these two mean anything here: lambda is
    # frozen, so lambda_growth and min_probe_rel -- both of which act on Step 3 --
    # have nothing to act on. Neutral defaults keep pc.yaml faithful.
    beta_fallback: Optional[float] = None
    float64_stats: bool = False
    precond: bool = False
    precond_beta2: float = 0.999
    precond_eps: float = 1e-8
    precond_amsgrad: bool = True
    precond_clamp: float = 1e3

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


@dataclass
class PcviPlusConfig(PcviConfig):
    """Algorithm 1 with the four departures that a real run showed it needs.

    Every one of them is off in :class:`PcviConfig`, whose defaults stay faithful
    to the paper, and each is a separate flag here so its contribution can be
    measured rather than assumed. What follows is why each exists, with the
    measurement that motivated it -- all from
    ``matrixgame/rock_paper_scissors``, 6 branches x 10 seeds, 500K frames:

    1. ``lambda_growth`` -- the paper's ``lambda_{k+1} = min(..., lambda_k)`` is
       non-increasing, so with stochastic gradients one bad
       ``||F(u_k)-F(v_k)||`` lowers ``lambda`` for good. Measured: ``lambda``
       fell from 1e-2 to **1.5e-10** and the branch stopped moving entirely
       (effective step 3e-10 against a gradient norm of 0.42). Allowing a bounded
       rise lets it recover; ``lambda_0`` is kept as a ceiling so it remains an
       upper bound on the step, as it is in the paper.
    2. ``min_probe_rel`` -- once ``||u_k - v_k||`` falls below the working
       precision relative to ``||theta||``, ``F(u_k) - F(v_k)`` is rounding noise
       and the Lipschitz estimate built from it is meaningless. Feeding it to
       Step 3 is what drives the collapse above. Below the threshold ``lambda``
       is held instead of updated.
    3. ``beta_fallback`` -- ``beta_k < 0`` means ``<u_k - v_k, d_k> < 0``: the
       inverse strong monotonicity the contraction rests on does not hold there,
       and the faithful step moves *against* ``d_k``. Measured: player_1 ran at
       ``beta_k = -0.22`` while player_0 sat at +1.95. Falling back to
       ``beta_k = 1`` is a plain extragradient step, which needs no such
       assumption. How often it fires is the measurement of the violation.
    4. ``precond`` -- ``pc``/``pcvi`` drop Adam's per-coordinate scaling, which
       is the handicap the ``sgd`` control exists to isolate. This runs Steps 3-6
       in the metric induced by Adam's diagonal ``P``, with AMSGrad keeping
       ``v_hat`` monotone so ``P`` converges. Steps 4-6 are unchanged; only the
       inner product they are read in changes.

    Points 1-3 make Algorithm 1 survive stochastic gradients. Point 4 extends it.
    All four are departures from the paper and must be declared as such.
    """

    lambda_growth: float = MISSING
    min_probe_rel: float = MISSING
    beta_fallback: Optional[float] = MISSING
    float64_stats: bool = MISSING
    precond: bool = MISSING
    precond_beta2: float = MISSING
    precond_eps: float = MISSING
    precond_amsgrad: bool = MISSING
    precond_clamp: float = MISSING


@dataclass
class PcPlusConfig(PcConfig):
    """``pc`` with the departures that mean anything for a frozen ``lambda``.

    Only two of the four apply. ``lambda_growth`` and ``min_probe_rel`` both act
    on Step 3, which this variant switches off, so there is nothing for them to
    do. What is left is real:

    * ``beta_fallback`` -- ``pc`` runs Step 5, so ``beta_k`` can go negative, and
      measured on ``matrixgame/rock_paper_scissors`` it did: player_1 sat at
      ``beta_k = -0.22`` while player_0 was at +1.95.
    * ``precond`` -- ``pc`` drops Adam's per-coordinate scaling, which is exactly
      the handicap the ``sgd`` control exists to isolate.

    Paired with :class:`PcviPlusConfig`, this keeps the 2x2 grid intact under the
    improvements: ``{lambda fixed, lambda adaptive} x {Step 5, beta_k = 1}``
    still differ by one mechanism each, not by which departures are switched on.
    """

