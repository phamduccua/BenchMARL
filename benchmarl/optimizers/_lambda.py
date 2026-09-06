#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Step 3 of Algorithm 1 (https://doi.org/10.1002/mma.11132), and the vector
helpers it needs.

Shared by :mod:`benchmarl.optimizers.pcvi` (which also uses Steps 4-6) and
:mod:`benchmarl.optimizers.adaptive` (which uses only this). Keeping a single
copy is what makes an Adaptive-vs-PCVI comparison meaningful: any difference
between the two is then attributable to Steps 4-6, not to two drifting
transcriptions of the same formula.
"""

from __future__ import annotations

from typing import List

import torch


def global_norm(tensors: List[torch.Tensor]) -> torch.Tensor:
    """The l2 norm of the concatenation of ``tensors``.

    This is the norm of the Hilbert space ``H`` of the paper: global over the
    whole parameter vector, *not* computed per tensor.
    """
    if not len(tensors):
        return torch.zeros(())
    return torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(t) for t in tensors])
    )


def global_dot(a: List[torch.Tensor], b: List[torch.Tensor]) -> torch.Tensor:
    """The inner product of the concatenation of ``a`` and of ``b``."""
    if not len(a):
        return torch.zeros(())
    return torch.stack([(x * y).sum() for x, y in zip(a, b)]).sum()


def global_dot_weighted(
    a: List[torch.Tensor], b: List[torch.Tensor], w: List[torch.Tensor]
) -> torch.Tensor:
    """``sum_i w_i a_i b_i``: the inner product in the metric given by ``w``.

    With ``w = P^-1`` this is ``<a, b>_{P^-1}``, the inner product Algorithm 1 has
    to be read in once the probe step is preconditioned. ``w = 1`` recovers
    :func:`global_dot`.
    """
    if not len(a):
        return torch.zeros(())
    return torch.stack([(x * y * u).sum() for x, y, u in zip(a, b, w)]).sum()


def global_norm_weighted(
    a: List[torch.Tensor], w: List[torch.Tensor]
) -> torch.Tensor:
    """``sqrt(sum_i w_i a_i^2)``, the norm of :func:`global_dot_weighted`."""
    if not len(a):
        return torch.zeros(())
    return torch.sqrt(
        torch.stack([(x * x * u).sum() for x, u in zip(a, w)]).sum().clamp_min(0)
    )


def to_float64(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """Upcast for the scalar statistics that drive ``lambda`` and ``beta_k``.

    Those statistics are norms and inner products of differences, and formula
    (3.1) computes ``a - a + b`` explicitly, so float32 cancellation shows up
    directly in them: measured 3.1e-5 of error against 3.6e-15 in float64. The
    cost is one temporary copy of the parameter vector, paid only on the
    reduction, not on the update.
    """
    return [t.double() for t in tensors]


def adaptive_lambda(
    lambda_k: float,
    norm_delta: torch.Tensor,
    norm_delta_grad: torch.Tensor,
    p: float,
    eps_denominator: float,
    lambda_min: float = 0.0,
    growth: float = 1.0,
    lambda_max: float = float('inf'),
) -> float:
    """Step 3 of Algorithm 1: the update of ``lambda``.

    .. code-block:: text

        lambda_{k+1} = min(p ||a - b|| / ||F(a) - F(b)||, lambda_k)   if the
                       denominator is nonzero, else lambda_k

    The ratio estimates ``p / L_local``, where ``L_local`` is the local Lipschitz
    constant of ``F`` along the segment ``[a, b]``. Two consequences that decide
    how ``lambda_0`` must be picked:

    * ``lambda_k`` only ever shrinks while ``lambda_k > p / L_local``
    * it settles at ``~ p / L_local``, so a ``lambda_0`` already below that
      leaves the mechanism inert and the method degenerates to a fixed step

    Args:
        lambda_k: the current step
        norm_delta: ``||a - b||``, the distance between the two probe points
        norm_delta_grad: ``||F(a) - F(b)||``
        p: the constant in ``(0, 1)`` of Step 1 (the paper calls it ``theta``)
        eps_denominator: below this, ``norm_delta_grad`` counts as zero
        lambda_min: floor for the result. **Not in the paper**; 0.0 is faithful.
            With stochastic gradients ``lambda_k`` can otherwise collapse to zero
            and stall training.
        growth: the largest factor by which ``lambda`` may *rise* in one step.
            **Not in the paper**; 1.0 is faithful and reproduces the ``min()``,
            i.e. a non-increasing ``lambda``. Above 1.0 the estimate is clamped
            to ``[lambda_k / growth, lambda_k * growth]``, so a step whose
            ``||F(u)-F(v)||`` was mostly noise can be recovered from.
        lambda_max: ceiling, applied only when ``growth > 1``. The paper needs
            none because ``lambda`` never rises; once it can, ``lambda_0`` stops
            being an upper bound on the step unless this puts it back.

    Returns:
        ``lambda_{k+1}``
    """
    if norm_delta_grad.item() > eps_denominator:
        estimate = p * (norm_delta / norm_delta_grad).item()
        if growth > 1.0:
            # Two-sided: lambda may recover, by at most `growth` per step. The
            # paper's min() makes lambda non-increasing, which is what its
            # convergence proof rests on -- and which, with stochastic gradients,
            # means a single unlucky ||F(u)-F(v)|| ruins lambda for the rest of
            # the run with no way back. Measured on matrixgame/rock_paper_scissors:
            # lambda fell from 1e-2 to 1.5e-10 and the branch stopped moving.
            # Bounding the per-step change keeps it tracking p/L_local without
            # letting it jump. This is the adaptive-extragradient variant
            # (Malitsky-Tam style), not Algorithm 1: say so in the write-up.
            lambda_next = min(max(estimate, lambda_k / growth), lambda_k * growth)
            # lambda_0 is an upper bound on the step for the whole run in the
            # paper, because lambda only decreases. Letting it grow removes
            # that guarantee unless it is put back explicitly.
            lambda_next = min(lambda_next, lambda_max)
        else:
            lambda_next = min(estimate, lambda_k)
    else:
        lambda_next = lambda_k
    return max(lambda_next, lambda_min)
