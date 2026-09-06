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


def adaptive_lambda(
    lambda_k: float,
    norm_delta: torch.Tensor,
    norm_delta_grad: torch.Tensor,
    p: float,
    eps_denominator: float,
    lambda_min: float = 0.0,
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

    Returns:
        ``lambda_{k+1}``
    """
    if norm_delta_grad.item() > eps_denominator:
        lambda_next = min(p * (norm_delta / norm_delta_grad).item(), lambda_k)
    else:
        lambda_next = lambda_k
    return max(lambda_next, lambda_min)
