#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""The four departures from Algorithm 1 in :class:`PcviPlusConfig`.

Each test reproduces the failure that motivated one flag, then shows the flag
stops it. The failures are not hypothetical: they were measured on
``matrixgame/rock_paper_scissors`` at 500K frames, where ``lambda`` collapsed
from 1e-2 to 1.5e-10 and one player ran at ``beta_k = -0.22``.
"""

import math

import pytest
import torch

from benchmarl.optimizers import PcviConfig, PcviPlusConfig
from benchmarl.optimizers.pcvi import Pcvi


def _run(steps: int, operator, lambda_0: float = 0.5, **kwargs):
    """Drive Pcvi by hand with a chosen operator F, returning its diagnostics."""
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(64) * 0.1)
    optimizer = Pcvi([w], lambda_0=lambda_0, beta=1.95, p=0.5, gamma=1.0, **kwargs)
    infos = []
    for k in range(steps):
        w.grad = operator(w.detach(), k)
        optimizer.probe()
        w.grad = operator(w.detach(), k)
        infos.append(optimizer.apply())
    return optimizer, infos


# --------------------------------------------------------- 1. lambda recovers


def _noisy_once(scale: float):
    """A benign operator with a single pathological step at k == 3.

    One step where ``F`` changes far more than the displacement is enough to
    ratchet the paper's `lambda` down for good, since it can only decrease.
    """

    def operator(w, k):
        if k == 3:
            return torch.randn_like(w) * scale
        return w * 0.05

    return operator


def test_faithful_lambda_never_recovers_from_one_bad_step():
    _, infos = _run(12, _noisy_once(50.0))
    lambdas = [info["pcvi_lambda_next"] for info in infos]
    assert lambdas[4] < lambdas[2] / 2, "the bad step should have cut lambda"
    assert all(a >= b for a, b in zip(lambdas, lambdas[1:])), "must be monotone"
    assert lambdas[-1] == pytest.approx(lambdas[4], rel=1e-6), "and never recover"


def test_growth_lets_lambda_recover():
    """The operator is benign from k == 4 on, so a lambda that may rise climbs
    back towards its ceiling while the faithful one stays where it was knocked."""
    _, faithful = _run(12, _noisy_once(50.0))
    _, grown = _run(12, _noisy_once(50.0), lambda_growth=1.5)
    faithful_lambdas = [info["pcvi_lambda_next"] for info in faithful]
    grown_lambdas = [info["pcvi_lambda_next"] for info in grown]

    assert min(grown_lambdas) < grown_lambdas[0], "the bad step still cuts lambda"
    assert grown_lambdas[-1] > min(grown_lambdas), "and it has to climb back"
    assert grown_lambdas[-1] == pytest.approx(grown_lambdas[0]), (
        "all the way back to the ceiling, since the operator is benign again"
    )
    assert grown_lambdas[-1] > faithful_lambdas[-1] * 2, (
        "recovery is the whole point: the faithful run stays where it was knocked"
    )
    assert any(info["pcvi_lambda_grew"] for info in grown)
    # The drop itself is bounded too: growth clamps both directions, so one bad
    # step can no longer take lambda anywhere it likes.
    assert min(grown_lambdas) > min(faithful_lambdas)


def test_lambda_never_exceeds_lambda_0_even_when_it_may_grow():
    """lambda_0 is an upper bound on the step in the paper because lambda only
    falls. Once it can rise, that has to be put back explicitly."""
    _, infos = _run(30, lambda w, k: w * 1e-6, lambda_0=0.5, lambda_growth=2.0)
    assert max(info["pcvi_lambda_next"] for info in infos) <= 0.5 + 1e-12


# ------------------------------------------- 2. refuse a probe below the noise


def test_probe_below_the_noise_floor_holds_lambda_instead_of_collapsing():
    """F identical at both points: ``||F(u)-F(v)||`` is pure rounding.

    The faithful rule divides by that noise and ratchets lambda down every step;
    the guard notices the probe is unresolvable and holds lambda instead.
    """
    constant = lambda w, k: torch.full_like(w, 1e-9)  # noqa: E731

    _, faithful = _run(25, constant, lambda_0=1e-3)
    _, guarded = _run(25, constant, lambda_0=1e-3, min_probe_rel=1e-5)

    assert guarded[-1]["pcvi_lambda_next"] >= faithful[-1]["pcvi_lambda_next"]
    assert any(info["pcvi_probe_too_small"] for info in guarded)
    assert guarded[-1]["pcvi_lambda_next"] == pytest.approx(1e-3, rel=1e-9), (
        "with an unresolvable probe lambda should not move at all"
    )


def test_probe_guard_counts_how_often_it_fired():
    optimizer, _ = _run(
        20, lambda w, k: torch.full_like(w, 1e-9), lambda_0=1e-3, min_probe_rel=1e-5
    )
    assert optimizer.n_probe_too_small > 0


# ------------------------------------------------ 3. beta_k < 0 -> extragradient


def _ism_violating(w, k):
    """``F(w) = 3w`` with ``lambda_0 = 0.5``, so ``lambda * c = 1.5 > 1``.

    The probe overshoots the origin, ``F(v_k)`` comes back with the opposite
    sign, and ``<u_k - v_k, d_k> = lambda^2 <F(u_k), F(v_k)>`` goes negative --
    which is exactly the regime the measured ``beta_k = -0.22`` on player_1 sat
    in. The faithful rule then steps *against* ``d_k``.
    """
    return w * 3.0


def test_negative_beta_moves_against_d_k_when_faithful():
    _, infos = _run(4, _ism_violating, lambda_0=0.5)
    assert infos[0]["pcvi_beta_k"] == pytest.approx(-3.9, rel=1e-3), (
        "this operator is supposed to violate ISM on the first step"
    )


def test_beta_fallback_replaces_the_reversed_step_with_extragradient():
    optimizer, infos = _run(4, _ism_violating, lambda_0=0.5, beta_fallback=1.0)
    assert all(info["pcvi_beta_k"] >= 0 for info in infos)
    assert optimizer.n_beta_fallback == 1
    fired = next(i for i in infos if i["pcvi_beta_fallback"])
    assert fired["pcvi_beta_k"] == pytest.approx(1.0)
    assert fired["pcvi_beta_k_raw"] < 0, "the raw value is still reported"


# ------------------------------------------------------ 4. Adam's metric


def test_preconditioner_changes_the_step_but_not_the_protocol():
    plain, _ = _run(10, lambda w, k: w * 0.05)
    precond, infos = _run(10, lambda w, k: w * 0.05, precond=True)
    a = plain.param_groups[0]["params"][0].detach()
    b = precond.param_groups[0]["params"][0].detach()
    assert not torch.allclose(a, b), "the metric has to change where it lands"
    assert all(math.isfinite(info["pcvi_beta_k"]) for info in infos)


def test_amsgrad_keeps_the_preconditioner_monotone():
    """P has to converge for a fixed-metric reading to mean anything."""
    optimizer, _ = _run(20, lambda w, k: w * (0.05 + 0.2 * (k % 3)), precond=True)
    assert optimizer._v_max is not None
    for ema, vmax in zip(optimizer._v_ema, optimizer._v_max):
        assert torch.all(vmax >= 0)
        assert torch.all(vmax + 1e-12 >= ema / 1.0), "v_max must dominate the EMA"


def test_preconditioner_is_clamped_both_ways():
    optimizer, _ = _run(8, lambda w, k: w * 1e-8, precond=True, precond_clamp=10.0)
    for block in optimizer._p_diag:
        assert torch.all(block <= 10.0 + 1e-6)
        assert torch.all(block >= 1.0 / 10.0 - 1e-6)


# ---------------------------------------------------------------- the config


def test_pcvi_plus_yaml_turns_all_four_on():
    config = PcviPlusConfig.get_from_yaml()
    assert config.lambda_growth > 1.0
    assert config.min_probe_rel > 0.0
    assert config.beta_fallback == 1.0
    assert config.precond is True
    assert config.float64_stats is True


def test_pcvi_yaml_stays_faithful():
    """The point of the split: pcvi remains the paper, pcvi_plus is the variant."""
    config = PcviConfig.get_from_yaml()
    assert getattr(config, "lambda_growth", 1.0) == 1.0
    assert getattr(config, "min_probe_rel", 0.0) == 0.0
    assert getattr(config, "beta_fallback", None) is None
    assert getattr(config, "precond", False) is False
    assert config.lambda_min == 0.0
    assert config.beta_k_min is None and config.beta_k_max is None


def test_pcvi_plus_still_needs_two_gradients_and_no_clipping():
    config = PcviPlusConfig.get_from_yaml()
    assert config.requires_two_gradient_evals()
    assert config.uses_gradient_as_operator()


def test_all_four_off_reproduces_pcvi_exactly():
    """pcvi_plus with every flag neutral must be bit-identical to pcvi."""
    operator = lambda w, k: w * 0.05 + 0.01  # noqa: E731
    a, _ = _run(15, operator)
    b, _ = _run(
        15,
        operator,
        lambda_growth=1.0,
        min_probe_rel=0.0,
        beta_fallback=None,
        float64_stats=False,
        precond=False,
    )
    for x, y in zip(a.param_groups[0]["params"], b.param_groups[0]["params"]):
        assert torch.equal(x.detach(), y.detach())
