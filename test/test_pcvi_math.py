#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Tier-1 tests for :class:`~benchmarl.optimizers.Pcvi`: pure maths, no RL.

These reproduce Example 5.1 of

    Mai Thi Ngoc Ha, Nguyen Thi Trang, Truong Minh Tuyen,
    "New Explicit Algorithms for a Class of the Split Common Solution Problem",
    Mathematical Methods in the Applied Sciences, 2025,
    https://doi.org/10.1002/mma.11132

which is the strongest available evidence that Algorithm 1 is implemented
correctly, independently of anything BenchMARL does.
"""

import math

import pytest
import torch

from benchmarl.optimizers import (
    AdaptiveExtragradientConfig,
    ExtragradientConfig,
    PcConfig,
    Pcvi,
    PcviConfig,
)

# Algorithm 1 subtracts two nearly-equal vectors in Step 4: formula (3.1) reads
# ``u_k - v_k - lambda_k (F(u_k) - F(v_k))`` where ``u_k - v_k = lambda_k F(u_k)``,
# so it computes ``a - a + b`` explicitly and cancels catastrophically in float32.
# The maths tests therefore run in float64; test_float32_precision_loss below
# documents what float32 costs.
DTYPE = torch.float64

# ----------------------------------------------------------------------------
# Example 5.1 of the paper (page 9-10)
# ----------------------------------------------------------------------------
# h (on R^5) and h_1..h_4 (on R^2, R^3, R^4, R^6) are all of the form
#   h_i(y) = 1/2 (<c_i, y> - b_i)^2
# so that grad h_i(y) = (<c_i, y> - b_i) c_i.

H_C = torch.tensor([1.0, -1.0, 1.0, 2.0, -1.0])  # h(x)  = 1/2(x1-x2+x3+2x4-x5-1)^2
H_B = 1.0

HS_C = [
    torch.tensor([2.0, 1.0]),  # h1(y) = 1/2(2y1+y2-3)^2
    torch.tensor([1.0, -2.0, 1.0]),  # h2(z) = 1/2(z1-2z2+z3-3)^2
    torch.tensor([1.0, -1.0, 1.0, 1.0]),  # h3(w) = 1/2(w1-w2+w3+w4-1)^2
    torch.tensor([1.0, 2.0, -1.0, 1.0, 1.0, 1.0]),  # h4(r) = 1/2(r1+2r2-r3+r4+r5+r6)^2
]
HS_B = [3.0, 3.0, 1.0, 0.0]

T1 = torch.tensor(
    [
        [1.0, -3.0, 2.0, 1.0, -1.0],
        [0.0, 5.0, -1.0, 0.0, 0.0],
    ]
)
T2 = torch.tensor(
    [
        [1.0, 2.0, -2.0, 0.0, 1.0],
        [3.0, -1.0, 1.0, 2.0, 1.0],
        [6.0, -3.0, 7.0, 2.0, 0.0],
    ]
)
T3 = torch.tensor(
    [
        [1.0, 1.0, 2.0, 1.0, -3.0],
        [2.0, 1.0, -1.0, 1.0, 0.0],
        [1.0, -2.0, 1.0, 0.0, 2.0],
        [0.0, 3.0, -3.0, -2.0, 1.0],
    ]
)
T4 = torch.tensor(
    [
        [-1.0, 1.0, 2.0, -2.0, 0.0],
        [1.0, -1.0, 1.0, 1.0, 2.0],
        [3.0, 0.0, -2.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, 1.0, 1.0],
        [2.0, 1.0, 0.0, 1.0, -1.0],
        [0.0, -1.0, -7.0, 3.0, -4.0],
    ]
)
TS = [T1, T2, T3, T4]

U0 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # u_0 of the paper


def _objective(x: torch.Tensor) -> torch.Tensor:
    """L(x) = h(x) + sum_i h_i(T_i x), so that grad L = F of the paper."""
    out = 0.5 * (H_C.to(x.dtype) @ x - H_B) ** 2
    for t, c, b in zip(TS, HS_C, HS_B):
        out = out + 0.5 * (c.to(x.dtype) @ (t.to(x.dtype) @ x) - b) ** 2
    return out


def _residuals(x: torch.Tensor) -> torch.Tensor:
    """The quantities ``A_i(T_i x) - f_i`` of Theorem 3.1, eq. (3.8)."""
    return torch.stack(
        [H_C.to(x.dtype) @ x - H_B]
        + [c.to(x.dtype) @ (t.to(x.dtype) @ x) - b for t, c, b in zip(TS, HS_C, HS_B)]
    )


def _omega_point(a: float, b: float, c: float) -> torch.Tensor:
    """A point of ``Omega = {(a, 2c-b+1, b, c, a+2b-2)}`` (page 10)."""
    return torch.tensor([a, 2 * c - b + 1, b, c, a + 2 * b - 2])


def _run(optimizer_kwargs, n_steps, u0=U0, dtype=None):
    """Runs Pcvi on the Example 5.1 objective, returning the iterates."""
    dtype = DTYPE if dtype is None else dtype
    x = torch.nn.Parameter(u0.clone().to(dtype))
    opt = Pcvi([x], **optimizer_kwargs)

    def closure():
        opt.zero_grad()
        _objective(x).backward()

    iterates, infos = [x.detach().clone()], []
    for _ in range(n_steps):
        closure()  # F(u_k)
        infos.append(opt.step(closure))  # needs F(v_k)
        iterates.append(x.detach().clone())
    return iterates, infos


PAPER_KWARGS = dict(lambda_0=1.0, beta=1.95, p=0.5, gamma=1.0)


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


def test_omega_is_the_solution_set():
    """Sanity check on the data: every point of Omega must be a solution.

    This validates the T_i matrices transcribed from page 10 against the solution
    set the paper states, independently of the optimizer.
    """
    for a, b, c in [(0.0, 0.0, 0.0), (1.0, -2.0, 3.0), (-1.5, 0.5, 2.25)]:
        x = _omega_point(a, b, c).to(DTYPE)
        assert torch.allclose(
            _residuals(x), torch.zeros(5, dtype=DTYPE), atol=1e-9
        ), (a, b, c)


def test_converges_to_omega():
    """Theorem 3.1: u_k converges to a point of Omega."""
    iterates, infos = _run(PAPER_KWARGS, n_steps=300)

    final_residual = _residuals(iterates[-1]).norm().item()
    assert final_residual < 1e-10, f"residual {final_residual}"

    # and the limit is reached, i.e. the iterates stop moving
    assert (iterates[-1] - iterates[-2]).norm().item() < 1e-12


def test_error_decreases_monotonically_after_k0():
    """||u_k - z|| is non-increasing (proof of Thm 3.1, eq. (3.4)-(3.6)).

    Only from ``k0`` on: eq. (3.6) needs ``beta_k > 0``, which Remark 3.1 (iii)
    only guarantees for ``k >= k0``. With the paper's own parameters ``beta_0``
    is in fact negative, see :func:`test_first_step_can_move_away`.
    """
    iterates, infos = _run(PAPER_KWARGS, n_steps=200)
    k0 = 1 + max(k for k, i in enumerate(infos) if i["pcvi_beta_k"] <= 0.0)
    limit = iterates[-1]
    errors = [(u - limit).norm().item() for u in iterates]
    for k in range(k0, len(errors) - 1):
        assert errors[k + 1] <= errors[k] + 1e-9, f"increased at step {k}"


def test_first_step_can_move_away():
    """With the paper's parameters beta_0 < 0 and step 0 increases the error.

    Not a bug: Remark 3.1 (iii) is asymptotic, and lambda_0 = 1 is far above the
    lambda the algorithm settles on (~1e-2). Worth knowing before reading a
    training curve that starts by getting worse.
    """
    iterates, infos = _run(PAPER_KWARGS, n_steps=3)
    limit = _run(PAPER_KWARGS, n_steps=200)[0][-1]
    assert infos[0]["pcvi_beta_k"] < 0
    assert (iterates[1] - limit).norm() > (iterates[0] - limit).norm()


def test_lambda_is_non_increasing_and_bounded_below():
    """Remark 3.1 (ii): lambda_k decreases but stays bounded away from 0."""
    _, infos = _run(PAPER_KWARGS, n_steps=200)
    lambdas = [i["pcvi_lambda"] for i in infos]
    for k in range(len(lambdas) - 1):
        assert lambdas[k + 1] <= lambdas[k] + 1e-12, f"increased at step {k}"
    assert lambdas[-1] > 0.0


def test_beta_k_lower_bound():
    """Remark 3.1 (iii): beta_k >= min(gamma, beta/2) for k >= k0.

    Only checked while ``||d_k||`` is above roundoff: once the iterates have
    converged, ``d_k`` sits at machine precision and ``beta_k`` is 0/0 noise.
    Harmless (``beta_k * d_k`` is still ~0) but not worth asserting on.
    """
    _, infos = _run(PAPER_KWARGS, n_steps=100)
    bound = min(PAPER_KWARGS["gamma"], PAPER_KWARGS["beta"] / 2)
    checked = 0
    for k, info in enumerate(infos):
        if k == 0 or info["pcvi_d_norm"] < 1e-6:
            continue
        assert info["pcvi_beta_k"] >= bound - 1e-6, f"beta_k too small at step {k}"
        checked += 1
    assert checked > 5, "the test did not actually check anything"


def test_identity_dk_matches_formula_3_1():
    """d_k = lambda_k F(v_k) is algebraically identical to formula (3.1)."""
    literal, _ = _run({**PAPER_KWARGS, "use_identity_dk": False}, n_steps=100)
    identity, _ = _run({**PAPER_KWARGS, "use_identity_dk": True}, n_steps=100)
    for k, (a, b) in enumerate(zip(literal, identity)):
        assert torch.allclose(a, b, atol=1e-12), f"diverged at step {k}"


def test_float32_precision_loss():
    """Formula (3.1) cancels catastrophically in float32; the identity does not.

    The two routes drift apart by ~1e-5 on this problem in float32, while in
    float64 they agree to 1e-12 (test above). Relevant when choosing
    ``use_identity_dk`` for a float32 training run.
    """
    literal, _ = _run(
        {**PAPER_KWARGS, "use_identity_dk": False}, n_steps=100, dtype=torch.float32
    )
    identity, _ = _run(
        {**PAPER_KWARGS, "use_identity_dk": True}, n_steps=100, dtype=torch.float32
    )
    drift = max((a - b).norm().item() for a, b in zip(literal, identity))
    assert drift > 1e-7, "float32 unexpectedly lossless; this warning may be stale"
    # but both still land on the solution
    for route in (literal, identity):
        assert _residuals(route[-1].double()).norm().item() < 1e-3


@pytest.mark.parametrize("u0_seed", [0, 1, 2])
def test_converges_from_any_start(u0_seed):
    torch.manual_seed(u0_seed)
    u0 = (torch.randn(5) * 3).to(DTYPE)
    iterates, _ = _run(PAPER_KWARGS, n_steps=400, u0=u0)
    assert _residuals(iterates[-1]).norm().item() < 1e-10


def test_stationary_point_does_not_move():
    """F(u_k) = 0 => v_k = u_k, d_k = 0, and the iterate stays put (no NaN)."""
    x0 = _omega_point(1.0, -2.0, 3.0)
    iterates, infos = _run(PAPER_KWARGS, n_steps=5, u0=x0)
    for u in iterates:
        assert torch.allclose(u, x0.to(u.dtype), atol=1e-12)
        assert not torch.isnan(u).any()
    # gamma branch was taken: d_k = 0
    assert infos[0]["pcvi_d_norm"] == pytest.approx(0.0, abs=1e-8)


def test_step_without_closure_raises():
    x = torch.nn.Parameter(U0.clone().to(DTYPE))
    opt = Pcvi([x], **PAPER_KWARGS)
    _objective(x).backward()
    with pytest.raises(ValueError, match="closure"):
        opt.step()


def test_parameters_restored_when_closure_raises():
    """A failing closure must not leave the parameters at the probe point v_k."""
    x = torch.nn.Parameter(U0.clone().to(DTYPE))
    opt = Pcvi([x], **PAPER_KWARGS)
    _objective(x).backward()

    def bad_closure():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        opt.step(bad_closure)
    assert torch.allclose(x.detach(), U0.to(DTYPE), atol=1e-12)


def test_invalid_hyperparameters():
    x = torch.nn.Parameter(U0.clone().to(DTYPE))
    for bad in [
        {"beta": 2.0},
        {"beta": 0.0},
        {"p": 1.0},
        {"p": 0.0},
        {"lambda_0": 0.0},
        {"gamma": 0.0},
    ]:
        with pytest.raises(ValueError):
            Pcvi([x], **{**PAPER_KWARGS, **bad})


def test_multiple_param_groups_rejected():
    a = torch.nn.Parameter(torch.zeros(3))
    b = torch.nn.Parameter(torch.zeros(3))
    with pytest.raises(ValueError, match="one param group"):
        Pcvi([{"params": [a]}, {"params": [b]}], **PAPER_KWARGS)


def test_state_dict_roundtrip():
    """lambda_k must survive a checkpoint, otherwise resume resets the lr."""
    x = torch.nn.Parameter(U0.clone().to(DTYPE))
    opt = Pcvi([x], **PAPER_KWARGS)

    def closure():
        opt.zero_grad()
        _objective(x).backward()

    for _ in range(10):
        closure()
        opt.step(closure)

    state = opt.state_dict()
    assert state["lambda_k"] < PAPER_KWARGS["lambda_0"]  # it did adapt

    new = Pcvi([torch.nn.Parameter(U0.clone().to(DTYPE))], **PAPER_KWARGS)
    new.load_state_dict(state)
    assert new.lambda_k == opt.lambda_k
    assert new.n_steps == opt.n_steps


# The four departures from Algorithm 1. They live on Pcvi.__init__ with neutral
# defaults, and only PcviPlusConfig exposes them; every faithful config omits
# them on purpose, which is what these invariants have to allow for.
DEPARTURES = {
    "lambda_growth",
    "min_probe_rel",
    "beta_fallback",
    "float64_stats",
    "precond",
    "precond_beta2",
    "precond_eps",
    "precond_amsgrad",
    "precond_clamp",
}


def test_config_fields_match_optimizer_signature():
    """The 3 places that must agree: yaml <-> PcviConfig <-> Pcvi.__init__."""
    import inspect

    config = PcviConfig.get_from_yaml()
    signature = inspect.signature(Pcvi.__init__).parameters
    for field in config.__dict__:
        assert field in signature, f"{field} is in PcviConfig but not in Pcvi.__init__"
    for name in signature:
        if name in ("self", "params") or name in DEPARTURES:
            continue
        assert name in config.__dict__, f"{name} is in Pcvi.__init__ but not in the yaml"
    assert not (set(config.__dict__) & DEPARTURES), (
        "pcvi.yaml has to stay the paper; the departures belong to pcvi_plus.yaml"
    )


def test_config_yaml_defaults_are_faithful_to_the_paper():
    """The shipped defaults must be the ones with no deviation from the paper."""
    config = PcviConfig.get_from_yaml()
    assert config.lambda_min == 0.0
    assert config.beta_k_min is None
    assert config.beta_k_max is None
    assert config.use_identity_dk is False
    assert config.reset_lambda_per_batch is False
    assert 0 < config.beta < 2
    assert 0 < config.p < 1
    assert config.lambda_0 > 0
    assert config.gamma > 0


def test_config_builds_optimizer():
    config = PcviConfig.get_from_yaml()
    x = torch.nn.Parameter(U0.clone().to(DTYPE))
    opt = config.get_optimizer([x], experiment_config=None)
    assert isinstance(opt, Pcvi)
    assert opt.beta_k_min == -math.inf
    assert opt.beta_k_max == math.inf


# ----------------------------------------------------------------------------
# The "pc" variant: Steps 4-6 with a fixed lambda (KE_HOACH_IMPLEMENT_PC.md)
# ----------------------------------------------------------------------------

PC_KWARGS = dict(lambda_0=1.0, beta=1.95, p=0.5, gamma=1.0, use_adaptive_lambda=False)


def _lipschitz_constant() -> float:
    """The exact Lipschitz constant of F for Example 5.1.

    F is affine: F(x) = M x - q with
    M = c0 c0^T + sum_i T_i^T c_i c_i^T T_i, so L = lambda_max(M).
    """
    m = torch.outer(H_C, H_C).to(DTYPE)
    for t, c in zip(TS, HS_C):
        tc = t.to(DTYPE).T @ c.to(DTYPE)
        m = m + torch.outer(tc, tc)
    return torch.linalg.eigvalsh(m).max().item()


def test_lipschitz_constant_matches_the_finite_difference():
    """Sanity check on _lipschitz_constant before the tests below rely on it."""
    lipschitz = _lipschitz_constant()
    torch.manual_seed(0)
    worst = 0.0
    for _ in range(200):
        a = torch.randn(5, dtype=DTYPE)
        b = torch.randn(5, dtype=DTYPE)
        fa = torch.autograd.functional.jacobian(_objective, a).squeeze()
        fb = torch.autograd.functional.jacobian(_objective, b).squeeze()
        worst = max(worst, ((fa - fb).norm() / (a - b).norm()).item())
    assert worst <= lipschitz + 1e-9
    assert worst > 0.5 * lipschitz, "the random probe never got near the bound"


def test_pc_converges_when_lambda_below_p_over_L():
    """Inside the valid range lambda <= p/L, PC reaches Omega just like PCVI."""
    lam = 0.9 * PC_KWARGS["p"] / _lipschitz_constant()
    iterates, _ = _run({**PC_KWARGS, "lambda_0": lam}, n_steps=4000)
    assert _residuals(iterates[-1]).norm().item() < 1e-8


def test_pc_keeps_lambda_fixed():
    lam = 0.9 * PC_KWARGS["p"] / _lipschitz_constant()
    _, infos = _run({**PC_KWARGS, "lambda_0": lam}, n_steps=50)
    assert all(i["pcvi_lambda"] == lam for i in infos)
    assert all(i["pcvi_lambda_next"] == lam for i in infos)


def test_pc_beta_k_bound_holds_from_step_zero():
    """With lambda fixed, Remark 3.1 (iii) holds for ALL k, not just k >= k0.

    The proof needs ``1 - p (lambda_k/lambda_{k+1})^2 > 0``. With a constant
    lambda the ratio is exactly 1 from the start, so the condition reduces to
    ``1 - p > 0``, true by assumption. Contrast with
    :func:`test_first_step_can_move_away`, where PCVI's beta_0 is negative.
    """
    lam = 0.9 * PC_KWARGS["p"] / _lipschitz_constant()
    _, infos = _run({**PC_KWARGS, "lambda_0": lam}, n_steps=200)
    bound = min(PC_KWARGS["gamma"], PC_KWARGS["beta"] / 2)
    checked = 0
    for k, info in enumerate(infos):
        if info["pcvi_d_norm"] < 1e-6:  # converged: beta_k is 0/0 noise
            continue
        assert info["pcvi_beta_k"] >= bound - 1e-9, f"beta_k too small at step {k}"
        checked += 1
    assert checked >= 10, f"only {checked} steps had a meaningful d_k"
    # in particular at k = 0, which is where PCVI fails it
    assert infos[0]["pcvi_beta_k"] >= bound - 1e-9


def test_pc_equals_pcvi_when_lambda_0_is_small():
    """For lambda_0 <= p/L the two variants are the same algorithm.

    PCVI's ``min`` always picks lambda_k there, so its lambda never moves either.
    This is why comparing pc and pcvi at a single lambda_0 measures nothing --
    see KE_HOACH_IMPLEMENT_PC.md section 1.3.
    """
    lam = 0.9 * PC_KWARGS["p"] / _lipschitz_constant()
    pc, _ = _run({**PC_KWARGS, "lambda_0": lam}, n_steps=300)
    pcvi, infos = _run({**PAPER_KWARGS, "lambda_0": lam}, n_steps=300)
    assert all(i["pcvi_lambda"] == lam for i in infos), "pcvi moved lambda"
    for k, (a, b) in enumerate(zip(pc, pcvi)):
        assert torch.equal(a, b), f"diverged at step {k}"


def test_pc_diverges_when_lambda_too_large_but_pcvi_does_not():
    """Outside the valid range nothing pulls lambda back -- that is Step 3's job.

    This is the reason Step 3 exists, stated as a test.
    """
    lam = 50.0 * PC_KWARGS["p"] / _lipschitz_constant()
    pc, _ = _run({**PC_KWARGS, "lambda_0": lam}, n_steps=300)
    pcvi, _ = _run({**PAPER_KWARGS, "lambda_0": lam}, n_steps=300)

    pc_residual = _residuals(pc[-1]).norm().item()
    pcvi_residual = _residuals(pcvi[-1]).norm().item()
    assert pcvi_residual < 1e-8, f"pcvi failed to recover ({pcvi_residual})"
    # blowing up to nan is a perfectly good way of not converging
    assert math.isnan(pc_residual) or pc_residual > 1.0, (
        f"pc unexpectedly survived ({pc_residual})"
    )


def test_pc_config_yaml_and_signature():
    """pc.yaml <-> PcConfig <-> Pcvi.__init__, via _optimizer_kwargs."""
    import inspect

    config = PcConfig.get_from_yaml()
    signature = set(inspect.signature(Pcvi.__init__).parameters) - {"self", "params"}
    # the yaml deliberately omits p, lambda_min and use_adaptive_lambda
    assert set(config.__dict__) < signature
    assert "p" not in config.__dict__
    assert "lambda_min" not in config.__dict__
    # but _optimizer_kwargs must supply every argument of Algorithm 1 proper,
    # plus the two departures that mean anything with a frozen lambda
    lambda_only = {"lambda_growth", "min_probe_rel"}
    assert set(config._optimizer_kwargs(None)) == signature - lambda_only
    assert not (set(config.__dict__) & lambda_only), (
        "lambda never moves in pc: a growth factor or a probe guard would be a lie"
    )


def test_pc_config_defaults():
    config = PcConfig.get_from_yaml()
    kwargs = config._optimizer_kwargs(None)
    assert kwargs["use_adaptive_lambda"] is False
    assert kwargs["lambda_min"] == 0.0
    assert config.lambda_0 > 0
    assert 0 < config.beta < 2


def test_lambda_min_rejected_when_lambda_is_fixed():
    x = torch.nn.Parameter(U0.clone().to(DTYPE))
    with pytest.raises(ValueError, match="lambda_min"):
        Pcvi([x], **{**PC_KWARGS, "lambda_min": 0.1})


# ----------------------------------------------------------------------------
# The "extragradient" variant: beta_k = 1 (KE_HOACH_IMPLEMENT_EXTRAGRADIENT.md)
# ----------------------------------------------------------------------------

EG_KWARGS = dict(
    lambda_0=1.0, beta=1.0, p=0.5, gamma=1.0,
    use_adaptive_lambda=False, use_contraction=False,
)


def _gradient(x: torch.Tensor) -> torch.Tensor:
    """F(x) = grad L(x), computed outside the optimizer."""
    x = x.detach().clone().requires_grad_(True)
    _objective(x).backward()
    return x.grad.detach().clone()


def _hand_written_extragradient(x0, lam, n_steps):
    """Korpelevich (1976), written out in full."""
    x = x0.clone()
    out = [x.clone()]
    for _ in range(n_steps):
        v = x - lam * _gradient(x)
        x = x - lam * _gradient(v)
        out.append(x.clone())
    return out


def test_extragradient_matches_a_hand_written_loop():
    """The flag must BE extragradient, not something close to it."""
    lam = 0.9 * 0.5 / _lipschitz_constant()
    ours, _ = _run({**EG_KWARGS, "lambda_0": lam, "use_identity_dk": True}, n_steps=120)
    theirs = _hand_written_extragradient(U0.to(DTYPE), lam, 120)
    for k, (a, b) in enumerate(zip(ours, theirs)):
        assert torch.equal(a, b), f"diverged at step {k}: {(a - b).abs().max()}"


def test_extragradient_beta_k_is_exactly_one():
    lam = 0.9 * 0.5 / _lipschitz_constant()
    _, infos = _run({**EG_KWARGS, "lambda_0": lam}, n_steps=50)
    assert all(i["pcvi_beta_k"] == 1.0 for i in infos)


def test_extragradient_stability_limit_is_one_over_L():
    """Converges below lambda*L = 1, blows up above it. Classical result."""
    lipschitz = _lipschitz_constant()
    stable, _ = _run({**EG_KWARGS, "lambda_0": 0.9 / lipschitz}, n_steps=400)
    assert _residuals(stable[-1]).norm().item() < 1e-8

    unstable, _ = _run({**EG_KWARGS, "lambda_0": 1.125 / lipschitz}, n_steps=400)
    final = _residuals(unstable[-1]).norm().item()
    assert math.isnan(final) or final > 1.0, f"expected divergence, got {final}"


def test_contraction_beats_extragradient_at_every_scaled_step():
    """beta_k is NOT a step multiplier.

    KE_HOACH_IMPLEMENT_EXTRAGRADIENT.md section 1.1: pc reaches 1e-8 far sooner
    than extragradient at *any* rescaling of lambda, and scaling lambda by beta
    makes extragradient worse rather than better.
    """
    lam = 0.9 * 0.5 / _lipschitz_constant()

    def steps_to_converge(iterates, tol=1e-8):
        for k, x in enumerate(iterates):
            if _residuals(x).norm().item() < tol:
                return k
        return None

    pc, _ = _run({**PC_KWARGS, "lambda_0": lam}, n_steps=400)
    pc_steps = steps_to_converge(pc)
    assert pc_steps is not None

    for multiplier in (1.0, 1.5, 1.95, 2.0):
        eg = _hand_written_extragradient(U0.to(DTYPE), multiplier * lam, 400)
        eg_steps = steps_to_converge(eg)
        assert eg_steps is None or eg_steps > pc_steps, (
            f"extragradient at {multiplier}*lambda took {eg_steps} steps, "
            f"pc took {pc_steps}"
        )


def test_pc_effective_step_exceeds_the_extragradient_limit():
    """pc runs at beta_k*lambda > 1/L, where a fixed step diverges."""
    lipschitz = _lipschitz_constant()
    lam = 0.9 * 0.5 / lipschitz

    _, infos = _run({**PC_KWARGS, "lambda_0": lam}, n_steps=40)
    betas = [i["pcvi_beta_k"] for i in infos if i["pcvi_d_norm"] > 1e-9]
    effective = min(betas) * lam * lipschitz
    assert effective > 1.0, f"effective step is only {effective}/L"

    # and a fixed step that long does diverge
    diverging, _ = _run({**EG_KWARGS, "lambda_0": effective / lipschitz}, n_steps=400)
    final = _residuals(diverging[-1]).norm().item()
    assert math.isnan(final) or final > 1.0

    # while pc converges
    pc, _ = _run({**PC_KWARGS, "lambda_0": lam}, n_steps=400)
    assert _residuals(pc[-1]).norm().item() < 1e-8


def test_adaptive_extragradient_adapts_lambda_and_keeps_beta_one():
    _, infos = _run(
        {**EG_KWARGS, "lambda_0": 1.0, "use_adaptive_lambda": True}, n_steps=200
    )
    assert all(i["pcvi_beta_k"] == 1.0 for i in infos)
    lambdas = [i["pcvi_lambda"] for i in infos]
    assert lambdas[-1] < lambdas[0], "lambda never adapted"
    for k in range(len(lambdas) - 1):
        assert lambdas[k + 1] <= lambdas[k] + 1e-15


def test_beta_and_gamma_rejected_without_contraction():
    x = torch.nn.Parameter(U0.clone().to(DTYPE))
    for bad in ({"beta": 1.5}, {"gamma": 2.0}):
        with pytest.raises(ValueError, match="Step 5"):
            Pcvi([x], **{**EG_KWARGS, **bad})


def test_extragradient_configs_and_yaml():
    import inspect

    signature = set(inspect.signature(Pcvi.__init__).parameters) - {"self", "params"}
    for config_class, adaptive in (
        (ExtragradientConfig, False),
        (AdaptiveExtragradientConfig, True),
    ):
        config = config_class.get_from_yaml()
        assert set(config.__dict__) < signature
        for absent in ("beta", "gamma", "use_contraction", "use_adaptive_lambda"):
            assert absent not in config.__dict__, f"{absent} should not be exposed"
        kwargs = config._optimizer_kwargs(None)
        # The variants predate the four departures and do not set them, so Pcvi's
        # own defaults (all neutral) apply. What they must cover is every
        # parameter of Algorithm 1 proper.
        assert set(kwargs) == signature - DEPARTURES
        assert kwargs["use_contraction"] is False
        assert kwargs["use_adaptive_lambda"] is adaptive
        assert kwargs["beta"] == 1.0 and kwargs["gamma"] == 1.0
    # p only makes sense for the adaptive one
    assert "p" not in ExtragradientConfig.get_from_yaml().__dict__
    assert "p" in AdaptiveExtragradientConfig.get_from_yaml().__dict__
    # d_k is the step itself here, so the numerically safer route is the default
    assert ExtragradientConfig.get_from_yaml().use_identity_dk is True
