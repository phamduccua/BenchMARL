#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Tests for :class:`~benchmarl.optimizers.Adaptive`.

Adaptive keeps IPPO's Adam update and replaces only the learning rate, which is
driven by Step 3 of Algorithm 1 (https://doi.org/10.1002/mma.11132).

Tier 1 (pure maths) cannot reuse the paper's Example 5.1: that example exercises
Steps 4-6 as well, which this variant does not use. It is replaced by a quadratic
whose Lipschitz constant is known analytically, so the estimator can be checked
against ground truth.
"""

import pytest
import torch

from benchmarl.algorithms import IppoConfig, MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment
from benchmarl.models.mlp import MlpConfig
from benchmarl.optimizers import Adaptive, AdaptiveConfig, AdamConfig

DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Tier 1: a quadratic with a known Lipschitz constant
# ---------------------------------------------------------------------------
# L(x) = 1/2 x^T Q x  =>  F(x) = grad L(x) = Q x, which is exactly
# lambda_max(Q)-Lipschitz. Step 3 must therefore drive lambda to p / lambda_max(Q).


def _quadratic(n=6, cond=50.0, seed=0):
    torch.manual_seed(seed)
    eigenvalues = torch.logspace(0, torch.log10(torch.tensor(cond)), n, dtype=DTYPE)
    basis, _ = torch.linalg.qr(torch.randn(n, n, dtype=DTYPE))
    q = basis @ torch.diag(eigenvalues) @ basis.T
    q = 0.5 * (q + q.T)  # exactly symmetric
    return q, eigenvalues.max().item()


def _run(optimizer_kwargs, n_steps, q=None, x0=None, seed=0):
    if q is None:
        q, _ = _quadratic(seed=seed)
    if x0 is None:
        torch.manual_seed(seed + 100)
        x0 = torch.randn(q.shape[0], dtype=DTYPE)
    x = torch.nn.Parameter(x0.clone())
    optimizer = Adaptive([x], **optimizer_kwargs)

    def closure():
        optimizer.zero_grad()
        (0.5 * x @ (q @ x)).backward()

    iterates, infos = [x.detach().clone()], []
    for _ in range(n_steps):
        closure()
        if optimizer.lipschitz_from == "probe":
            infos.append(optimizer.step(closure))
        else:
            infos.append(optimizer.step())
        iterates.append(x.detach().clone())
    return iterates, infos, optimizer


BASE = dict(lambda_0=1.0, p=0.5, eps_denominator=1e-12)


@pytest.mark.parametrize("c", [2.0, 10.0, 37.5])
def test_lambda_converges_to_exactly_p_over_L_when_isotropic(c):
    """The estimator must measure what it claims to measure.

    With ``Q = c I`` every direction has the same Lipschitz ratio ``c``, so there
    is a single ground truth and Step 3 must find it exactly.
    """
    q = c * torch.eye(6, dtype=DTYPE)
    _, _, optimizer = _run(BASE, n_steps=200, q=q)
    assert optimizer.lambda_k == pytest.approx(BASE["p"] / c, rel=1e-9)


def test_lambda_measures_the_directional_lipschitz_ratio():
    """With an anisotropic Q, lambda lands inside the spectrum, not at lambda_max.

    Step 3 evaluates ``||F(u)-F(v)|| / ||u-v||`` along the direction the iterates
    actually travel, so it estimates a *directional* Lipschitz ratio, which lies
    somewhere in ``[lambda_min, lambda_max]`` -- not the operator norm.

    Worth knowing when reading lambda on a real task: it reports the curvature
    along the optimisation path, which is what the step should adapt to, but it
    is NOT an estimate of the global Lipschitz constant, and the theory's
    condition ``lambda <= p / L`` is stated in terms of the global one.
    """
    q, lambda_max = _quadratic()
    lambda_min = torch.linalg.eigvalsh(q).min().item()
    _, _, optimizer = _run(BASE, n_steps=200, q=q)
    assert BASE["p"] / lambda_max <= optimizer.lambda_k <= BASE["p"] / lambda_min
    assert optimizer.lambda_k < BASE["lambda_0"], "lambda never adapted at all"


def test_lambda_is_non_increasing():
    _, infos, _ = _run(BASE, n_steps=100)
    lambdas = [i["adaptive_lambda"] for i in infos]
    for k in range(len(lambdas) - 1):
        assert lambdas[k + 1] <= lambdas[k] + 1e-15, f"increased at step {k}"


def test_mechanism_is_inert_when_lambda_0_below_p_over_L():
    """The central trap: too small a lambda_0 and Step 3 never fires.

    Documented in KE_HOACH_IMPLEMENT_ADAPTIVE.md section 2.1; asserted here so it
    cannot quietly stop being true.
    """
    q, lipschitz = _quadratic()
    small = 0.1 * BASE["p"] / lipschitz
    _, infos, optimizer = _run({**BASE, "lambda_0": small}, n_steps=100, q=q)
    assert optimizer.lambda_k == small, "lambda moved when it should not have"
    assert all(i["adaptive_lambda_ratio"] == 1.0 for i in infos)


def test_lr_is_lr_scale_times_lambda():
    _, infos, _ = _run({**BASE, "lr_scale": 0.03}, n_steps=20)
    for info in infos:
        assert info["adaptive_lr"] == pytest.approx(0.03 * info["adaptive_lambda"])


def test_uses_lambda_k_not_lambda_next():
    """Step 6 builds its update from lambda_k; the lr has to match."""
    _, infos, _ = _run(BASE, n_steps=30)
    for previous, current in zip(infos, infos[1:]):
        assert current["adaptive_lambda"] == previous["adaptive_lambda_next"]


def test_iterates_mode_tracks_probe_mode():
    """The 1-gradient variant must land in the same ballpark as the faithful one.

    Not identical: the two estimate the ratio along different directions
    (``-lambda_k F(u_k)`` versus the Adam step actually taken), which is exactly
    the deviation ``lipschitz_from="iterates"`` trades for halving the cost.
    """
    q, lambda_max = _quadratic()
    lambda_min = torch.linalg.eigvalsh(q).min().item()
    _, _, probe = _run({**BASE, "lipschitz_from": "probe"}, n_steps=200, q=q)
    _, _, iterates = _run({**BASE, "lipschitz_from": "iterates"}, n_steps=200, q=q)

    for optimizer in (probe, iterates):
        assert BASE["p"] / lambda_max <= optimizer.lambda_k <= BASE["p"] / lambda_min
    assert iterates.lambda_k == pytest.approx(probe.lambda_k, rel=0.5)


@pytest.mark.parametrize("c", [2.0, 10.0])
def test_iterates_mode_is_exact_when_isotropic(c):
    """With a single ground truth, both estimators must find the same number."""
    q = c * torch.eye(6, dtype=DTYPE)
    _, _, optimizer = _run({**BASE, "lipschitz_from": "iterates"}, n_steps=200, q=q)
    assert optimizer.lambda_k == pytest.approx(BASE["p"] / c, rel=1e-9)


def test_iterates_mode_does_not_move_lambda_on_the_first_step():
    _, infos, _ = _run({**BASE, "lipschitz_from": "iterates"}, n_steps=3)
    assert infos[0]["adaptive_lambda_next"] == BASE["lambda_0"]


def test_step_without_closure_raises_in_probe_mode():
    q, _ = _quadratic()
    x = torch.nn.Parameter(torch.ones(q.shape[0], dtype=DTYPE))
    optimizer = Adaptive([x], **BASE)
    (0.5 * x @ (q @ x)).backward()
    with pytest.raises(ValueError, match="closure"):
        optimizer.step()


def test_parameters_restored_when_closure_raises():
    q, _ = _quadratic()
    x0 = torch.ones(q.shape[0], dtype=DTYPE)
    x = torch.nn.Parameter(x0.clone())
    optimizer = Adaptive([x], **BASE)
    (0.5 * x @ (q @ x)).backward()

    def bad_closure():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        optimizer.step(bad_closure)
    assert torch.allclose(x.detach(), x0, atol=1e-12)


def test_invalid_hyperparameters():
    x = torch.nn.Parameter(torch.zeros(3, dtype=DTYPE))
    for bad in [
        {"lambda_0": 0.0},
        {"p": 0.0},
        {"p": 1.0},
        {"lr_scale": 0.0},
        {"lambda_min": -1.0},
        {"lambda_min": 10.0},  # > lambda_0
        {"lipschitz_from": "nonsense"},
    ]:
        with pytest.raises(ValueError):
            Adaptive([x], **{**BASE, **bad})


def test_multiple_param_groups_rejected():
    a = torch.nn.Parameter(torch.zeros(3, dtype=DTYPE))
    b = torch.nn.Parameter(torch.zeros(3, dtype=DTYPE))
    with pytest.raises(ValueError, match="one param group"):
        Adaptive([{"params": [a]}, {"params": [b]}], **BASE)


def test_state_dict_roundtrip_keeps_lambda_and_adam_moments():
    q, _ = _quadratic()
    _, _, optimizer = _run(BASE, n_steps=15, q=q)
    assert optimizer.lambda_k < BASE["lambda_0"]  # it did adapt
    state = optimizer.state_dict()

    fresh = Adaptive([torch.nn.Parameter(torch.zeros(q.shape[0], dtype=DTYPE))], **BASE)
    fresh.load_state_dict(state)
    assert fresh.lambda_k == optimizer.lambda_k
    assert fresh.n_steps == optimizer.n_steps
    # Adam's moments have to survive too, not just lambda
    assert len(fresh._adam.state_dict()["state"]) == len(
        optimizer._adam.state_dict()["state"]
    )


# ---------------------------------------------------------------------------
# THE test: with lambda frozen, Adaptive must BE Adam
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lipschitz_from", ["probe", "iterates"])
def test_equals_adam_when_lambda_frozen(lipschitz_from):
    """Freeze lambda and Adaptive must reproduce plain Adam, bit for bit.

    This is the test that catches the central trap of the implementation: after
    the probe, ``.grad`` holds ``F(v_k)``, but the update must be Adam on the
    gradient at ``u_k``. Forgetting to restore it turns this into
    extragradient-with-Adam and raises no error whatsoever.
    """
    q, _ = _quadratic()
    torch.manual_seed(7)
    x0 = torch.randn(q.shape[0], dtype=DTYPE)
    lam, scale, eps = 0.01, 0.005, 1e-8
    n_steps = 40

    # --- Adaptive with lambda pinned by lambda_min == lambda_0 ---
    _, _, _ = None, None, None
    x_adaptive = torch.nn.Parameter(x0.clone())
    adaptive = Adaptive(
        [x_adaptive],
        lambda_0=lam,
        p=0.5,
        eps_denominator=1e-12,
        lr_scale=scale,
        lambda_min=lam,  # freezes the schedule
        lipschitz_from=lipschitz_from,
        eps=eps,
    )

    def closure():
        adaptive.zero_grad()
        (0.5 * x_adaptive @ (q @ x_adaptive)).backward()

    for _ in range(n_steps):
        closure()
        adaptive.step(closure if lipschitz_from == "probe" else None)

    assert adaptive.lambda_k == lam, "lambda_min failed to freeze the schedule"

    # --- plain Adam at the same effective lr ---
    x_adam = torch.nn.Parameter(x0.clone())
    adam = torch.optim.Adam([x_adam], lr=scale * lam, eps=eps)
    for _ in range(n_steps):
        adam.zero_grad()
        (0.5 * x_adam @ (q @ x_adam)).backward()
        adam.step()

    assert torch.equal(x_adaptive.detach(), x_adam.detach()), (
        f"max diff {(x_adaptive - x_adam).abs().max().item()} -- Adaptive is not "
        f"taking the Adam step at u_k"
    )


# ---------------------------------------------------------------------------
# Tier 2: wired into BenchMARL
# ---------------------------------------------------------------------------


@pytest.fixture
def config(experiment_config):
    experiment_config.max_n_iters = 2
    experiment_config.on_policy_n_minibatch_iters = 2
    experiment_config.on_policy_minibatch_size = 50
    experiment_config.evaluation = False
    experiment_config.render = False
    experiment_config.checkpoint_interval = 0
    experiment_config.loggers = []
    experiment_config.create_json = False
    experiment_config.clip_grad_val = None
    return experiment_config


def _experiment(config, optimizer_config, algorithm_config=None, **overrides):
    for key, value in overrides.items():
        setattr(config, key, value)
    return Experiment(
        task=VmasTask.BALANCE.get_from_yaml(),
        algorithm_config=(
            algorithm_config
            if algorithm_config is not None
            else IppoConfig.get_from_yaml()
        ),
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        optimizer_config=optimizer_config,
        seed=0,
        config=config,
    )


def test_yaml_matches_dataclass_and_signature():
    """The 3 places that must agree: yaml <-> AdaptiveConfig <-> Adaptive.__init__."""
    import inspect

    optimizer_config = AdaptiveConfig.get_from_yaml()
    signature = inspect.signature(Adaptive.__init__).parameters
    assert set(optimizer_config.__dict__) == set(signature) - {"self", "params"}


def test_yaml_defaults_are_faithful_to_the_paper():
    optimizer_config = AdaptiveConfig.get_from_yaml()
    assert optimizer_config.lr_scale == 1.0
    assert optimizer_config.lambda_min == 0.0
    assert optimizer_config.lipschitz_from == "probe"
    assert optimizer_config.reset_lambda_per_batch is False
    assert 0 < optimizer_config.p < 1
    assert optimizer_config.lambda_0 > 0


def test_refuses_gradient_clipping(config):
    with pytest.raises(ValueError, match="clip_grad_val"):
        _experiment(config, AdaptiveConfig.get_from_yaml(), clip_grad_val=5)


def test_refuses_gradient_clipping_in_iterates_mode_too(config):
    """Clipping breaks the Lipschitz estimate whichever way it is computed."""
    optimizer_config = AdaptiveConfig.get_from_yaml()
    optimizer_config.lipschitz_from = "iterates"
    with pytest.raises(ValueError, match="clip_grad_val"):
        _experiment(config, optimizer_config, clip_grad_val=5)


def test_warns_that_experiment_lr_is_ignored(config):
    with pytest.warns(UserWarning, match="ignored"):
        _experiment(config, AdaptiveConfig.get_from_yaml())


@pytest.mark.parametrize(
    "algorithm_config", [IppoConfig.get_from_yaml(), MappoConfig.get_from_yaml()]
)
def test_runs(config, algorithm_config):
    experiment = _experiment(
        config, AdaptiveConfig.get_from_yaml(), algorithm_config=algorithm_config
    )
    assert experiment.two_point_optimizers
    experiment.run()
    for optimizers in experiment.optimizers.values():
        for optimizer in optimizers.values():
            assert isinstance(optimizer, Adaptive)
            assert optimizer.n_steps > 0


def test_iterates_mode_uses_the_single_gradient_loop(config):
    optimizer_config = AdaptiveConfig.get_from_yaml()
    optimizer_config.lipschitz_from = "iterates"
    experiment = _experiment(config, optimizer_config)
    assert not experiment.two_point_optimizers, "should use the 1-gradient loop"
    experiment.run()

    group = "agents"
    calls = []
    loss = experiment.losses[group]
    original_forward = loss.forward

    def counting_forward(*args, **kwargs):
        calls.append(1)
        return original_forward(*args, **kwargs)

    loss.forward = counting_forward
    try:
        training_td = experiment._optimizer_loop(group)
    finally:
        loss.forward = original_forward
    assert len(calls) == 1, f"{len(calls)} forward passes, expected 1"
    keys = {k for k in training_td.keys() if isinstance(k, str)}
    assert "adaptive_lambda_loss_objective" in keys


def test_probe_mode_uses_exactly_two_forward_passes(config):
    experiment = _experiment(config, AdaptiveConfig.get_from_yaml())
    experiment.run()

    group = "agents"
    calls = []
    loss = experiment.losses[group]
    original_forward = loss.forward

    def counting_forward(*args, **kwargs):
        calls.append(1)
        return original_forward(*args, **kwargs)

    loss.forward = counting_forward
    try:
        experiment._optimizer_loop_two_point(group)
    finally:
        loss.forward = original_forward
    assert len(calls) == 2, f"{len(calls)} forward passes, expected 2"


def test_leaves_no_probe_state_and_moves_the_parameters(config):
    experiment = _experiment(config, AdaptiveConfig.get_from_yaml())
    experiment.run()

    group = "agents"
    optimizer = experiment.optimizers[group]["loss_objective"]
    before = [p.detach().clone() for p in optimizer.param_groups[0]["params"]]
    experiment._optimizer_loop_two_point(group)
    after = [p.detach().clone() for p in optimizer.param_groups[0]["params"]]

    assert max((a - b).abs().max().item() for a, b in zip(before, after)) > 0
    for optimizers in experiment.optimizers.values():
        for opt in optimizers.values():
            assert opt._u is None, "probe state left behind"


def test_logs_its_diagnostics(config):
    experiment = _experiment(config, AdaptiveConfig.get_from_yaml())
    experiment.run()
    training_td = experiment._optimizer_loop_two_point("agents")
    keys = {k for k in training_td.keys() if isinstance(k, str)}
    for loss_name in ("loss_objective", "loss_critic"):
        for stat in ("adaptive_lambda", "adaptive_lr", "adaptive_lambda_ratio"):
            assert f"{stat}_{loss_name}" in keys, f"{stat}_{loss_name} not logged"


def test_lambda_survives_a_checkpoint(config):
    experiment = _experiment(
        config,
        AdaptiveConfig.get_from_yaml(),
        # torchrl 0.11 drops SyncDataCollector.env on shutdown, which breaks
        # Experiment.state_dict() after run(); unrelated to the optimizer.
        collect_with_grad=True,
    )
    experiment.run()
    optimizer = experiment.optimizers["agents"]["loss_objective"]
    state_dict = experiment.state_dict()

    key = "optimizer_agents_loss_objective"
    assert state_dict[key]["lambda_k"] == optimizer.lambda_k

    reloaded = _experiment(
        config, AdaptiveConfig.get_from_yaml(), collect_with_grad=True
    )
    reloaded.load_state_dict(state_dict)
    restored = reloaded.optimizers["agents"]["loss_objective"]
    assert restored.lambda_k == optimizer.lambda_k
    assert restored.n_steps == optimizer.n_steps


def test_adam_path_still_works(config):
    """Adaptive must not have disturbed the baseline."""
    experiment = _experiment(config, AdamConfig.get_from_yaml(), clip_grad_val=5)
    assert not experiment.two_point_optimizers
    experiment.run()
    for optimizers in experiment.optimizers.values():
        for optimizer in optimizers.values():
            assert isinstance(optimizer, torch.optim.Adam)
