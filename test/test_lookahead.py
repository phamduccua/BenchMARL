#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Tests for :class:`~benchmarl.optimizers.Lookahead` (Zhang et al., NeurIPS 2019).

Lookahead is a wrapper, so most of the risk is in *forwarding*: if the
``probe``/``restore``/``apply`` protocol of a two-gradient inner optimizer is not
passed through correctly, the run is silently wrong.
``test_equals_inner_when_k_is_huge`` is the test that catches that.
"""

import pytest
import torch

from benchmarl.algorithms import IppoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment
from benchmarl.models.mlp import MlpConfig
from benchmarl.optimizers import (
    AdamConfig,
    Lookahead,
    LookaheadConfig,
    optimizer_config_registry,
)

DTYPE = torch.float64
WRAPPABLE = ["adam", "adaptive", "extragradient", "pc", "pcvi"]


# ---------------------------------------------------------------------------
# Tier 1: pure maths
# ---------------------------------------------------------------------------


def _quadratic(n=6, seed=0):
    torch.manual_seed(seed)
    basis, _ = torch.linalg.qr(torch.randn(n, n, dtype=DTYPE))
    q = basis @ torch.diag(torch.linspace(1.0, 10.0, n, dtype=DTYPE)) @ basis.T
    return 0.5 * (q + q.T)


def _sgd_run(lookahead_kwargs, n_steps, lr=0.01, seed=0):
    q = _quadratic(seed=seed)
    torch.manual_seed(seed + 1)
    x0 = torch.randn(q.shape[0], dtype=DTYPE)
    x = torch.nn.Parameter(x0.clone())
    optimizer = Lookahead([x], inner=torch.optim.SGD([x], lr=lr), **lookahead_kwargs)
    out = [x.detach().clone()]
    for _ in range(n_steps):
        optimizer.zero_grad()
        (0.5 * x @ (q @ x)).backward()
        optimizer.step()
        out.append(x.detach().clone())
    return out, q, x0


def _hand_written_lookahead(q, x0, lr, k, alpha, n_steps):
    """Algorithm 1 of Zhang et al., written out in full.

    Drives a real ``torch.optim.SGD`` for the inner steps so that the inner
    trajectory is bit-identical to the wrapper's by construction, leaving only
    the sync arithmetic under test.
    """
    theta = torch.nn.Parameter(x0.clone())
    sgd = torch.optim.SGD([theta], lr=lr)
    phi = x0.clone()
    since = 0
    out = [theta.detach().clone()]
    for _ in range(n_steps):
        sgd.zero_grad()
        (0.5 * theta @ (q @ theta)).backward()
        sgd.step()
        since += 1
        if since == k:
            phi = phi + alpha * (theta.detach() - phi)
            theta.data.copy_(phi)
            since = 0
        out.append(theta.detach().clone())
    return out


# Exact bit-equality is not attainable here: the wrapper does the interpolation
# with torch._foreach_* while the reference uses plain ops, and those round
# differently in the last bit (~1e-16). Any logic error -- a wrong alpha, an
# off-by-one on k, syncing before instead of after the inner step -- shows up at
# 1e-2 or larger, so 1e-14 over 60 steps is a sharp test all the same. The sync
# arithmetic itself is pinned exactly by test_slow_weights_sit_on_the_segment.
ROUNDING = 1e-14


def test_matches_a_hand_written_lookahead():
    """The wrapper must BE Lookahead, not something close to it."""
    k, alpha, lr, n = 5, 0.5, 0.01, 60
    ours, q, x0 = _sgd_run({"k": k, "alpha": alpha}, n_steps=n, lr=lr)
    theirs = _hand_written_lookahead(q, x0, lr, k, alpha, n)
    for step, (a, b) in enumerate(zip(ours, theirs)):
        assert torch.allclose(a, b, atol=ROUNDING), (
            f"diverged at step {step} by {(a - b).abs().max().item()}"
        )


def test_alpha_one_is_a_noop():
    """alpha = 1 gives phi <- theta, so the inner optimizer runs untouched.

    Only up to rounding: ``phi + 1*(theta - phi)`` is not exactly ``theta`` in
    floating point.
    """
    with pytest.warns(UserWarning, match="no-op"):
        wrapped, q, x0 = _sgd_run({"k": 3, "alpha": 1.0}, n_steps=40)
    plain = _hand_written_lookahead(q, x0, 0.01, 10**9, 0.5, 40)
    for step, (a, b) in enumerate(zip(wrapped, plain)):
        assert torch.allclose(a, b, atol=ROUNDING), f"diverged at step {step}"


def test_a_wrong_k_would_be_caught():
    """Guards the tolerance above: an off-by-one on k is orders of magnitude bigger."""
    ours, q, x0 = _sgd_run({"k": 5, "alpha": 0.5}, n_steps=60, lr=0.01)
    wrong = _hand_written_lookahead(q, x0, 0.01, 6, 0.5, 60)
    drift = max((a - b).abs().max().item() for a, b in zip(ours, wrong))
    assert drift > 1e4 * ROUNDING, f"drift {drift} is too close to the tolerance"


def test_slow_weights_sit_on_the_segment():
    """phi lands exactly alpha of the way from phi_prev to theta."""
    k, alpha, lr = 4, 0.3, 0.02
    q = _quadratic()
    x0 = torch.ones(q.shape[0], dtype=DTYPE)
    x = torch.nn.Parameter(x0.clone())
    optimizer = Lookahead([x], inner=torch.optim.SGD([x], lr=lr), k=k, alpha=alpha)

    phi_previous = x0.clone()
    for _ in range(k - 1):
        optimizer.zero_grad()
        (0.5 * x @ (q @ x)).backward()
        info = optimizer.step()
        assert info["lookahead_synced"] == 0.0
    theta_before = x.detach().clone()

    optimizer.zero_grad()
    (0.5 * x @ (q @ x)).backward()
    theta_k = theta_before - lr * (q @ theta_before)  # what SGD will do
    info = optimizer.step()

    assert info["lookahead_synced"] == 1.0
    expected = phi_previous + alpha * (theta_k - phi_previous)
    assert torch.allclose(x.detach(), expected, atol=1e-12)
    assert torch.allclose(torch.stack(optimizer._slow), expected, atol=1e-12)


def test_sync_happens_every_k_steps():
    k = 3
    q = _quadratic()
    x = torch.nn.Parameter(torch.ones(q.shape[0], dtype=DTYPE))
    optimizer = Lookahead([x], inner=torch.optim.SGD([x], lr=0.01), k=k, alpha=0.5)
    synced = []
    for _ in range(12):
        optimizer.zero_grad()
        (0.5 * x @ (q @ x)).backward()
        synced.append(optimizer.step()["lookahead_synced"])
    assert synced == [0.0, 0.0, 1.0] * 4
    assert optimizer.n_syncs == 4


def test_invalid_hyperparameters():
    x = torch.nn.Parameter(torch.zeros(3, dtype=DTYPE))
    for bad in ({"k": 0}, {"k": 1.5}, {"alpha": 0.0}, {"alpha": 1.5}):
        with pytest.raises(ValueError):
            Lookahead([x], inner=torch.optim.SGD([x], lr=0.1), **bad)


def test_state_dict_roundtrip():
    q = _quadratic()
    x = torch.nn.Parameter(torch.ones(q.shape[0], dtype=DTYPE))
    optimizer = Lookahead([x], inner=torch.optim.SGD([x], lr=0.01), k=4, alpha=0.5)
    for _ in range(6):
        optimizer.zero_grad()
        (0.5 * x @ (q @ x)).backward()
        optimizer.step()
    state = optimizer.state_dict()
    assert state["since_sync"] == 2 and state["n_syncs"] == 1

    y = torch.nn.Parameter(torch.zeros(q.shape[0], dtype=DTYPE))
    fresh = Lookahead([y], inner=torch.optim.SGD([y], lr=0.01), k=4, alpha=0.5)
    fresh.load_state_dict(state)
    assert fresh._since_sync == 2
    assert fresh.n_syncs == 1
    for a, b in zip(fresh._slow, optimizer._slow):
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Tier 2: wrapped around the real optimizers, inside BenchMARL
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


def _experiment(config, optimizer_config, **overrides):
    for key, value in overrides.items():
        setattr(config, key, value)
    return Experiment(
        task=VmasTask.BALANCE.get_from_yaml(),
        algorithm_config=IppoConfig.get_from_yaml(),
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        optimizer_config=optimizer_config,
        seed=0,
        config=config,
    )


def _lookahead_config(inner, **kwargs):
    optimizer_config = LookaheadConfig.get_from_yaml()
    optimizer_config.inner = inner
    for key, value in kwargs.items():
        setattr(optimizer_config, key, value)
    return optimizer_config


def test_yaml_matches_dataclass_and_signature():
    import inspect

    optimizer_config = LookaheadConfig.get_from_yaml()
    signature = set(inspect.signature(Lookahead.__init__).parameters)
    fields = set(optimizer_config.__dict__)
    assert fields - {"inner_overrides"} <= signature | {"inner"}
    assert optimizer_config.alpha == 0.5 and optimizer_config.k == 5


def test_refuses_to_wrap_itself():
    with pytest.raises(ValueError, match="itself"):
        _lookahead_config("lookahead").inner_config()


def test_rejects_unknown_inner():
    with pytest.raises(ValueError, match="Unknown inner"):
        _lookahead_config("nonesuch").inner_config()


def test_rejects_unknown_override():
    with pytest.raises(ValueError, match="no field"):
        _lookahead_config("adam", inner_overrides={"nope": 1}).inner_config()


@pytest.mark.parametrize("inner", WRAPPABLE)
def test_delegates_its_capabilities(inner):
    """Whether a second backward is needed is a property of the INNER optimizer."""
    optimizer_config = _lookahead_config(inner)
    reference = optimizer_config_registry[inner].get_from_yaml()
    assert (
        optimizer_config.requires_two_gradient_evals()
        == reference.requires_two_gradient_evals()
    )
    assert (
        optimizer_config.uses_gradient_as_operator()
        == reference.uses_gradient_as_operator()
    )


@pytest.mark.parametrize("inner", WRAPPABLE)
def test_wraps_every_registry_optimizer(config, inner):
    experiment = _experiment(config, _lookahead_config(inner, k=2))
    experiment.run()
    for optimizers in experiment.optimizers.values():
        for optimizer in optimizers.values():
            assert isinstance(optimizer, Lookahead)
            assert optimizer.n_syncs > 0


@pytest.mark.parametrize("inner", ["pcvi", "extragradient"])
def test_forwards_the_two_gradient_protocol(config, inner):
    experiment = _experiment(config, _lookahead_config(inner, k=2))
    assert experiment.two_point_optimizers, "delegation failed"
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
        training_td = experiment._optimizer_loop_two_point(group)
    finally:
        loss.forward = original_forward
    assert len(calls) == 2, f"{len(calls)} forward passes, expected 2"
    keys = {k for k in training_td.keys() if isinstance(k, str)}
    assert "lookahead_synced_loss_objective" in keys
    # the inner optimizer's own diagnostics must survive the wrapper
    assert "pcvi_lambda_loss_objective" in keys


@pytest.mark.parametrize("inner", WRAPPABLE)
def test_equals_inner_when_k_is_huge(config, inner):
    """No sync ever fires => bit-identical to the bare inner optimizer.

    This is the test that catches a broken probe/restore/apply forwarding: a
    wrapper that does not pass the protocol through would produce different
    parameters even with the sync disabled.
    """
    import copy

    wrapped = _experiment(
        copy.deepcopy(config), _lookahead_config(inner, k=10**9, alpha=0.5)
    )
    wrapped.run()
    bare = _experiment(
        copy.deepcopy(config), optimizer_config_registry[inner].get_from_yaml()
    )
    bare.run()

    for group in wrapped.optimizers:
        for loss_name in wrapped.optimizers[group]:
            a = wrapped.optimizers[group][loss_name].param_groups[0]["params"]
            b = bare.optimizers[group][loss_name].param_groups[0]["params"]
            for pa, pb in zip(a, b):
                assert torch.equal(pa, pb), f"{inner}: {group}/{loss_name} differs"
        for optimizer in wrapped.optimizers[group].values():
            assert optimizer.n_syncs == 0


def test_iterate_history_is_invalidated_on_sync(config):
    """Lookahead's pull-back is not an optimizer step.

    ``Adaptive`` with ``lipschitz_from="iterates"`` estimates the Lipschitz
    constant from consecutive iterates; measuring across the jump would drag
    lambda down for no reason.
    """
    optimizer_config = _lookahead_config(
        "adaptive", k=2, inner_overrides={"lipschitz_from": "iterates", "lambda_0": 1.0}
    )
    experiment = _experiment(config, optimizer_config)
    assert not experiment.two_point_optimizers  # "iterates" is a 1-gradient method
    experiment.run()

    optimizer = experiment.optimizers["agents"]["loss_objective"]
    assert optimizer.n_syncs > 0
    inner = optimizer.inner
    # right after a sync the history is empty, so the next step cannot move lambda
    optimizer._sync()
    assert inner._prev_params is None and inner._prev_grads is None


def test_warns_when_k_does_not_divide_the_updates_per_round(config):
    updates = (
        config.on_policy_n_minibatch_iters
        * -(-config.on_policy_collected_frames_per_batch // config.on_policy_minibatch_size)
    )
    with pytest.warns(UserWarning, match="mid-cycle"):
        _experiment(config, _lookahead_config("adam", k=updates + 1))


def test_sync_at_round_end(config):
    experiment = _experiment(
        config, _lookahead_config("adam", k=10**9, sync_at_round_end=True)
    )
    experiment.run()
    optimizer = experiment.optimizers["agents"]["loss_objective"]
    # one forced sync per collection round, despite k being unreachable
    assert optimizer.n_syncs == config.max_n_iters


def test_checkpoint_roundtrip(config):
    experiment = _experiment(
        config, _lookahead_config("pcvi", k=2), collect_with_grad=True
    )
    experiment.run()
    optimizer = experiment.optimizers["agents"]["loss_objective"]
    state_dict = experiment.state_dict()
    saved = state_dict["optimizer_agents_loss_objective"]
    assert saved["n_syncs"] == optimizer.n_syncs
    assert saved["inner"]["lambda_k"] == optimizer.inner.lambda_k

    reloaded = _experiment(
        config, _lookahead_config("pcvi", k=2), collect_with_grad=True
    )
    reloaded.load_state_dict(state_dict)
    restored = reloaded.optimizers["agents"]["loss_objective"]
    assert restored.n_syncs == optimizer.n_syncs
    assert restored.inner.lambda_k == optimizer.inner.lambda_k
    for a, b in zip(restored._slow, optimizer._slow):
        assert torch.equal(a, b)


def test_adam_baseline_is_one_gradient_per_step(config):
    """Lookahead + Adam must stay at IPPO's cost: one forward per update."""
    experiment = _experiment(config, _lookahead_config("adam", k=2))
    assert not experiment.two_point_optimizers
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
        experiment._optimizer_loop(group)
    finally:
        loss.forward = original_forward
    assert len(calls) == 1


def test_does_not_disturb_the_plain_adam_path(config):
    experiment = _experiment(config, AdamConfig.get_from_yaml(), clip_grad_val=5)
    assert not experiment.two_point_optimizers
    experiment.run()
    for optimizers in experiment.optimizers.values():
        for optimizer in optimizers.values():
            assert isinstance(optimizer, torch.optim.Adam)
