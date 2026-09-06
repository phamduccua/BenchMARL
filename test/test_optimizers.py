#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Tier-2 tests: the optimizer config group wired into BenchMARL.

Tier-1 (``test_pcvi_math.py``) checks that Algorithm 1 is implemented correctly.
These check that it is *plugged in* correctly, and that the Adam path is
unchanged.
"""

import pytest
import torch

from benchmarl.algorithms import IppoConfig, MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment
from benchmarl.models.mlp import MlpConfig
from benchmarl.optimizers import (
    AdamConfig,
    AdaptiveExtragradientConfig,
    ExtragradientConfig,
    optimizer_config_registry,
    PcConfig,
    Pcvi,
    PcviConfig,
)


@pytest.fixture
def config(experiment_config):
    """The shared fixture, trimmed down to make these tests fast."""
    experiment_config.max_n_iters = 2
    experiment_config.on_policy_n_minibatch_iters = 2
    experiment_config.on_policy_minibatch_size = 50
    experiment_config.evaluation = False
    experiment_config.render = False
    experiment_config.checkpoint_interval = 0
    experiment_config.loggers = []
    experiment_config.create_json = False
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


# ----------------------------------------------------------------- registry


def test_registry_configs_load_from_yaml():
    for name, config_class in optimizer_config_registry.items():
        config = config_class.get_from_yaml()
        assert isinstance(config, config_class), name


def test_pcvi_yaml_matches_dataclass_and_signature():
    """The 3 places that must agree: yaml <-> PcviConfig <-> Pcvi.__init__."""
    import inspect

    config = PcviConfig.get_from_yaml()
    signature = inspect.signature(Pcvi.__init__).parameters
    assert set(config.__dict__) == set(signature) - {"self", "params"}


# ------------------------------------------------------- Adam is unchanged


def test_default_optimizer_is_adam_reading_the_experiment_config(config):
    """No optimizer_config => the exact pre-existing behaviour."""
    experiment = _experiment(config, None)
    assert not experiment.two_point_optimizers
    for optimizers in experiment.optimizers.values():
        for optimizer in optimizers.values():
            assert isinstance(optimizer, torch.optim.Adam)
            assert optimizer.param_groups[0]["lr"] == experiment.config.lr
            assert optimizer.param_groups[0]["eps"] == experiment.config.adam_eps


def test_adam_config_overrides_the_experiment_config(config):
    adam = AdamConfig.get_from_yaml()
    adam.lr = 0.123
    experiment = _experiment(config, adam)
    optimizer = experiment.optimizers["agents"]["loss_objective"]
    assert optimizer.param_groups[0]["lr"] == 0.123
    assert optimizer.param_groups[0]["eps"] == experiment.config.adam_eps


def test_adam_run_is_unaffected_by_the_optimizer_plumbing(config, experiment_config):
    """Same seed, with and without an explicit AdamConfig => same parameters."""
    implicit = _experiment(config, None)
    implicit.run()
    explicit = _experiment(config, AdamConfig.get_from_yaml())
    explicit.run()
    for group in implicit.optimizers:
        a = implicit.optimizers[group]["loss_objective"].param_groups[0]["params"]
        b = explicit.optimizers[group]["loss_objective"].param_groups[0]["params"]
        for pa, pb in zip(a, b):
            assert torch.allclose(pa, pb, atol=0.0), "the Adam path changed"


# ------------------------------------------------------------ PCVI wiring


def test_pcvi_refuses_gradient_clipping(config):
    """Clipping distorts F and corrupts the Lipschitz estimate driving lambda."""
    with pytest.raises(ValueError, match="clip_grad_val"):
        _experiment(config, PcviConfig.get_from_yaml(), clip_grad_val=5)


@pytest.mark.parametrize(
    "algorithm_config", [IppoConfig.get_from_yaml(), MappoConfig.get_from_yaml()]
)
def test_pcvi_runs(config, algorithm_config):
    experiment = _experiment(
        config,
        PcviConfig.get_from_yaml(),
        algorithm_config=algorithm_config,
        clip_grad_val=None,
    )
    assert experiment.two_point_optimizers
    experiment.run()
    for optimizers in experiment.optimizers.values():
        for optimizer in optimizers.values():
            assert isinstance(optimizer, Pcvi)
            assert optimizer.n_steps > 0


def test_pcvi_uses_exactly_two_forward_passes_per_update(config):
    """One second forward for the whole group, not one per loss."""
    experiment = _experiment(config, PcviConfig.get_from_yaml(), clip_grad_val=None)
    experiment.run()  # fill the buffer

    group = "agents"
    assert len(experiment.optimizers[group]) == 2, "expected an actor and a critic loss"

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


def test_pcvi_moves_the_parameters_and_leaves_no_probe_state(config):
    experiment = _experiment(config, PcviConfig.get_from_yaml(), clip_grad_val=None)
    experiment.run()

    group = "agents"
    optimizer = experiment.optimizers[group]["loss_objective"]
    before = [p.detach().clone() for p in optimizer.param_groups[0]["params"]]
    experiment._optimizer_loop_two_point(group)
    after = [p.detach().clone() for p in optimizer.param_groups[0]["params"]]

    assert max((a - b).abs().max().item() for a, b in zip(before, after)) > 0
    # the parameters must be at u_{k+1}, not left at the probe point v_k
    for optimizers in experiment.optimizers.values():
        for opt in optimizers.values():
            assert opt._u is None, "probe state left behind"


def test_pcvi_logs_its_diagnostics(config):
    experiment = _experiment(config, PcviConfig.get_from_yaml(), clip_grad_val=None)
    experiment.run()
    training_td = experiment._optimizer_loop_two_point("agents")
    keys = {k for k in training_td.keys() if isinstance(k, str)}
    for loss_name in ("loss_objective", "loss_critic"):
        for stat in ("pcvi_lambda", "pcvi_beta_k", "pcvi_effective_lr", "pcvi_beta_clamped"):
            assert f"{stat}_{loss_name}" in keys, f"{stat}_{loss_name} not logged"
        assert f"grad_norm_{loss_name}" in keys


def test_pcvi_lambda_survives_a_checkpoint(config):
    experiment = _experiment(
        config,
        PcviConfig.get_from_yaml(),
        clip_grad_val=None,
        # torchrl 0.11 drops SyncDataCollector.env on shutdown, which breaks
        # Experiment.state_dict() after run(); unrelated to the optimizer.
        collect_with_grad=True,
    )
    experiment.run()
    optimizer = experiment.optimizers["agents"]["loss_objective"]
    state_dict = experiment.state_dict()

    key = "optimizer_agents_loss_objective"
    assert key in state_dict
    assert state_dict[key]["lambda_k"] == optimizer.lambda_k

    reloaded = _experiment(
        config, PcviConfig.get_from_yaml(), clip_grad_val=None, collect_with_grad=True
    )
    reloaded.load_state_dict(state_dict)
    assert reloaded.optimizers["agents"]["loss_objective"].lambda_k == optimizer.lambda_k
    assert reloaded.optimizers["agents"]["loss_objective"].n_steps == optimizer.n_steps


def test_load_state_dict_without_optimizers_still_works(config):
    """Checkpoints taken before optimizers were saved must still load."""
    experiment = _experiment(config, None, collect_with_grad=True)
    experiment.run()
    state_dict = experiment.state_dict()
    for key in [k for k in state_dict if k.startswith("optimizer_")]:
        del state_dict[key]

    other = _experiment(config, None, collect_with_grad=True)
    other.load_state_dict(state_dict)  # must not raise


def test_reset_lambda_per_batch(config):
    pcvi = PcviConfig.get_from_yaml()
    pcvi.lambda_0 = 1.0
    pcvi.reset_lambda_per_batch = True
    experiment = _experiment(config, pcvi, clip_grad_val=None)
    experiment.run()
    optimizer = experiment.optimizers["agents"]["loss_objective"]
    assert optimizer.n_steps > 0
    # lambda was reset at the start of the last collection round, so it can only
    # have been shrunk by the steps of that round
    assert optimizer.lambda_k <= pcvi.lambda_0


# ------------------------------------------------------------- the pc variant


def test_pc_runs_and_keeps_lambda_fixed(config):
    experiment = _experiment(config, PcConfig.get_from_yaml(), clip_grad_val=None)
    assert experiment.two_point_optimizers
    experiment.run()
    for optimizers in experiment.optimizers.values():
        for optimizer in optimizers.values():
            assert isinstance(optimizer, Pcvi)
            assert optimizer.use_adaptive_lambda is False
            assert optimizer.n_steps > 0
            assert optimizer.lambda_k == optimizer.lambda_0, "lambda moved"


def test_pc_refuses_gradient_clipping(config):
    with pytest.raises(ValueError, match="clip_grad_val"):
        _experiment(config, PcConfig.get_from_yaml(), clip_grad_val=5)


def test_pc_matches_pcvi_when_lambda_0_is_small(config):
    """The equivalence of KE_HOACH_IMPLEMENT_PC.md section 1.3, end to end.

    Below ``p / L`` the two variants are the same algorithm, so a comparison at a
    single small lambda_0 measures nothing. Asserted here on a real task, not
    just on the paper's example.
    """
    small = 1e-5  # far below the ~2e-3 measured on this task

    pc_config = PcConfig.get_from_yaml()
    pc_config.lambda_0 = small
    pc = _experiment(config, pc_config, clip_grad_val=None)
    pc.run()

    pcvi_config = PcviConfig.get_from_yaml()
    pcvi_config.lambda_0 = small
    pcvi = _experiment(config, pcvi_config, clip_grad_val=None)
    pcvi.run()

    assert pcvi.optimizers["agents"]["loss_objective"].lambda_k == small, (
        "pcvi moved lambda; the premise of the equivalence does not hold"
    )
    for group in pc.optimizers:
        for loss_name in pc.optimizers[group]:
            a = pc.optimizers[group][loss_name].param_groups[0]["params"]
            b = pcvi.optimizers[group][loss_name].param_groups[0]["params"]
            for pa, pb in zip(a, b):
                assert torch.equal(pa, pb), f"{group}/{loss_name} diverged"


def test_pc_lambda_survives_a_checkpoint(config):
    experiment = _experiment(
        config,
        PcConfig.get_from_yaml(),
        clip_grad_val=None,
        collect_with_grad=True,
    )
    experiment.run()
    optimizer = experiment.optimizers["agents"]["loss_objective"]
    state_dict = experiment.state_dict()
    assert state_dict["optimizer_agents_loss_objective"]["lambda_k"] == optimizer.lambda_k

    reloaded = _experiment(
        config, PcConfig.get_from_yaml(), clip_grad_val=None, collect_with_grad=True
    )
    reloaded.load_state_dict(state_dict)
    assert reloaded.optimizers["agents"]["loss_objective"].lambda_k == optimizer.lambda_k


def test_every_registry_config_names_its_own_yaml():
    """Several configs share one optimizer class, so the yaml name has to come
    from the config class, not from associated_class()."""
    for name, config_class in optimizer_config_registry.items():
        loaded = config_class.get_from_yaml()
        assert type(loaded) is config_class, name
    assert PcConfig.get_from_yaml().lambda_0 != PcviConfig.get_from_yaml().lambda_0, (
        "pc and pcvi appear to be reading the same yaml"
    )


# --------------------------------------------------- the extragradient variants


@pytest.mark.parametrize(
    "config_class", [ExtragradientConfig, AdaptiveExtragradientConfig]
)
def test_extragradient_runs_with_beta_k_pinned_to_one(config, config_class):
    experiment = _experiment(config, config_class.get_from_yaml(), clip_grad_val=None)
    assert experiment.two_point_optimizers
    experiment.run()
    training_td = experiment._optimizer_loop_two_point("agents")
    for loss_name in ("loss_objective", "loss_critic"):
        assert training_td[f"pcvi_beta_k_{loss_name}"].item() == 1.0
    for optimizers in experiment.optimizers.values():
        for optimizer in optimizers.values():
            assert isinstance(optimizer, Pcvi)
            assert optimizer.use_contraction is False
            assert optimizer.n_steps > 0


def test_extragradient_keeps_lambda_fixed_and_adaptive_one_does_not(config):
    fixed = _experiment(config, ExtragradientConfig.get_from_yaml(), clip_grad_val=None)
    fixed.run()
    optimizer = fixed.optimizers["agents"]["loss_objective"]
    assert optimizer.lambda_k == optimizer.lambda_0

    adaptive_config = AdaptiveExtragradientConfig.get_from_yaml()
    adaptive_config.lambda_0 = 1.0  # well above p/L, so Step 3 must fire
    adaptive = _experiment(config, adaptive_config, clip_grad_val=None)
    adaptive.run()
    optimizer = adaptive.optimizers["agents"]["loss_objective"]
    assert optimizer.lambda_k < optimizer.lambda_0


@pytest.mark.parametrize(
    "config_class", [ExtragradientConfig, AdaptiveExtragradientConfig]
)
def test_extragradient_refuses_gradient_clipping(config, config_class):
    with pytest.raises(ValueError, match="clip_grad_val"):
        _experiment(config, config_class.get_from_yaml(), clip_grad_val=5)


def test_the_four_grid_cells_are_distinct_configs():
    """{fixed, adaptive} lambda x {Step 5, beta_k = 1}, one code path."""
    cells = {
        "extragradient": (False, False),
        "adaptive_extragradient": (True, False),
        "pc": (False, True),
        "pcvi": (True, True),
    }
    for name, (adaptive, contraction) in cells.items():
        kwargs = optimizer_config_registry[name].get_from_yaml()._optimizer_kwargs(None)
        assert kwargs["use_adaptive_lambda"] is adaptive, name
        assert kwargs["use_contraction"] is contraction, name
        assert optimizer_config_registry[name].associated_class() is Pcvi, name
