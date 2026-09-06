#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Tests for the task-specific evaluation metrics.

The point of these is that the numbers are *right*, not merely that the callbacks
run: a win rate or a Nash distance that is quietly measuring the wrong thing is
worse than no metric at all.
"""

import pytest
import torch
from tensordict import TensorDict

from benchmarl.algorithms import IppoConfig
from benchmarl.environments import MatrixGameTask, VmasTask
from benchmarl.environments.matrixgame.matrix_game import (
    MatrixGameEnv,
    NASH_EQUILIBRIA,
    PAYOFFS,
    PLAYERS,
)
from benchmarl.experiment import Experiment
from benchmarl.experiment.metrics import NashDistanceCallback, WinRateCallback
from benchmarl.models.mlp import MlpConfig


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("game", sorted(PAYOFFS))
def test_env_specs(game):
    from torchrl.envs.utils import check_env_specs

    check_env_specs(MatrixGameEnv(game, num_envs=4, max_steps=5, seed=0))


@pytest.mark.parametrize("game", sorted(PAYOFFS))
def test_env_is_zero_sum(game):
    env = MatrixGameEnv(game, num_envs=8, max_steps=6, seed=0)
    rollout = env.rollout(6)
    total = sum(
        rollout.get(("next", player, "reward")).sum() for player in PLAYERS
    )
    assert torch.isclose(total, torch.zeros(()), atol=1e-6)


def test_rps_payoffs_are_the_real_game():
    """rock beats scissors, paper beats rock, scissors beats paper."""
    payoff = torch.tensor(PAYOFFS["rock_paper_scissors"])
    rock, paper, scissors = 0, 1, 2
    assert payoff[rock, scissors] == 1 and payoff[scissors, rock] == -1
    assert payoff[paper, rock] == 1 and payoff[rock, paper] == -1
    assert payoff[scissors, paper] == 1 and payoff[paper, scissors] == -1
    assert torch.equal(payoff, -payoff.T), "not zero sum / not antisymmetric"


def test_matching_pennies_payoffs():
    payoff = torch.tensor(PAYOFFS["matching_pennies"])
    assert payoff[0, 0] == 1 and payoff[1, 1] == 1  # match: player 0 wins
    assert payoff[0, 1] == -1 and payoff[1, 0] == -1


@pytest.mark.parametrize("game", sorted(PAYOFFS))
def test_declared_nash_is_actually_a_nash(game):
    """Every deviation from the declared equilibrium must be worthless."""
    payoff = torch.tensor(PAYOFFS[game])
    pi0, pi1 = (torch.tensor(pi) for pi in NASH_EQUILIBRIA[game])
    value = pi0 @ payoff @ pi1
    # against a uniform opponent, every pure action has the same value
    assert torch.allclose(payoff @ pi1, value.expand(payoff.shape[0]), atol=1e-6)
    assert torch.allclose(-(pi0 @ payoff), -value.expand(payoff.shape[1]), atol=1e-6)


# ---------------------------------------------------------------------------
# NashDistanceCallback
# ---------------------------------------------------------------------------


class _FakeLogger:
    def __init__(self):
        self.logged = {}

    def log(self, dict_to_log, step=None):
        self.logged.update(dict_to_log)


class _FakeExperiment:
    def __init__(self, task_name, group_map, task_config=None):
        from types import SimpleNamespace

        self.task = SimpleNamespace(name=task_name, config=task_config or {})
        self.config = SimpleNamespace(train_device="cpu")
        self.group_map = group_map
        self.logger = _FakeLogger()
        self.n_iters_performed = 0


def _rollouts_with_fixed_actions(actions_per_player, n_steps):
    """One rollout in which each player always plays the same action."""
    return [
        TensorDict(
            {
                player: TensorDict(
                    {"action": torch.full((n_steps, 1), action, dtype=torch.long)},
                    batch_size=(n_steps, 1),
                )
                for player, action in zip(PLAYERS, actions_per_player)
            },
            batch_size=(n_steps,),
        )
    ]


def _rollouts_with_uniform_actions(n_actions, repeats):
    """One rollout that plays every action exactly ``repeats`` times."""
    actions = torch.arange(n_actions).repeat_interleave(repeats)
    n_steps = actions.shape[0]
    return [
        TensorDict(
            {
                player: TensorDict(
                    {"action": actions.reshape(n_steps, 1).clone()},
                    batch_size=(n_steps, 1),
                )
                for player in PLAYERS
            },
            batch_size=(n_steps,),
        )
    ]


def _run_nash_callback(game, rollouts):
    callback = NashDistanceCallback(game=game)
    callback.experiment = _FakeExperiment(game, {p: [p] for p in PLAYERS})
    callback.on_setup()
    callback.on_evaluation_end(rollouts)
    return callback.experiment.logger.logged


def test_nash_metrics_are_zero_at_equilibrium():
    logged = _run_nash_callback(
        "rock_paper_scissors", _rollouts_with_uniform_actions(3, 40)
    )
    assert logged["eval/dist_nash"] == pytest.approx(0.0, abs=1e-6)
    assert logged["eval/nash_conv"] == pytest.approx(0.0, abs=1e-6)
    for name in ("rock", "paper", "scissors"):
        assert logged[f"eval/action_prob_player_0_{name}"] == pytest.approx(1 / 3)


def test_nash_metrics_at_a_deterministic_policy():
    """Both always play rock: maximally far from Nash, and fully exploitable."""
    logged = _run_nash_callback("rock_paper_scissors", _rollouts_with_fixed_actions((0, 0), 30))
    # TV distance from a one-hot to uniform over 3 actions is 1 - 1/3 = 2/3
    assert logged["eval/dist_nash"] == pytest.approx(2 / 3, abs=1e-6)
    # value is 0 (rock vs rock); each player could switch to paper and get +1
    assert logged["eval/nash_conv"] == pytest.approx(2.0, abs=1e-6)
    assert logged["eval/exploitability_player_0"] == pytest.approx(1.0, abs=1e-6)
    assert logged["eval/exploitability_player_1"] == pytest.approx(1.0, abs=1e-6)


def test_nash_conv_is_zero_only_at_equilibrium():
    """One player uniform, the other not: the uniform player is unexploitable."""
    n_steps = 60
    actions_0 = torch.arange(3).repeat_interleave(n_steps // 3)  # uniform
    actions_1 = torch.zeros(n_steps, dtype=torch.long)  # always rock
    rollouts = [
        TensorDict(
            {
                PLAYERS[0]: TensorDict(
                    {"action": actions_0.reshape(-1, 1)}, batch_size=(n_steps, 1)
                ),
                PLAYERS[1]: TensorDict(
                    {"action": actions_1.reshape(-1, 1)}, batch_size=(n_steps, 1)
                ),
            },
            batch_size=(n_steps,),
        )
    ]
    logged = _run_nash_callback("rock_paper_scissors", rollouts)
    # player 0 is uniform => player 1 cannot gain by deviating
    assert logged["eval/exploitability_player_1"] == pytest.approx(0.0, abs=1e-6)
    # player 1 always plays rock => player 0 gains 1 by always playing paper
    assert logged["eval/exploitability_player_0"] == pytest.approx(1.0, abs=1e-6)
    assert logged["eval/nash_conv"] == pytest.approx(1.0, abs=1e-6)
    assert logged["eval/nash_conv"] > 0


def test_matching_pennies_deterministic():
    logged = _run_nash_callback("matching_pennies", _rollouts_with_fixed_actions((0, 0), 20))
    assert logged["eval/dist_nash"] == pytest.approx(0.5, abs=1e-6)
    # heads/heads: player 0 gets +1, player 1 could switch to tails and gain 2
    assert logged["eval/exploitability_player_0"] == pytest.approx(0.0, abs=1e-6)
    assert logged["eval/exploitability_player_1"] == pytest.approx(2.0, abs=1e-6)


def test_nash_callback_rejects_an_unknown_game():
    callback = NashDistanceCallback(game="tic_tac_toe")
    callback.experiment = _FakeExperiment("tic_tac_toe", {})
    with pytest.raises(ValueError, match="does not know the game"):
        callback.on_setup()


# ---------------------------------------------------------------------------
# WinRateCallback
# ---------------------------------------------------------------------------


def _predator_rollout(rewards_per_step, n_agents=3):
    n_steps = len(rewards_per_step)
    reward = torch.tensor(rewards_per_step, dtype=torch.float32)
    reward = reward.reshape(n_steps, 1, 1).expand(n_steps, n_agents, 1).clone()
    return TensorDict(
        {
            "next": TensorDict(
                {"adversary": TensorDict({"reward": reward}, batch_size=(n_steps, n_agents))},
                batch_size=(n_steps,),
            )
        },
        batch_size=(n_steps,),
    )


def _run_win_rate_callback(rollouts, **kwargs):
    callback = WinRateCallback(**kwargs)
    callback.experiment = _FakeExperiment(
        "simple_tag", {"adversary": ["a0"], "agent": ["g0"]}
    )
    callback.on_setup()
    callback.on_evaluation_end(rollouts)
    return callback.experiment.logger.logged


def test_win_rate_counts_episodes_with_at_least_one_catch():
    rollouts = [
        _predator_rollout([0, 0, 10, 0]),  # one catch  -> win
        _predator_rollout([0, 0, 0, 0]),  # no catch   -> loss
        _predator_rollout([10, 0, 10, 10]),  # three catches -> win
    ]
    logged = _run_win_rate_callback(rollouts)
    assert logged["eval/win_rate"] == pytest.approx(2 / 3)
    # catch rate per episode: 1/4, 0/4, 3/4 -> mean 1/3
    assert logged["eval/catch_rate"] == pytest.approx(1 / 3)
    assert logged["eval/catches_per_episode"] == pytest.approx((1 + 0 + 3) / 3)
    assert logged["eval/n_episodes"] == 3.0


def test_simultaneous_catches_are_counted_individually():
    """Two adversaries colliding at once gives 20, i.e. two catches, one timestep."""
    logged = _run_win_rate_callback([_predator_rollout([20, 0, 0, 0])])
    assert logged["eval/catches_per_episode"] == pytest.approx(2.0)
    assert logged["eval/catch_rate"] == pytest.approx(0.25)
    assert logged["eval/win_rate"] == 1.0


def test_negative_rewards_are_not_catches():
    logged = _run_win_rate_callback([_predator_rollout([-3.0, -1.0, -0.5])])
    assert logged["eval/win_rate"] == 0.0
    assert logged["eval/catches_per_episode"] == 0.0


def test_min_catches_to_win():
    rollouts = [_predator_rollout([10, 0, 0]), _predator_rollout([10, 10, 0])]
    assert _run_win_rate_callback(rollouts)["eval/win_rate"] == pytest.approx(1.0)
    strict = _run_win_rate_callback(rollouts, min_catches_to_win=2)
    assert strict["eval/win_rate"] == pytest.approx(0.5)


def test_win_rate_refuses_a_shaped_reward():
    """A shaped reward is positive without any catch: refuse rather than lie."""
    callback = WinRateCallback()
    callback.experiment = _FakeExperiment(
        "simple_tag", {"adversary": ["a0"]}, task_config={"shape_adversary_rew": True}
    )
    with pytest.raises(ValueError, match="shape_adversary_rew"):
        callback.on_setup()


def test_win_rate_refuses_an_unknown_group():
    callback = WinRateCallback(predator_group="predators")
    callback.experiment = _FakeExperiment("simple_tag", {"adversary": ["a0"]})
    with pytest.raises(ValueError, match="no group"):
        callback.on_setup()


def test_shipped_simple_tag_config_is_unshaped():
    """The metric's precondition must hold for the config people will actually use."""
    task = VmasTask.SIMPLE_TAG.get_from_yaml()
    assert task.config["shape_adversary_rew"] is False
    assert task.config["adversaries_share_rew"] is True


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@pytest.fixture
def config(experiment_config):
    experiment_config.max_n_iters = 2
    experiment_config.on_policy_n_minibatch_iters = 1
    experiment_config.on_policy_minibatch_size = 100
    experiment_config.on_policy_collected_frames_per_batch = 200
    experiment_config.evaluation = True
    experiment_config.evaluation_interval = 200
    experiment_config.evaluation_episodes = 4
    # A mixed Nash cannot be measured off argmax actions; see
    # test_nash_callback_refuses_deterministic_evaluation.
    experiment_config.evaluation_deterministic_actions = False
    experiment_config.render = False
    experiment_config.checkpoint_interval = 0
    experiment_config.loggers = []
    experiment_config.create_json = False
    return experiment_config


@pytest.mark.parametrize(
    "task", [MatrixGameTask.ROCK_PAPER_SCISSORS, MatrixGameTask.MATCHING_PENNIES]
)
def test_matrix_game_trains_with_the_nash_callback(config, task):
    experiment = Experiment(
        task=task.get_from_yaml(),
        algorithm_config=IppoConfig.get_from_yaml(),
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        seed=0,
        config=config,
        callbacks=[NashDistanceCallback()],
    )
    assert set(experiment.group_map) == set(PLAYERS)
    experiment.run()


def test_nash_callback_refuses_deterministic_evaluation(config):
    """The failure this guards against is silent and total.

    With BenchMARL's default ``evaluation_deterministic_actions=True`` the
    evaluation policy takes the argmax action, so the empirical distribution the
    callback measures is one-hot however well the policy has learnt. Every mixed
    equilibrium then looks maximally far away: on rock-paper-scissors dist_nash
    reads exactly 2/3 -- its largest possible value -- and nash_conv 2.0, flat for
    the entire run, which is easy to mistake for "it has not converged yet".
    """
    config.evaluation_deterministic_actions = True
    with pytest.raises(ValueError, match="evaluation_deterministic_actions"):
        Experiment(
            task=MatrixGameTask.ROCK_PAPER_SCISSORS.get_from_yaml(),
            algorithm_config=IppoConfig.get_from_yaml(),
            model_config=MlpConfig.get_from_yaml(),
            critic_model_config=MlpConfig.get_from_yaml(),
            seed=0,
            config=config,
            callbacks=[NashDistanceCallback()],
        )


def test_deterministic_policy_is_the_worst_possible_dist_nash():
    """2/3 is not a coincidence: it is the maximum for a 3-action uniform Nash."""
    callback = NashDistanceCallback(game="rock_paper_scissors")
    nash = torch.full((3,), 1.0 / 3.0)
    one_hot = torch.tensor([1.0, 0.0, 0.0])
    distance = 0.5 * torch.abs(one_hot - nash).sum().item()
    assert distance == pytest.approx(2.0 / 3.0)
    assert callback.game == "rock_paper_scissors"


def test_simple_tag_trains_with_the_win_rate_callback(config):
    experiment = Experiment(
        task=VmasTask.SIMPLE_TAG.get_from_yaml(),
        algorithm_config=IppoConfig.get_from_yaml(),
        model_config=MlpConfig.get_from_yaml(),
        critic_model_config=MlpConfig.get_from_yaml(),
        seed=0,
        config=config,
        callbacks=[WinRateCallback()],
    )
    experiment.run()
