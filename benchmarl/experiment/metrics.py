#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Task-specific evaluation metrics, as :class:`~benchmarl.experiment.Callback`s.

* :class:`NashDistanceCallback` -- exploitability and distance to the Nash
  equilibrium, for the two-player zero-sum matrix games.
* :class:`WinRateCallback` -- win rate and catch rate for predator-prey tasks
  such as ``vmas/simple_tag`` and ``vmas/simple_adversary``.

Both read the evaluation rollouts only, so they add no cost to training and work
with any algorithm or optimizer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from tensordict import TensorDictBase

from benchmarl.experiment.callback import Callback


def _flatten_leading(tensor: torch.Tensor, n_keep: int = 1) -> torch.Tensor:
    """Collapses everything but the last ``n_keep`` dimensions."""
    return tensor.reshape(-1, *tensor.shape[-n_keep:]) if n_keep else tensor.reshape(-1)


class NashDistanceCallback(Callback):
    """Distance to Nash equilibrium for a two-player zero-sum matrix game.

    Reports, at every evaluation:

    ``dist_nash``
        ``0.5 * sum_i ||pi_i - pi_i^*||_1``, the mean total-variation distance
        between each player's empirical action distribution and its Nash
        strategy. 0 at equilibrium, 1 for a deterministic policy against a
        uniform Nash.
    ``nash_conv``
        ``sum_i [ max_a u_i(a, pi_-i) - u_i(pi) ]``, the summed exploitability of
        the joint policy: how much each player could gain by switching to its
        best response while the other stays put. This is the standard measure and
        is 0 **iff** the joint policy is a Nash equilibrium. Computed exactly from
        the payoff matrix given the two empirical distributions.
    ``exploitability_<player>``
        each player's own term of ``nash_conv``.
    ``action_prob_<player>_<action name>``
        the empirical distribution itself, which is what you plot to see whether
        rock-paper-scissors has actually converged to uniform.

    The distributions are **empirical**: they are the action frequencies over the
    evaluation rollouts. The games are stateless, so a policy is a fixed
    distribution and the estimate is unbiased; its noise is the usual
    ``~1/sqrt(n)`` with ``n`` the number of sampled rounds, which
    ``n_samples_<player>`` reports so a small evaluation budget cannot be mistaken
    for a converged policy.

    Args:
        game (str, optional): key into
            :data:`~benchmarl.environments.matrixgame.matrix_game.PAYOFFS`. If
            ``None``, taken from the experiment's task name at setup.
    """

    def __init__(self, game: Optional[str] = None):
        super().__init__()
        self.game = game
        self.last_stats: Dict[str, float] = {}

    def on_setup(self):
        from benchmarl.environments.matrixgame.matrix_game import (
            ACTION_NAMES,
            NASH_EQUILIBRIA,
            PAYOFFS,
            PLAYERS,
        )

        if self.game is None:
            self.game = self.experiment.task.name.lower()
        if self.game not in PAYOFFS:
            raise ValueError(
                f"NashDistanceCallback does not know the game {self.game!r}. "
                f"Available: {sorted(PAYOFFS)}"
            )
        # getattr: the unit tests drive this callback with a stub config that has
        # no such field, and absent means "not deterministic".
        if getattr(self.experiment.config, "evaluation_deterministic_actions", False):
            raise ValueError(
                "NashDistanceCallback needs stochastic evaluation, but "
                "experiment.evaluation_deterministic_actions is True (BenchMARL's "
                "default). With it on, the evaluation policy takes the argmax "
                "action, so the empirical distribution this callback measures is "
                "one-hot no matter what the policy has learnt. Every mixed Nash "
                "equilibrium then looks maximally far away: on rock-paper-scissors "
                "dist_nash sits at exactly 2/3 -- its largest possible value -- and "
                "nash_conv at 2.0, for the whole run. Set "
                "experiment.evaluation_deterministic_actions=False."
            )
        device = self.experiment.config.train_device
        self.players = list(PLAYERS)
        self.payoff = torch.tensor(PAYOFFS[self.game], dtype=torch.float32).to(device)
        self.nash = [
            torch.tensor(pi, dtype=torch.float32).to(device)
            for pi in NASH_EQUILIBRIA[self.game]
        ]
        self.action_names = ACTION_NAMES[self.game]
        self.n_actions = self.payoff.shape[0]

    def _empirical_policy(self, rollouts: List[TensorDictBase], player: str):
        counts = torch.zeros(self.n_actions, device=self.payoff.device)
        for rollout in rollouts:
            actions = rollout.get((player, "action")).reshape(-1).long()
            counts += torch.bincount(actions, minlength=self.n_actions).to(counts)
        total = counts.sum()
        if total == 0:
            return None, 0
        return counts / total, int(total.item())

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        policies, samples = [], []
        for player in self.players:
            policy, n = self._empirical_policy(rollouts, player)
            if policy is None:
                return
            policies.append(policy)
            samples.append(n)

        pi0, pi1 = policies
        # zero sum: player 1's payoff matrix is -payoff
        value_0 = pi0 @ self.payoff @ pi1  # u_0(pi)
        # best response values
        best_0 = (self.payoff @ pi1).max()  # max_a u_0(a, pi_1)
        best_1 = (-(pi0 @ self.payoff)).max()  # max_b u_1(pi_0, b)
        exploitability = [(best_0 - value_0).item(), (best_1 + value_0).item()]

        # total-variation distance to the Nash strategy, averaged over players
        dist = [
            0.5 * torch.abs(policy - nash).sum().item()
            for policy, nash in zip(policies, self.nash)
        ]

        to_log: Dict[str, Any] = {
            "eval/dist_nash": sum(dist) / len(dist),
            "eval/nash_conv": sum(exploitability),
        }
        for player, expl, n, policy in zip(
            self.players, exploitability, samples, policies
        ):
            to_log[f"eval/exploitability_{player}"] = expl
            to_log[f"eval/n_samples_{player}"] = float(n)
            for name, probability in zip(self.action_names, policy.tolist()):
                to_log[f"eval/action_prob_{player}_{name}"] = probability

        self.last_stats = {key.removeprefix("eval/"): value
                           for key, value in to_log.items()}

        self.experiment.logger.log(to_log, step=self.experiment.n_iters_performed)


class WinRateCallback(Callback):
    """Win rate and catch rate for a predator faction in predator-prey tasks.

    Reports, at every evaluation:

    ``win_rate``
        fraction of evaluation episodes in which the predator faction caught the
        prey at least ``min_catches_to_win`` times.
    ``catch_rate``
        mean over episodes of the fraction of timesteps at which a catch
        occurred.
    ``catches_per_episode``
        mean number of catches per episode.

    **How a catch is detected.** On ``vmas/simple_tag`` with the shipped task
    config (``shape_adversary_rew: False``), an adversary's reward at a timestep
    is exactly ``10 * (number of adversary-prey collisions at that timestep)``,
    so a strictly positive reward is an exact catch signal and the count is
    ``reward / 10``. Nothing has to be added to the environment.

    This breaks if the reward is shaped: with ``shape_adversary_rew: True`` the
    reward also contains a distance term and is positive for reasons that are not
    catches. The callback checks the task config where it can and refuses to run
    rather than report a silently wrong number.

    Args:
        predator_group (str): the group whose win rate is measured. Defaults to
            ``"adversary"``, the predator group of ``vmas/simple_tag``
            (``vmas/simple_world_comm`` also uses ``"adversary"``).
        catch_reward (float): the per-collision reward bonus. 10.0 in VMAS MPE.
        min_catches_to_win (int): catches needed for an episode to count as a win.
    """

    def __init__(
        self,
        predator_group: str = "adversary",
        catch_reward: float = 10.0,
        min_catches_to_win: int = 1,
    ):
        super().__init__()
        self.predator_group = predator_group
        self.catch_reward = catch_reward
        self.min_catches_to_win = min_catches_to_win
        self.last_stats: Dict[str, float] = {}

    def on_setup(self):
        if self.predator_group not in self.experiment.group_map:
            raise ValueError(
                f"WinRateCallback: no group {self.predator_group!r} in this task "
                f"(groups: {sorted(self.experiment.group_map)}). Pass the right "
                f"predator_group."
            )
        config = getattr(self.experiment.task, "config", {}) or {}
        if config.get("shape_adversary_rew", False):
            raise ValueError(
                "WinRateCallback detects a catch from a strictly positive adversary "
                "reward, which is only exact when the reward is unshaped. This task "
                "has shape_adversary_rew=True, so the reward also contains a "
                "distance term and would be positive without any catch. Set "
                "shape_adversary_rew=False, or measure catches another way."
            )

    def episode_stats(self, rollout: TensorDictBase) -> Dict[str, float]:
        """Win / catch statistics for a single evaluation episode."""
        reward = rollout.get(("next", self.predator_group, "reward"))
        # (time, n_agents, 1) -> per-timestep faction reward. The group shares its
        # reward (adversaries_share_rew), so any agent's entry will do; take the
        # max so the statistic is right either way.
        per_step = reward.reshape(reward.shape[0], -1).max(dim=-1).values
        catches = torch.clamp(per_step / self.catch_reward, min=0.0).round()
        n_catches = catches.sum().item()
        n_steps = catches.shape[0]
        return {
            "won": float(n_catches >= self.min_catches_to_win),
            "catch_rate": float((catches > 0).float().mean().item()),
            "catches": float(n_catches),
            "steps": float(n_steps),
        }

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        if not len(rollouts):
            return
        stats = [self.episode_stats(rollout) for rollout in rollouts]
        n = len(stats)
        to_log = {
            "eval/win_rate": sum(s["won"] for s in stats) / n,
            "eval/catch_rate": sum(s["catch_rate"] for s in stats) / n,
            "eval/catches_per_episode": sum(s["catches"] for s in stats) / n,
            "eval/n_episodes": float(n),
        }
        # Kept so a runner can put a meaningful number in its summary table.
        # `mean_return` cannot: it averages over groups, and simple_tag's two
        # groups earn +10 and -10 for the same collision, so it sits at ~0
        # however well the predators learn.
        self.last_stats = {key.removeprefix("eval/"): value
                           for key, value in to_log.items()}
        self.experiment.logger.log(to_log, step=self.experiment.n_iters_performed)
