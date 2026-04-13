#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

"""Approximate NashConv callback for BenchMARL experiments.

NashConv = Σ_i  max(0, V_i^{BR} - V_i^{π})

where V_i^{BR} is agent i's return when it deviates to a best-response policy
while all other agents keep their current policies.

Usage::

    from benchmarl.algorithms.nashconv_callback import NashConvCallback

    experiment = Experiment(
        ...
        callbacks=[NashConvCallback(
            br_updates=5,
            br_episodes=4,
            eval_episodes=5,
            br_lr=3e-4,
            eval_interval=1,         # compute at every evaluation step
        )],
    )
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDictBase
from torch.distributions import Categorical
from torchrl.envs.utils import step_mdp

from benchmarl.experiment.callback import Callback


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class _BRActor(nn.Module):
    """Lightweight MLP actor for discrete action spaces (Categorical)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def logits(self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.net(obs)
        if mask is not None:
            x = x.masked_fill(~mask, -1e9)
        return x

    def act(self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Returns (action, log_prob, entropy) — all scalar tensors."""
        dist = Categorical(logits=self.logits(obs, mask))
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def greedy_action(self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.logits(obs, mask).argmax(-1)


class _BRActorContinuous(nn.Module):
    """Lightweight MLP actor for continuous action spaces (Gaussian)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        low: Optional[torch.Tensor] = None,
        high: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.register_buffer("low", low if low is not None else torch.full((action_dim,), -1.0))
        self.register_buffer("high", high if high is not None else torch.full((action_dim,), 1.0))
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def _dist(self, obs: torch.Tensor):
        feat = self.net(obs)
        mean = torch.tanh(self.mean_head(feat))
        # Scale mean to [low, high]
        scale = (self.high - self.low) / 2.0
        center = (self.high + self.low) / 2.0
        mean = mean * scale + center
        std = self.log_std.exp().clamp(1e-4, 2.0)
        return torch.distributions.Normal(mean, std)

    def act(self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None):  # noqa: ARG002
        """Returns (action, log_prob, entropy). mask is unused for continuous."""
        dist = self._dist(obs)
        action = dist.rsample()
        action = action.clamp(self.low, self.high)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy

    def greedy_action(self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # noqa: ARG002
        dist = self._dist(obs)
        return dist.mean.clamp(self.low, self.high)


def _extract_agent_obs(td: TensorDictBase, group: str, agent_idx: int, obs_key: str) -> torch.Tensor:
    """Return obs tensor for a single agent: shape (*batch, obs_dim)."""
    obs = td.get((group, obs_key))          # (*batch, n_agents, obs_dim)
    return obs[..., agent_idx, :]


def _extract_agent_mask(
    td: TensorDictBase, group: str, agent_idx: int
) -> Optional[torch.Tensor]:
    mask = td.get((group, "action_mask"), None)
    if mask is None:
        return None
    return mask[..., agent_idx, :]          # (*batch, n_actions)


def _extract_agent_reward(
    td: TensorDictBase, group: str, agent_idx: int, n_agents: int = 0  # noqa: ARG001
) -> float:
    """Scalar per-agent reward for one step."""
    reward = td.get(("next", group, "reward"), None)
    if reward is None:
        # Global shared reward — broadcast to all agents
        reward = td.get(("next", "reward"))
        return float(reward.mean())
    # reward shape: (*batch, n_agents) or (*batch, n_agents, 1)
    r = reward.squeeze(-1) if reward.shape[-1] == 1 else reward
    return float(r[..., agent_idx].mean())


def _is_done(td: TensorDictBase, group: str) -> bool:
    done = td.get(("next", group, "done"), None)
    if done is None:
        done = td.get(("next", "done"), None)
    if done is None:
        return False
    return bool(done.any())


# ─────────────────────────────────────────────────────────────────────────────
# Main callback
# ─────────────────────────────────────────────────────────────────────────────

class NashConvCallback(Callback):
    """Approximate NashConv callback.

    Args:
        br_updates (int): Number of REINFORCE update steps per agent per
            evaluation call.
        br_episodes (int): Episodes collected per BR update step.
        eval_episodes (int): Episodes used to evaluate the BR utility.
        br_lr (float): Learning rate for the BR actor.
        br_hidden_dim (int): Hidden size of the BR MLP actor.
        entropy_coef (float): Entropy bonus coefficient for BR training.
        gamma (float): Discount factor used when computing returns.
        eval_interval (int): Run NashConv every *eval_interval* evaluation
            calls (``1`` = every evaluation).
        deterministic_eval (bool): If True use greedy actions when evaluating
            BR utility (recommended).
        obs_key (str): Key under which agent observations are stored inside
            the group TensorDict (default: ``"observation"``).
    """

    def __init__(
        self,
        br_updates: int = 5,
        br_episodes: int = 4,
        eval_episodes: int = 5,
        br_lr: float = 3e-4,
        br_hidden_dim: int = 64,
        entropy_coef: float = 0.001,
        gamma: float = 0.99,
        eval_interval: int = 1,
        deterministic_eval: bool = True,
        obs_key: str = "observation",
    ):
        super().__init__()
        self.br_updates = br_updates
        self.br_episodes = br_episodes
        self.eval_episodes = eval_episodes
        self.br_lr = br_lr
        self.br_hidden_dim = br_hidden_dim
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.eval_interval = eval_interval
        self.deterministic_eval = deterministic_eval
        self.obs_key = obs_key

        self._eval_count = 0

    # ── Callback entry point ─────────────────────────────────────────────────

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        self._eval_count += 1
        if (self._eval_count - 1) % self.eval_interval != 0:
            return

        exp = self.experiment
        device = exp.config.train_device

        # 1. Base utilities from the existing evaluation rollouts
        base_utils: Dict[str, np.ndarray] = self._base_utils_from_rollouts(rollouts)

        to_log: Dict[str, float] = {}
        nashconv_total = 0.0

        for group in exp.group_map.keys():
            n_agents = len(exp.group_map[group])

            # Observation / action dimensions for this group
            obs_dim = exp.observation_spec[group][self.obs_key].shape[-1]
            action_spec = exp.action_spec[group, "action"]
            is_discrete = hasattr(action_spec.space, "n")
            if is_discrete:
                action_dim = action_spec.space.n
                action_low, action_high = None, None
            else:
                action_dim = action_spec.shape[-1]
                action_low = torch.tensor(action_spec.space.low, dtype=torch.float32)
                action_high = torch.tensor(action_spec.space.high, dtype=torch.float32)

            group_gaps: List[float] = []
            for agent_idx in range(n_agents):
                # 2. Train a best-response actor for agent `agent_idx`
                br_actor = self._train_br_actor(
                    group, agent_idx, obs_dim, action_dim, device,
                    is_discrete=is_discrete,
                    action_low=action_low,
                    action_high=action_high,
                )

                # 3. Evaluate BR utility
                br_util = self._eval_br_utility(
                    group, agent_idx, br_actor, device
                )

                base_util = float(base_utils[group][agent_idx])
                gap = max(0.0, br_util - base_util)
                group_gaps.append(gap)

                to_log[f"eval/nashconv/{group}/gap_agent_{agent_idx}"] = gap
                to_log[f"eval/nashconv/{group}/base_util_agent_{agent_idx}"] = base_util
                to_log[f"eval/nashconv/{group}/br_util_agent_{agent_idx}"] = br_util

            group_nashconv = float(np.sum(group_gaps))
            nashconv_total += group_nashconv
            to_log[f"eval/nashconv/{group}/nashconv"] = group_nashconv
            to_log[f"eval/nashconv/{group}/max_gap"] = float(np.max(group_gaps)) if group_gaps else 0.0

        to_log["eval/nashconv/total"] = nashconv_total
        exp.logger.log(to_log, step=exp.n_iters_performed)

    # ── Utility helpers ──────────────────────────────────────────────────────

    def _base_utils_from_rollouts(
        self, rollouts: List[TensorDictBase]
    ) -> Dict[str, np.ndarray]:
        """Extract per-agent cumulative return from existing rollouts."""
        exp = self.experiment
        result: Dict[str, np.ndarray] = {}
        for group in exp.group_map.keys():
            n_agents = len(exp.group_map[group])
            ep_returns = []
            for td in rollouts:
                reward = td.get(("next", group, "reward"), None)
                if reward is None:
                    # Shared reward: broadcast to all agents
                    reward = td.get(("next", "reward"))
                    per_agent = reward.mean().item()
                    ep_returns.append(np.full(n_agents, per_agent))
                else:
                    r = reward.squeeze(-1) if reward.shape[-1] == 1 else reward
                    # r shape: (T, n_agents)
                    ep_returns.append(r.sum(0).cpu().numpy())   # (n_agents,)
            result[group] = np.mean(ep_returns, axis=0)         # (n_agents,)
        return result

    def _make_br_policy(
        self, group: str, agent_idx: int, br_actor: Union[_BRActor, _BRActorContinuous], deterministic: bool
    ):
        """Return a callable that wraps exp.policy, replacing agent_idx's action."""
        base_policy = self.experiment.policy
        obs_key = self.obs_key

        def policy_fn(td: TensorDictBase) -> TensorDictBase:
            with torch.no_grad():
                td = base_policy(td)

            obs_i = _extract_agent_obs(td, group, agent_idx, obs_key)
            mask_i = _extract_agent_mask(td, group, agent_idx)

            if deterministic:
                action_i = br_actor.greedy_action(obs_i, mask_i)
            else:
                action_i, _, _ = br_actor.act(obs_i, mask_i)

            actions = td.get((group, "action")).clone()
            if isinstance(br_actor, _BRActorContinuous):
                actions[..., agent_idx, :] = action_i
            else:
                actions[..., agent_idx] = action_i
            td.set((group, "action"), actions)
            return td

        return policy_fn

    def _run_episode(
        self, env, policy_fn, group: str, agent_idx: int
    ):
        """Run one episode. Returns (ep_reward, log_probs, entropies, rewards_per_step)."""
        td = env.reset()
        done = False
        ep_rewards: List[float] = []

        while not done:
            # Call base policy to fill all actions
            with torch.no_grad():
                td = self.experiment.policy(td)

            # Let BR actor act for agent_idx (with grad for training)
            obs_i = _extract_agent_obs(td, group, agent_idx, self.obs_key)
            mask_i = _extract_agent_mask(td, group, agent_idx)
            action_i, log_prob_i, entropy_i = policy_fn(obs_i, mask_i)

            actions = td.get((group, "action")).clone()
            if action_i.dim() > 0 and action_i.shape[-1] > 1:
                # continuous: action_i shape (..., action_dim)
                actions[..., agent_idx, :] = action_i.detach()
            else:
                # discrete: action_i is scalar per env
                actions[..., agent_idx] = action_i.detach()
            td.set((group, "action"), actions)

            td = env.step(td)
            ep_rewards.append(_extract_agent_reward(td, group, agent_idx, len(self.experiment.group_map[group])))
            done = _is_done(td, group)
            if not done:
                td = step_mdp(td)

        return ep_rewards, log_prob_i, entropy_i  # note: only last step returned — handled below

    def _train_br_actor(
        self,
        group: str,
        agent_idx: int,
        obs_dim: int,
        action_dim: int,
        device: str,
        is_discrete: bool = True,
        action_low: Optional[torch.Tensor] = None,
        action_high: Optional[torch.Tensor] = None,
    ) -> Union[_BRActor, _BRActorContinuous]:
        """Train a best-response actor for agent `agent_idx` via REINFORCE."""
        exp = self.experiment
        if is_discrete:
            br_actor = _BRActor(obs_dim, action_dim, self.br_hidden_dim).to(device)
        else:
            low = action_low.to(device) if action_low is not None else None
            high = action_high.to(device) if action_high is not None else None
            br_actor = _BRActorContinuous(
                obs_dim, action_dim, self.br_hidden_dim, low=low, high=high
            ).to(device)
        optimizer = optim.Adam(br_actor.parameters(), lr=self.br_lr)

        for _ in range(self.br_updates):
            all_log_probs: List[torch.Tensor] = []
            all_entropies: List[torch.Tensor] = []
            all_returns: List[float] = []

            for _ in range(self.br_episodes):
                env = exp.env_func()
                td = env.reset().to(device)
                done = False
                ep_log_probs: List[torch.Tensor] = []
                ep_entropies: List[torch.Tensor] = []
                ep_rewards: List[float] = []

                while not done:
                    with torch.no_grad():
                        td = exp.policy(td)

                    obs_i = _extract_agent_obs(td, group, agent_idx, self.obs_key)
                    mask_i = _extract_agent_mask(td, group, agent_idx)
                    action_i, log_prob_i, entropy_i = br_actor.act(obs_i, mask_i)

                    actions = td.get((group, "action")).clone()
                    if is_discrete:
                        actions[..., agent_idx] = action_i.detach()
                    else:
                        actions[..., agent_idx, :] = action_i.detach()
                    td.set((group, "action"), actions)

                    td = env.step(td)
                    ep_rewards.append(
                        _extract_agent_reward(td, group, agent_idx, len(exp.group_map[group]))
                    )
                    ep_log_probs.append(log_prob_i)
                    ep_entropies.append(entropy_i)
                    done = _is_done(td, group)
                    if not done:
                        td = step_mdp(td)

                env.close()

                # Discounted returns (Monte-Carlo)
                G = 0.0
                ep_returns: List[float] = []
                for r in reversed(ep_rewards):
                    G = r + self.gamma * G
                    ep_returns.append(G)
                ep_returns.reverse()

                all_log_probs.extend(ep_log_probs)
                all_entropies.extend(ep_entropies)
                all_returns.extend(ep_returns)

            if not all_log_probs:
                continue

            returns = torch.tensor(all_returns, dtype=torch.float32, device=device)
            if returns.numel() > 1:
                returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-6)

            log_probs = torch.stack(all_log_probs)
            entropies = torch.stack(all_entropies)
            loss = -(log_probs * returns).mean() - self.entropy_coef * entropies.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return br_actor

    def _eval_br_utility(
        self,
        group: str,
        agent_idx: int,
        br_actor: Union[_BRActor, _BRActorContinuous],
        device: str,
    ) -> float:
        """Evaluate expected return for agent `agent_idx` when using the BR actor."""
        exp = self.experiment
        ep_returns: List[float] = []

        for _ in range(self.eval_episodes):
            env = exp.env_func()
            td = env.reset().to(device)
            done = False
            ep_ret = 0.0

            while not done:
                with torch.no_grad():
                    td = exp.policy(td)

                    obs_i = _extract_agent_obs(td, group, agent_idx, self.obs_key)
                    mask_i = _extract_agent_mask(td, group, agent_idx)

                    if self.deterministic_eval:
                        action_i = br_actor.greedy_action(obs_i, mask_i)
                    else:
                        action_i, _, _ = br_actor.act(obs_i, mask_i)

                    actions = td.get((group, "action")).clone()
                    if isinstance(br_actor, _BRActorContinuous):
                        actions[..., agent_idx, :] = action_i
                    else:
                        actions[..., agent_idx] = action_i
                    td.set((group, "action"), actions)

                td = env.step(td)
                ep_ret += _extract_agent_reward(td, group, agent_idx, len(exp.group_map[group]))
                done = _is_done(td, group)
                if not done:
                    td = step_mdp(td)

            env.close()
            ep_returns.append(ep_ret)

        return float(np.mean(ep_returns))
