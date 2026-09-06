#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

"""Two-player zero-sum normal-form games as a batched TorchRL environment.

Rock-paper-scissors and matching pennies are the standard sanity checks for
multi-agent learning: they are tiny, they have a **unique** Nash equilibrium in
mixed strategies (the uniform distribution for both games), and no deterministic
policy can be any good. That makes them the natural place to measure how far a
learned joint policy sits from equilibrium -- see
:class:`~benchmarl.experiment.metrics.NashDistanceCallback`.

The game is **stateless**: every round is independent and the observation is a
constant. An episode is ``max_steps`` independent rounds. This matters for the
Nash analysis: with no observation to condition on, each player's policy *is* a
fixed distribution over actions, so exploitability can be computed exactly from
the payoff matrix rather than approximated.

Each player is its own group (``player_0``, ``player_1``), so the two learn
independent policies even when ``share_policy_params`` is on. Sharing one policy
between the two sides of matching pennies would be meaningless.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

PLAYERS = ("player_0", "player_1")

#: Payoff matrices for player 0. Player 1's payoff is the negative (zero sum).
#: ``PAYOFFS[game][i][j]`` is player 0's payoff when it plays ``i`` and player 1
#: plays ``j``.
PAYOFFS: Dict[str, List[List[float]]] = {
    # rock, paper, scissors -- rock beats scissors, and so on
    "rock_paper_scissors": [
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 1.0, 0.0],
    ],
    # heads, tails -- player 0 wins on a match, player 1 on a mismatch
    "matching_pennies": [
        [1.0, -1.0],
        [-1.0, 1.0],
    ],
}

#: The unique Nash equilibrium of each game: uniform for both players.
NASH_EQUILIBRIA: Dict[str, List[List[float]]] = {
    "rock_paper_scissors": [[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]],
    "matching_pennies": [[0.5, 0.5], [0.5, 0.5]],
}

ACTION_NAMES: Dict[str, List[str]] = {
    "rock_paper_scissors": ["rock", "paper", "scissors"],
    "matching_pennies": ["heads", "tails"],
}


class MatrixGameEnv(EnvBase):
    """A batched, repeated two-player zero-sum matrix game.

    Args:
        game (str): a key of :data:`PAYOFFS`.
        num_envs (int): batch size.
        max_steps (int): rounds per episode. Each round is independent.
        device: torch device.
        seed (int, optional): seed.
    """

    def __init__(
        self,
        game: str,
        num_envs: int = 1,
        max_steps: int = 32,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        if game not in PAYOFFS:
            raise ValueError(f"Unknown game {game!r}. Available: {sorted(PAYOFFS)}")
        super().__init__(device=device, batch_size=torch.Size([num_envs]))

        self.game = game
        self.max_steps = max_steps
        self.payoff = torch.tensor(PAYOFFS[game], device=device, dtype=torch.float32)
        self.n_actions = self.payoff.shape[0]

        self._make_specs()
        self._step_count = torch.zeros(
            num_envs, dtype=torch.int64, device=self.device
        )
        self._set_seed(seed)

    # ------------------------------------------------------------------ specs
    def _make_specs(self):
        batch = self.batch_size
        # One agent per group, hence the trailing 1 in every agent-dimension.
        self.observation_spec = Composite(
            {
                player: Composite(
                    {
                        # stateless game: a constant, so the policy is a fixed
                        # distribution over actions
                        "observation": Unbounded(
                            shape=(*batch, 1, 1), device=self.device
                        )
                    },
                    shape=(*batch, 1),
                )
                for player in PLAYERS
            },
            shape=batch,
        )
        self.action_spec = Composite(
            {
                player: Composite(
                    {
                        "action": Categorical(
                            self.n_actions, shape=(*batch, 1), device=self.device
                        )
                    },
                    shape=(*batch, 1),
                )
                for player in PLAYERS
            },
            shape=batch,
        )
        self.reward_spec = Composite(
            {
                player: Composite(
                    {"reward": Unbounded(shape=(*batch, 1, 1), device=self.device)},
                    shape=(*batch, 1),
                )
                for player in PLAYERS
            },
            shape=batch,
        )
        self.done_spec = Composite(
            {
                key: Categorical(
                    2, shape=(*batch, 1), dtype=torch.bool, device=self.device
                )
                for key in ("done", "terminated", "truncated")
            },
            shape=batch,
        )

    # ------------------------------------------------------------------ logic
    def _observation(self) -> Dict[str, TensorDict]:
        zero = torch.zeros(*self.batch_size, 1, 1, device=self.device)
        return {
            player: TensorDict(
                {"observation": zero.clone()}, batch_size=(*self.batch_size, 1)
            )
            for player in PLAYERS
        }

    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs):
        if tensordict is not None and "_reset" in tensordict.keys():
            mask = tensordict.get("_reset").reshape(self.batch_size)
            self._step_count = torch.where(
                mask, torch.zeros_like(self._step_count), self._step_count
            )
        else:
            self._step_count = torch.zeros_like(self._step_count)

        out = TensorDict(self._observation(), batch_size=self.batch_size)
        false = torch.zeros(
            *self.batch_size, 1, dtype=torch.bool, device=self.device
        )
        for key in ("done", "terminated", "truncated"):
            out.set(key, false.clone())
        return out

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # (num_envs,) each
        a0 = tensordict.get((PLAYERS[0], "action")).reshape(self.batch_size)
        a1 = tensordict.get((PLAYERS[1], "action")).reshape(self.batch_size)
        payoff_0 = self.payoff[a0.long(), a1.long()]  # zero sum: player 1 gets -this

        self._step_count = self._step_count + 1
        truncated = (self._step_count >= self.max_steps).reshape(*self.batch_size, 1)
        terminated = torch.zeros_like(truncated)

        out = TensorDict(self._observation(), batch_size=self.batch_size)
        for player, sign in zip(PLAYERS, (1.0, -1.0)):
            out.set(
                (player, "reward"),
                (sign * payoff_0).reshape(*self.batch_size, 1, 1),
            )
        out.set("terminated", terminated)
        out.set("truncated", truncated.clone())
        out.set("done", truncated.clone())
        return out

    def _set_seed(self, seed: Optional[int]):
        self.rng = torch.Generator(device=self.device)
        if seed is not None:
            self.rng.manual_seed(int(seed))

    # ------------------------------------------------------------------ utils
    @property
    def group_map(self) -> Dict[str, List[str]]:
        return {player: [player] for player in PLAYERS}

    def nash_equilibrium(self) -> List[torch.Tensor]:
        """The unique Nash equilibrium, one distribution per player."""
        return [
            torch.tensor(pi, device=self.device, dtype=torch.float32)
            for pi in NASH_EQUILIBRIA[self.game]
        ]
