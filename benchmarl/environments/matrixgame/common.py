#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import copy
from typing import Callable, Dict, List, Optional

from torchrl.data import Composite
from torchrl.envs import EnvBase

from benchmarl.environments.common import Task, TaskClass
from benchmarl.environments.matrixgame.matrix_game import MatrixGameEnv, PLAYERS
from benchmarl.utils import DEVICE_TYPING


class MatrixGameClass(TaskClass):
    """Two-player zero-sum normal-form games (see :mod:`~.matrix_game`)."""

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        return lambda: MatrixGameEnv(
            game=self.name.lower(),
            num_envs=num_envs,
            device=device,
            seed=seed,
            **config,
        )

    def supports_continuous_actions(self) -> bool:
        return False

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        return False

    def max_steps(self, env: EnvBase) -> int:
        return self.config["max_steps"]

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return {player: [player] for player in PLAYERS}

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def observation_spec(self, env: EnvBase) -> Composite:
        return env.full_observation_spec_unbatched.clone()

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec_unbatched

    @staticmethod
    def env_name() -> str:
        return "matrixgame"


class MatrixGameTask(Task):
    """Enum for the matrix games."""

    ROCK_PAPER_SCISSORS = None
    MATCHING_PENNIES = None

    @staticmethod
    def associated_class():
        return MatrixGameClass
