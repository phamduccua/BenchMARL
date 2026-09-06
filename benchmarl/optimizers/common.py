#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import pathlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Type

import torch

from benchmarl.utils import _read_yaml_config


@dataclass
class OptimizerConfig(ABC):
    """
    Dataclass representing an optimizer configuration.
    This should be overridden by implemented optimizers.

    It mirrors the structure of :class:`~benchmarl.algorithms.common.AlgorithmConfig`:
    implementors should

        1. add the configuration parameters of their optimizer as fields
        2. implement :meth:`associated_class`

    The field names of the dataclass have to match exactly the argument names of
    ``associated_class().__init__`` (excluding ``params``) and the keys of the
    associated yaml file in ``benchmarl/conf/optimizer/``.
    """

    def get_optimizer(self, params: Iterable, experiment_config) -> torch.optim.Optimizer:
        """
        Main function to turn the config into the associated optimizer.

        Args:
            params (iterable): the parameters to optimize
            experiment_config (ExperimentConfig): the experiment config, for
                optimizers that read their hyperparameters from it

        Returns: the optimizer

        """
        return self.associated_class()(params, **self._optimizer_kwargs(experiment_config))

    def _optimizer_kwargs(self, experiment_config) -> Dict[str, Any]:
        """
        The kwargs passed to ``associated_class()``. By default, all the fields of
        the dataclass.
        """
        return dict(self.__dict__)  # Passes all the custom config parameters

    @staticmethod
    @abstractmethod
    def associated_class() -> Type[torch.optim.Optimizer]:
        """
        The optimizer class associated to the config
        """
        raise NotImplementedError

    def requires_two_gradient_evals(self) -> bool:
        """
        Whether ``optimizer.step()`` needs a ``closure`` that recomputes the
        gradient at a probe point.

        Optimizers returning ``True`` are driven by
        :meth:`~benchmarl.experiment.Experiment._optimizer_loop_two_point`
        instead of the standard single-gradient loop.
        """
        return False

    def uses_gradient_as_operator(self) -> bool:
        """
        Whether the optimizer treats the gradient as the operator ``F`` of a
        monotone-operator method rather than as a mere descent direction.

        Such optimizers are incompatible with gradient clipping: a clipped
        gradient is not the gradient of anything, which invalidates the local
        Lipschitz estimate they are built on. The experiment refuses to run them
        unless ``clip_grad_val`` is ``None``.
        """
        return False

    @staticmethod
    def _load_from_yaml(name: str) -> Dict[str, Any]:
        yaml_path = (
            pathlib.Path(__file__).parent.parent
            / "conf"
            / "optimizer"
            / f"{name.lower()}.yaml"
        )
        return _read_yaml_config(str(yaml_path.resolve()))

    @classmethod
    def get_from_yaml(cls, path: Optional[str] = None):
        """
        Load the optimizer configuration from yaml

        Args:
            path (str, optional): The full path of the yaml file to load from.
                If None, it will default to
                ``benchmarl/conf/optimizer/self.associated_class().__name__``

        Returns: the loaded OptimizerConfig
        """
        if path is None:
            # From the config class name, not from associated_class(): several
            # configs can share one optimizer class (PcConfig and PcviConfig both
            # build a Pcvi), and each still needs its own yaml.
            name = cls.__name__
            if name.endswith("Config"):
                name = name[: -len("Config")]
            # CamelCase -> snake_case: AdaptiveExtragradient -> adaptive_extragradient
            name = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
            config = OptimizerConfig._load_from_yaml(name=name)
        else:
            config = _read_yaml_config(path)
        return cls(**config)
