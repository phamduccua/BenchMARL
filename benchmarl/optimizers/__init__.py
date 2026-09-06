#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from .adam import AdamConfig
from .adam_cosine import AdamCosine, AdamCosineConfig
from .adaptive import Adaptive, AdaptiveConfig, AdaptivePlusConfig
from .lookahead import Lookahead, LookaheadConfig
from .common import OptimizerConfig
from .sgd import SgdConfig
from .pcvi import (
    AdaptiveExtragradientConfig,
    ExtragradientConfig,
    PcConfig,
    PcPlusConfig,
    Pcvi,
    PcviConfig,
    PcviPlusConfig,
)

classes = [
    "AdamConfig",
    "AdamCosine",
    "AdamCosineConfig",
    "Adaptive",
    "AdaptiveConfig",
    "AdaptivePlusConfig",
    "AdaptiveExtragradientConfig",
    "ExtragradientConfig",
    "Lookahead",
    "LookaheadConfig",
    "PcConfig",
    "PcPlusConfig",
    "Pcvi",
    "PcviConfig",
    "PcviPlusConfig",
    "SgdConfig",
]

# A registry mapping "optimizername" to its config dataclass
# This is used to aid loading of optimizers from yaml
optimizer_config_registry = {
    "adam": AdamConfig,
    "adam_cosine": AdamCosineConfig,
    "adaptive": AdaptiveConfig,
    "adaptive_plus": AdaptivePlusConfig,
    "extragradient": ExtragradientConfig,
    "adaptive_extragradient": AdaptiveExtragradientConfig,
    "lookahead": LookaheadConfig,
    "pc": PcConfig,
    "pc_plus": PcPlusConfig,
    "pcvi": PcviConfig,
    "pcvi_plus": PcviPlusConfig,
    "sgd": SgdConfig,
}
