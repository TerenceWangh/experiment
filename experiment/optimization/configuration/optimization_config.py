"""Dataclasses for optimization configs.

This file define the dataclass for optimization configs (OptimizationConfig).
It also has two helper functions get_optimizer_config, and get_lr_config from
an OptimizationConfig class.
"""
import dataclasses
from typing import Optional

from experiment.hyperparams import base_config, oneof
from experiment.optimization.configuration import learning_rate_config as \
  lr_cfg
from experiment.optimization.configuration import optimizer_config as opt_cfg


@dataclasses.dataclass
class OptimizerConfig(oneof.OneOfConfig):
  """Configuration for optimizer.

  Attributes:
  -----------
  - type: Type of optimizer to be used, on the of fields below.
  - sgd: sgd optimizer config
  - adam: adam optimizer config.
  - adamw: adam with weight decay.
  - lamb: lamb optimizer.
  - rmsprop: rmsprop optimizer.
  - lars: lars optimizer.
  - adagrad: adagrad optimizer.
  - slide: slide optimizer.
  """
  type: Optional[str] = None
  sgd: opt_cfg.SGDConfig = opt_cfg.SGDConfig()
  sgd_experimental: opt_cfg.SGDExperimentalConfig = (
    opt_cfg.SGDExperimentalConfig())
  adam: opt_cfg.AdamConfig = opt_cfg.AdamConfig()
  adam_experimental: opt_cfg.AdamExperimentalConfig = (
    opt_cfg.AdamExperimentalConfig())
  adamw: opt_cfg.AdamWeightDecayConfig = opt_cfg.AdamWeightDecayConfig()
  lamb: opt_cfg.LAMBConfig = opt_cfg.LAMBConfig()
  rmsprop: opt_cfg.RMSPropConfig = opt_cfg.RMSPropConfig()
  lars: opt_cfg.LARSConfig = opt_cfg.LARSConfig()
  adagrad: opt_cfg.AdagradConfig = opt_cfg.AdagradConfig()


@dataclasses.dataclass
class LrConfig(oneof.OneOfConfig):
  """Configuration for lr schedule.

  Attributes:
  -----------
  - type: type of lr schedule to be used, one of the fields below.
  - constant: constant learning rate config.
  - stepwise: stepwise learning rate config.
  - exponential: exponential learning rate config.
  - polynomial: polynomial learning rate config.
  - cosine: cosine learning rate config.
  - power: step^power learning rate config.
  - power_linear: learning rate config of `step^power` followed by
    `step^power*linear`.
  - power_with_offset: power decay with a step offset.
  - step_cosine_with_offset: Step cosine with a step offset.
  """
  type: Optional[str] = None
  constant: lr_cfg.ConstantLrConfig = lr_cfg.ConstantLrConfig()
  stepwise: lr_cfg.StepwiseLrConfig = lr_cfg.StepwiseLrConfig()
  exponential: lr_cfg.ExponentialLrConfig = lr_cfg.ExponentialLrConfig()
  polynomial: lr_cfg.PolynomialLrConfig = lr_cfg.PolynomialLrConfig()
  cosine: lr_cfg.CosineLrConfig = lr_cfg.CosineLrConfig()
  power: lr_cfg.DirectPowerLrConfig = lr_cfg.DirectPowerLrConfig()
  power_linear: lr_cfg.PowerAndLinearDecayLrConfig = (
    lr_cfg.PowerAndLinearDecayLrConfig())
  power_with_offset: lr_cfg.PowerDecayWithOffsetLrConfig = (
    lr_cfg.PowerDecayWithOffsetLrConfig())
  step_cosine_with_offset: lr_cfg.StepCosineLrConfig = (
    lr_cfg.StepCosineLrConfig())


@dataclasses.dataclass
class WarmupConfig(oneof.OneOfConfig):
  """Configuration for lr schedule.

  Attributes:
  -----------
  - type: 'str', type of warmup schedule to be used, one of the fields below.
  - linear: linear warmup config.
  - polynomial: polynomial warmup config.
  """
  type: Optional[str] = None
  linear: lr_cfg.LinearWarmupConfig = lr_cfg.LinearWarmupConfig()
  polynomial: lr_cfg.PolynomialWarmupConfig = lr_cfg.PolynomialWarmupConfig()


@dataclasses.dataclass
class OptimizationConfig(base_config.Config):
  """Configuration for optimizer and learning rate schedule.

  Attributes:
  -----------
  - optimizer: optimizer oneof config.
  - ema: optional exponential moving average optimizer config, if specified,
    ema optimizer will be used.
  - learning_rate: learning rate oneof config.
  - warmup: warmup oneof config.
  """
  optimizer: OptimizerConfig = OptimizerConfig()
  ema: Optional[opt_cfg.EMAConfig] = None
  learning_rate: LrConfig = LrConfig()
  warmup: WarmupConfig = WarmupConfig()
