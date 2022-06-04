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
  """
  type: Optional[str] = None
  linear: lr_cfg.LinearWarmupConfig = lr_cfg.LinearWarmupConfig()
  polynomial: lr_cfg.PolynomialWarmupConfig = lr_cfg.PolynomialWarmupConfig()


@dataclasses.dataclass
class OptimizationConfig(base_config.Config):
  """Configuration for optimizer and learning rate schedule.
  """
  optimizer: OptimizerConfig = OptimizerConfig()
  ema: Optional[opt_cfg.EMAConfig] = None
  learning_rate: LrConfig = LrConfig()
  warmup: WarmupConfig = WarmupConfig()
