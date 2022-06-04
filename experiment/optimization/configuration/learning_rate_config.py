"""Dataclasses for learning rate schedule config."""
from typing import List, Optional

import dataclasses
from experiment.hyperparams import base_config


@dataclasses.dataclass
class ConstantLrConfig(base_config.Config):
  """Configuration for constant learning rate.

  This class is a containers for the constant learning rate decay configuration.
  """
  name: str = 'Constant'
  learning_rate: float = 0.1


@dataclasses.dataclass
class StepwiseLrConfig(base_config.Config):
  """Configuration for stepwise learning rate decay.

  This class is a container for the piecewise constant learning rate scheduling
  configuration. It will configure an instance of PiecewiseConstantDecay keras
  learning rate schedule.

  An example (from keras docs): use a learning rate that's 1.0 for the first
  100001 steps, 0.5 for the next 10000 steps, and 0.1 for any additional steps.

  Examples
  ========
  >>> boundaries: [100000, 110000]
  >>> values: [1.0, 0.5, 0.1]
  """
  name: str = 'PiecewiseConstantDecay'
  boundaries: Optional[List[int]] = None
  values: Optional[List[float]] = None
  offset: int = 0


@dataclasses.dataclass
class ExponentialLrConfig(base_config.Config):
  """Configuration for exponential learning rate decay.

  This class is a containers for the exponential learning rate decay
  configuration.
  """
  name: str = 'ExponentialDecay'
  initial_learning_rate: Optional[float] = None
  decay_steps: Optional[int] = None
  decay_rate: Optional[float] = None
  staircase: Optional[bool] = None
  offset: int = 0


@dataclasses.dataclass
class PolynomialLrConfig(base_config.Config):
  """Configuration for polynomial learning rate decay.

  This class is a containers for the polynomial learning rate decay
  configuration.
  """
  name: str = 'PolynomialDecay'
  initial_learning_rate: Optional[float] = None
  decay_steps: Optional[int] = None
  end_learning_rate: float = 0.0001
  power: float = 1.0
  cycle: bool = False
  offset: int = 0


@dataclasses.dataclass
class CosineLrConfig(base_config.Config):
  """Configuration for Cosine learning rate decay.

  This class is a containers for the cosine learning rate decay configuration,
  tf.keras.experimental.CosineDecay.
  """
  name: str = 'CosineDecay'
  initial_learning_rate: Optional[float] = None
  decay_steps: Optional[int] = None
  alpha: float = 0.0
  offset: int = 0


@dataclasses.dataclass
class DirectPowerLrConfig(base_config.Config):
  """Configuration for DirectPower learning rate decay.

  This class configures a schedule following follows

  .. math:: lr * (step)^power.
  """
  name: str = 'DirectPowerDecay'
  initial_learning_rate: Optional[float] = None
  power: float = -0.5


@dataclasses.dataclass
class PowerAndLinearDecayLrConfig(base_config.Config):
  """Configuration for DirectPower learning rate decay.

  The schedule has the following behaviors.

  Let ``offset_step = step - offset``.

  * offset_step < 0, the actual learning rate equals initial_learning_rate.
  * offset_step <= total_decay_steps * (1 - linear_decay_fraction), the
    actual learning rate equals lr * offset_step^power.
  * total_decay_steps * (1 - linear_decay_fraction) <= offset_step <
    total_decay_steps, the actual learning rate equals lr * offset_step^power *
    (total_decay_steps - offset_step) / (total_decay_steps *
    linear_decay_fraction).
  * offset_step >= total_decay_steps, the actual learning rate equals zero.
  """
  name: str = 'PowerAndLinearDecay'
  initial_learning_rate: Optional[float] = None
  total_decay_steps: Optional[int] = None
  power: float = -0.5
  linear_decay_fraction: float = 0.1
  offset: int = 0


@dataclasses.dataclass
class PowerDecayWithOffsetLrConfig(base_config.Config):
  """Configuration for power learning rate decay with step offset.

  Learning rate equals to `pre_offset_learning_rate` if `step` < `offset`.
  Otherwise, learning rate equals to `lr * (step - offset)^power`.
  """
  name: str = 'PowerDecayWithOffset'
  initial_learning_rate: Optional[float] = None
  power: float = -0.5
  offset: int = 0
  pre_offset_learning_rate: float = 1.0e6


@dataclasses.dataclass
class StepCosineLrConfig(base_config.Config):
  """Configuration for stepwise learning rate decay.

  This class is a container for the piecewise cosine learning rate scheduling
  configuration. It will configure an instance of StepCosineDecayWithOffset
  keras learning rate schedule.

  Example
  =======
  >>> boundaries: [100000, 110000]
  >>> values: [1.0, 0.5]
  >>> lr_decayed_fn = (
  >>> lr_schedule.StepCosineDecayWithOffset(
  >>>     boundaries,
  >>>     values))

  .. note:: from 0 to 100000 step, it will cosine decay from 1.0 to 0.5
  .. note:: from 100000 to 110000 step, it cosine decay from 0.5 to 0.0
  """
  name: str = 'StepCosineDecayWithOffset'
  boundaries: Optional[List[int]] = None
  values: Optional[List[float]] = None
  offset: int = 0


@dataclasses.dataclass
class LinearWarmupConfig(base_config.Config):
  """Configuration for linear warmup schedule config.

  This class is a container for the linear warmup schedule configuration.
  Warmup_learning_rate is the initial learning rate, the final learning rate of
  the warmup period is the learning_rate of the optimizer in use. The learning
  rate at each step linearly increased according to the following formula:

  .. math::
    warmup\_learning\_rate = warmup\_learning\_rate +
    step / warmup\_steps * (final\_learning\_rate - warmup\_learning\_rate).

  Using warmup overrides the learning rate schedule by the number of warmup
  steps.
  """
  name: str = 'linear'
  warmup_learning_rate: float = 0
  warmup_steps: Optional[int] = None


@dataclasses.dataclass
class PolynomialWarmupConfig(base_config.Config):
  """Configuration for linear warmup schedule config.

  This class is a container for the polynomial warmup schedule configuration.
  """
  name: str = 'polynomial'
  power: float = 1
  warmup_steps: Optional[int] = None
