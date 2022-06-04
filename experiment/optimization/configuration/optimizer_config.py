"""Dataclasses for optimizer configs."""
import dataclasses
from typing import List, Optional

from experiment.hyperparams import base_config


@dataclasses.dataclass
class BaseOptimizerConfig(base_config.Config):
  """Base optimizer config.
  """
  clipnorm: Optional[float] = None
  clipvalue: Optional[float] = None
  global_clipnorm: Optional[float] = None


@dataclasses.dataclass
class SGDConfig(BaseOptimizerConfig):
  """Configuration for SGD optimizer.

  The attributes for this class matches the arguments of tf.keras.optimizer.SGD.
  """
  name: str = 'SGD'
  decay: float = 0.0
  nesterov: bool = False
  momentum: float = 0.0


@dataclasses.dataclass
class SGDExperimentalConfig(BaseOptimizerConfig):
  """Configuration for SGD optimizer.

  The attributes for this class matches the argument of
  `tf.keras.experimental.SGD`.
  """
  name: str = 'SGD'
  nesterov: bool = False
  momentum: float = 0.0
  jit_compile: bool = False


@dataclasses.dataclass
class RMSPropConfig(BaseOptimizerConfig):
  """Configuration for RMSProp optimizer.

  The attributes for this class matches the arguments of
  tf.keras.optimizers.RMSprop.
  """
  name: str = 'RMSprop'
  rho = float = 0.9
  momentum: float = 0.0
  epsilon: float = 1e-7
  centered: bool = False


@dataclasses.dataclass
class AdagradConfig(BaseOptimizerConfig):
  """Configuration for Adagrad optimizer.

  The attributes of this class match the arguments of
  tf.keras.optimizer.Adagrad.
  """
  name: str = 'Adagrad'
  initial_accumulator_value: float = 0.1
  epsilon: float = 1e-07


@dataclasses.dataclass
class AdamConfig(BaseOptimizerConfig):
  """Configuration for Adam optimizer.

  The attributes for this class matches the arguments of
  tf.keras.optimizer.Adam.
  """
  name: str = 'Adam'
  beta_1: float = 0.9
  beta_2: float = 0.999
  epsilon: float = 1e-07
  amsgrad: bool = False


@dataclasses.dataclass
class AdamExperimentalConfig(BaseOptimizerConfig):
  """Configuration for experimental Adam optimizer.

  The attributes for this class matches the arguments of
  `tf.keras.optimizer.experimental.Adam`.
  """
  name: str = 'Adam'
  beta_1: float = 0.9
  beta_2: float = 0.999
  epsilon: float = 1e-07
  amsgrad: bool = False
  jit_compile: bool = False


@dataclasses.dataclass
class AdamWeightDecayConfig(BaseOptimizerConfig):
  """Configuration for Adam optimizer with weight decay.
  """
  name: str = 'AdamWeightDecay'
  beta_1: float = 0.9
  beta_2: float = 0.999
  epsilon: float = 1e-07
  amsgrad: bool = False
  weight_decay: float = 0.0
  include_weight_decay: Optional[List[str]] = None
  exclude_weight_decay: Optional[List[str]] = None
  gradient_clip_norm: float = 1.0


@dataclasses.dataclass
class LAMBConfig(BaseOptimizerConfig):
  """Configuration for LAMB optimizer.

  The attributes for this class matches the arguments of
  tensorflow_addons.optimizers.LAMB.
  """
  name: str = 'LAMB'
  beta_1: float = 0.9
  beta_2: float = 0.999
  epsilon: float = 1e-6
  weight_decay: float = 0.0
  exclude_weight_decay: Optional[List[str]] = None
  exclude_layer_adaptation: Optional[List[str]] = None


@dataclasses.dataclass
class EMAConfig(BaseOptimizerConfig):
  """Exponential moving average optimizer config.
  """
  name: str = 'ExponentialMovingAverage'
  trainable_weights_only: bool = True
  average_decay: float = 0.99
  start_step: int = 0
  dynamic_decay: bool = True


@dataclasses.dataclass
class LARSConfig(BaseOptimizerConfig):
  """Layer-wise adaptive rate scaling config.
  """
  name: str = 'LARS'
  momentum: float = 0.9
  eeta: float = 0.001
  weight_decay: float = 0.0
  nesterov: bool = False
  classic_momentum: bool = True
  exclude_from_weight_decay: Optional[List[str]] = None
  exclude_from_layer_adaptation: Optional[List[str]] = None
