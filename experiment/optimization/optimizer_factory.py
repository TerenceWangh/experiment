"""Optimizer factory class."""

from typing import Callable, Optional, Union, List, Tuple

import gin
import tensorflow as tf

from experiment.optimization import ema_optimizer
from experiment.optimization import lamb_optimizer
from experiment.optimization import lars_optimizer
from experiment.optimization import legacy_adamw
from experiment.optimization import lr_schedule
from experiment.optimization.configuration import optimization_config as opt_cfg

# pylint: disable=bad-whitespace
OPTIMIZERS_CLS = {
    'lamb'    : lamb_optimizer.LAMB,
    'lars'    : lars_optimizer.LARS,
    'sgd'     : tf.keras.optimizers.SGD,
    'adam'    : tf.keras.optimizers.Adam,
    'adamw'   : legacy_adamw.AdamWeightDecay,
    'rmsprop' : tf.keras.optimizers.RMSprop,
    'adagrad' : tf.keras.optimizers.Adagrad,
}

LR_CLS = {
    'stepwise'                : lr_schedule.PiecewiseConstantDecayWithOffset,
    'polynomial'              : lr_schedule.PolynomialDecayWithOffset,
    'exponential'             : lr_schedule.ExponentialDecayWithOffset,
    'cosine'                  : lr_schedule.CosineDecayWithOffset,
    'power'                   : lr_schedule.DirectPowerDecay,
    'power_linear'            : lr_schedule.PowerAndLinearDecay,
    'power_with_offset'       : lr_schedule.PowerDecayWithOffset,
    'step_cosine_with_offset' : lr_schedule.StepCosineDecayWithOffset,
}

WARMUP_CLS = {
    'linear'    : lr_schedule.LinearWarmup,
    'polynomial': lr_schedule.PolynomialWarmup,
}
# pylint: enable=bad-whitespace


def register_optimizer_cls(key: str,
                           optimizer_config_cls: tf.keras.optimizers.Optimizer):
  """Register customize optimizer cls.

  The user will still need to subclass data classes in
  configs.optimization_config to be used with OptimizerFactory.

  :param key: A string to that the optimizer_config_cls is registered with.
  :param optimizer_config_cls: A class which inherits
  `tf.keras.optimizers.Optimizer`.
  """
  if key in OPTIMIZERS_CLS:
    raise ValueError('{} already registered in OPTIMIZERS_CLS.'.format(key))
  OPTIMIZERS_CLS[key] = optimizer_config_cls


class OptimizerFactory:
  """Optimizer factory class.

  This class builds learning rate and optimizer based on an optimization config.
  To use this class, you need to do the following:

    | 1. Define optimization config, this includes optimizer, and learning rate
      schedule.
    | 2. Initialize the class using the optimization config.
    | 3. Build learning rate.
    | 4. Build Optimizer.

  This is a typical example for using this class:
  ::

    params = {
      'optimizer': {
        'type': 'sgd',
        'sgd': { 'momentum': 0.9 }
      },
      'learning_rate': {
        'type': 'stepwise',
        'stepwise': {
          'boundaries': [ 10000, 20000 ],
          'values': [ 0.1, 0.01, 0.001 ]
        }
      },
      'warmup': {
        'type': linear,
        'linear': {
          'warmup_steps': 500,
          'warmup_learning_rate': 0.01
        }
      }
    }
    opt_config = OptimizationConfig(params)
    opt_factory = OptimizerFactory(opt_config)
    lr = opt_factory.build_learning_rate()
    optimizer = opt_factory.build_optimizer(lr)
  """

  def __init__(self, config: opt_cfg.OptimizationConfig):
    """Initializing OptimizerFactory.

    :param config: OptimizationConfig instance contain optimization config.
    """
    self._config = config
    self._optimizer_config = config.optimizer.get()
    self._optimizer_type = config.optimizer.type

    self._use_ema = config.ema is not None
    self._ema_config = config.ema

    if self._optimizer_config is None:
      raise ValueError('Optimizer type must be specified.')

    self._lr_config = config.learning_rate.get()
    self._lr_type = config.learning_rate.type

    if self._lr_type is None:
      raise ValueError('Learning rate type must be specified.')

    self._warmup_config = config.warmup.get()
    self._warmup_type = config.warmup.type

  def build_learning_rate(self):
    """Build learning rate.

    Builds learning rate from config. Learning rate schedule is built according
    to the learning rate config. If learning rate type is constant, lr_config.
    learning_rate is returned.

    :return: tf.keras.optimizers.schedules.LearningRateSchedule instance. If
    learning rate type is constant, lr_config.learning_rate is returned.
    """
    if self._lr_type == 'constant':
      lr = self._lr_config.learning_rate
    else:
      lr = LR_CLS[self._lr_type](**self._lr_config.as_dict())

    if self._warmup_config:
      lr = WARMUP_CLS[self._warmup_type](lr, **self._warmup_config.as_dict())
    return lr

  @gin.configurable
  def build_optimizer(
      self,
      lr: Union[tf.keras.optimizers.schedules.LearningRateSchedule, float],
      gradient_aggregator: Optional[Callable[
          [List[Tuple[tf.Tensor, tf.Tensor]]],
          List[Tuple[tf.Tensor, tf.Tensor]]]] = None,
      gradient_transformers: Optional[List[Callable[
          [List[Tuple[tf.Tensor, tf.Tensor]]],
          List[Tuple[tf.Tensor, tf.Tensor]]]]] = None,
      postprocessor: Optional[Callable[[tf.keras.optimizers.Optimizer],
                                       tf.keras.optimizers.Optimizer]] = None):
    """Build optimizer.

    Builds optimizer from config. It takes learning rate as input, and builds
    the optimizer according to the optimizer config. Typically, the learning
    rate built using self.build_lr() is passed as an argument to this method.

    :param lr: A floating point value, or a
    `tf.keras.optimizers.schedules.LearningRateSchedule` instance.
    :param gradient_aggregator: Optional function to overwrite gradient
    aggregation.
    :param gradient_transformers: Optional list of functions to use to transform
    gradients before applying updates to Variables. The functions are applied
    after gradient_aggregator. The functions should accept and return a list of
    (gradient, variable) tuples. clipvalue, clipnorm, global_clipnorm should not
    be set when gradient_transformers is passed.
    :param postprocessor: An optional function for postprocessing the optimizer.
    It takes an optimizer and returns an optimizer.
    :return: `tf.keras.optimizers.Optimizer` instance.
    """
    optimizer_dict = self._optimizer_config.as_dict()

    # Delete clipnorm, clipvalue, global_clipnorm if None
    if optimizer_dict['clipnorm'] is None:
      del optimizer_dict['clipnorm']
    if optimizer_dict['clipvalue'] is None:
      del optimizer_dict['clipvalue']
    if optimizer_dict['global_clipnorm'] is None:
      del optimizer_dict['global_clipnorm']

    optimizer_dict['learning_rate'] = lr
    if gradient_aggregator is not None:
      optimizer_dict['gradient_aggregator'] = gradient_aggregator
    if gradient_transformers is not None:
      optimizer_dict['gradient_transformers'] = gradient_transformers

    optimizer = OPTIMIZERS_CLS[self._optimizer_type](**optimizer_dict)
    if self._use_ema:
      optimizer = ema_optimizer.ExponentialMovingAverage(
          optimizer, **self._ema_config.as_dict())
    if postprocessor:
      optimizer = postprocessor(optimizer)
    assert isinstance(
        optimizer, tf.keras.optimizers.Optimizer
    ), ('OptimizerFactory.build_optimizer returning a non-optimizer object: '
        '{}'.format(optimizer))

    return optimizer
