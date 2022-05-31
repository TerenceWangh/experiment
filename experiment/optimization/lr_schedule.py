"""Learning rate schedule classes."""

import math
import tensorflow as tf
from typing import Mapping, Any, Union, Optional

BaseSchedule = tf.keras.optimizers.schedules.LearningRateSchedule


def _make_offset_wrapper(new_cls_name: str, base_lr_cls):
  """Generates a offset wrapper of learning rate schedule.

  It will returns a subclass of the the `base_lr_class`, the subclass takes an
  `offset` argument in the constructor. When the new class instance is called,
  the behavior is:
    new_class_object(step) = base_lr_class_object(step - offset)

  *******

  Example:

  ::

    CosineDecayWithOffset = _make_offset_wrapper(
                     'CosineDecayWithOffset', tf.keras.experimental.CosineDecay)
    # Use the lr:
    lr = CosineDecayWithOffset(offset=100, initial_learning_rate=0.1,
                               decay_steps=1000)
    lr(101) # equals to tf.keras.experimental.CosineDecay(...)(101-100)

  :param new_cls_name: the name of the new class.
  :param base_lr_cls: the base learning rate schedule class. Should be subclass
  of tf.keras.optimizers.schedules.LearningRateSchedule
  :return: A new class (subclass of the base_lr_cls) that can take an offset.
  """
  assert issubclass(base_lr_cls, BaseSchedule), (
         'base_lr_cls should be subclass of keras LearningRateSchedule, got '
         '{}'.format(base_lr_cls))

  # pylint: disable=protected-access,pointless-statement
  def offset_learning_rate_init(self, offset=0, **kwargs):
    """Construct learning rate schedule object.

    When this object is called, its behavior is
       self.__call__(step) == base_lr_class.__call__(step - offset)

    :param self: this object.
    :param offset: The offset when computing the learning rate schedule.
    :param kwargs: Pass through to base learning rate class constructor.
    """
    base_lr_cls.__init__(self, **kwargs)
    self._offset = offset

  def offset_learning_rate_call(self, step):
    step = tf.cast(step - self._offset, tf.float32)
    return base_lr_cls.__call__(self, step)

  # pylint: enable=protected-access,pointless-statement

  return type(
      new_cls_name, (base_lr_cls,), {
        'base_lr_class': base_lr_cls,
        '__init__': offset_learning_rate_init,
        '__call__': offset_learning_rate_call,
  })


PiecewiseConstantDecayWithOffset = _make_offset_wrapper(
    'PiecewiseConstantDecayWithOffset',
    tf.keras.optimizers.schedules.PiecewiseConstantDecay)
PolynomialDecayWithOffset = _make_offset_wrapper(
    'PolynomialDecayWithOffset', tf.keras.optimizers.schedules.PolynomialDecay)
ExponentialDecayWithOffset = _make_offset_wrapper(
    'ExponentialDecayWithOffset',
    tf.keras.optimizers.schedules.ExponentialDecay)
CosineDecayWithOffset = _make_offset_wrapper(
    'CosineDecayWithOffset',
    tf.keras.experimental.CosineDecay)


class LinearWarmup(BaseSchedule):
  """Linear warmup schedule."""

  def __init__(self,
               after_warmup_lr_schedule: Union[BaseSchedule, float],
               warmup_steps: int,
               warmup_learning_rate: float,
               name: Optional[str] = None):
    """Add linear warmup schedule to a learning rate schedule.

    warmup_lr is the initial learning rate, the final learning rate of the
    init_warmup period is the initial learning rate of lr_schedule in use.
    The learning rate at each step linearly increased according to the following
    formula:
      learning_rate = warmup_lr + step / warmup_steps
                    * (final_warmup_lr - warmup_lr).
    Using warmup overrides the learning rate schedule by the number of warmup
    steps.

    :param after_warmup_lr_schedule: learning rate or schedules.
    :param warmup_steps: Number of the warmup steps.
    :param warmup_learning_rate: Initial learning rate for the warmup.
    :param name: Optional, name of warmup schedule.
    """
    super(LinearWarmup, self).__init__()
    self._name = name
    self._after_warmup_lr_schedule = after_warmup_lr_schedule
    self._warmup_step = warmup_steps
    self._init_warmup_lr = warmup_learning_rate

    if isinstance(after_warmup_lr_schedule, BaseSchedule):
      self._final_warmup_lr = after_warmup_lr_schedule(warmup_steps)
    else:
      self._final_warmup_lr = tf.cast(after_warmup_lr_schedule,
                                      dtype=tf.float32)

  def __call__(self, step: int):
    global_step = tf.cast(step, dtype=tf.float32)
    linear_warmup_lr = (self._init_warmup_lr + global_step / self._warmup_step *
                        (self._final_warmup_lr - self._init_warmup_lr))

    if isinstance(self._after_warmup_lr_schedule, BaseSchedule):
      after_warmup_lr = self._after_warmup_lr_schedule(step)
    else:
      after_warmup_lr = tf.cast(self._after_warmup_lr_schedule,
                                dtype=tf.float32)

    lr = tf.cond(global_step < self._warmup_step,
                 lambda: linear_warmup_lr,
                 lambda: after_warmup_lr)

    return lr

  def get_config(self) -> Mapping[str, Any]:
    if isinstance(self._after_warmup_lr_schedule, BaseSchedule):
      config = {
        # pylint: disable=attribute-error
        'after_warmup_lr_schedule': self._after_warmup_lr_schedule.get_config()
        # pylint: enable=attribute-error
      }
    else:
      config = {
        'after_warmup_lr_schedule': self._after_warmup_lr_schedule
      }

    config.update({
      'warmup_steps': self._warmup_step,
      'warmup_learning_rate': self._init_warmup_lr,
      'name': self._name,
    })
    return config


class PolynomialWarmup(BaseSchedule):
  """Applies polynomial warmup schedule on a given learning rate decay schedule.
  """

  def __init__(self,
               after_warmup_lr_schedule: Union[BaseSchedule, float],
               warmup_steps: int,
               power: float = 1.0,
               name: str = 'PolynomialWarmup'):
    super(PolynomialWarmup, self).__init__()
    if isinstance(after_warmup_lr_schedule, BaseSchedule):
      self._initial_learning_rate = after_warmup_lr_schedule(warmup_steps)
    else:
      self._initial_learning_rate = tf.cast(after_warmup_lr_schedule,
                                            dtype=tf.float32)

    self._warmup_steps = warmup_steps
    self._power = power
    self._after_warmup_lr_schedule = after_warmup_lr_schedule
    self._name = name

  def __call__(self, step: int):
    with tf.name_scope(self._name or 'PolynomialWarmup') as name:
      # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
      # learning rate will be `global_step/num_warmup_steps * init_lr`.
      global_step_float = tf.cast(step, tf.float32)
      warmup_steps_float = tf.cast(self._warmup_steps, tf.float32)

      if self._warmup_steps <= 0:
        warmup_percent_done = 1.0
      else:
        # A zero `step` may cause Inf, so make `step` positive.
        step_non_zero = tf.math.maximum(global_step_float, 1.0)
        warmup_percent_done = step_non_zero / warmup_steps_float

      warmup_learning_rate = (
          self._initial_learning_rate *
          tf.math.pow(warmup_percent_done, self._power))

      if isinstance(self._after_warmup_lr_schedule, BaseSchedule):
        after_warmup_lr = self._after_warmup_lr_schedule(step)
      else:
        after_warmup_lr = tf.cast(self._after_warmup_lr_schedule,
                                  dtype=tf.float32)

      return tf.cond(
          global_step_float < warmup_steps_float,
          lambda: warmup_learning_rate,
          lambda: after_warmup_lr,
          name=name)

  def get_config(self) -> Mapping[str, Any]:
    if isinstance(self._after_warmup_lr_schedule, BaseSchedule):
      config = {
        # pylint: disable=attribute-error
        'after_warmup_lr_schedule': self._after_warmup_lr_schedule.get_config()
        # pylint: enable=attribute-error
      }
    else:
      config = {
        'after_warmup_lr_schedule': self._after_warmup_lr_schedule
      }

    config.update({
      'warmup_steps': self._warmup_steps,
      'power': self._power,
      'name': self._name,
    })
    return  config


class DirectPowerDecay(BaseSchedule):
  """Learning rate schedule follows lr * (step)^power."""

  def __init__(self,
               initial_learning_rate: float,
               power: float = 1.0,
               name: str = 'DirectPowerDecay'):
    """Initialize configuration of the learning rate schedule.

    :param initial_learning_rate: The initial learning rate.
    :param power: The order of the polynomial.
    :param name: Optional, name of learning rate schedule.
    """
    super(DirectPowerDecay, self).__init__()
    self._initial_learning_rate = initial_learning_rate
    self._power = power
    self._name = name

  def __call__(self, step: int):
    with tf.name_scope(self._name or 'DirectPowerDecay'):
      step = tf.cast(step, tf.float32)
      learning_rate = self._initial_learning_rate
      # A zero `step` may cause Inf. So make `step` positive.
      step_non_zero = tf.math.maximum(step, 1.0)
      learning_rate *= tf.math.pow(step_non_zero, self._power)
      return learning_rate

  def get_config(self):
    """Get the configuration of the learning rate schedule."""
    return {
      'initial_learning_rate': self._initial_learning_rate,
      'power': self._power,
      'name': self._name,
    }


class PowerAndLinearDecay(BaseSchedule):
  """Learning rate schedule with multiplied by linear decay at the end.

  The schedule has the following behavior.

  ----------

    | Let **offset_step = step - offset**.
    | 1. offset_step < 0, the actual learning rate equals initial_learning_rate.
    | 2. offset_step <= total_decay_steps * (1 - linear_decay_fraction), the
      actual learning rate equals lr * offset_step^power.
    | 3. total_decay_steps * (1 - linear_decay_fraction) <= offset_step <
      total_decay_steps, the actual learning rate equals lr * offset_step^power
      * (total_decay_steps - offset_step) / (total_decay_steps *
      linear_decay_fraction).
    | 4. offset_step >= total_decay_steps, the actual learning rate equals zero.
  """

  def __init__(self,
               initial_learning_rate: float,
               total_decay_steps: int,
               power: float = 1.0,
               linear_decay_fraction: float = 0.1,
               offset: int = 0,
               name: str = 'PowerAndLinearDecay'):
    """Initialize configuration of the learning rate schedule.

    :param initial_learning_rate: The initial learning rate.
    :param total_decay_steps: The total number of steps for power + linear
    decay.
    :param power: The order of the polynomial.
    :param linear_decay_fraction: In the last `linear_decay_fraction` steps, the
    learning rate will be multiplied by a linear decay.
    :param offset: The offset applied to steps.
    :param name: Optional, name of learning rate schedule.
    """
    super(PowerAndLinearDecay, self).__init__()
    self._initial_learning_rate = initial_learning_rate
    self._total_decay_steps = total_decay_steps
    self._power = power
    self._linear_decay_fraction = linear_decay_fraction
    self._offset = offset
    self._name = name

  def __call__(self, step: int):
    with tf.name_scope(self._name or 'PowerAndLinearDecay'):
      step = tf.cast(step - self._offset, tf.float32)
      learning_rate = self._initial_learning_rate
      # A zero `step` may cause Inf. So make `step` positive.
      step_non_zero = tf.math.maximum(step, 1.0)
      learning_rate *= tf.math.pow(step_non_zero, self._power)
      if self._total_decay_steps * self._linear_decay_fraction > 0:
        learning_rate *= tf.minimum(
            1.0, (self._total_decay_steps - step) /
                 (self._total_decay_steps * self._linear_decay_fraction))
        learning_rate = tf.maximum(0.0, learning_rate)
      return learning_rate

  def get_config(self) -> Mapping[str, Any]:
    return {
      'initial_learning_rate': self._initial_learning_rate,
      'total_decay_steps': self._total_decay_steps,
      'power': self._power,
      'linear_decay_fraction': self._linear_decay_fraction,
      'offset': self._offset,
      'name': self._name,
    }


class PowerDecayWithOffset(BaseSchedule):
  """Power learning rate decay with offset.

  Learning rate equals to `pre_offset_learning_rate` if `step` < `offset`.
  Otherwise, learning rate equals to `lr * (step - offset)^power`.
  """

  def __init__(self,
               initial_learning_rate: float,
               power: float = 1.0,
               offset: int = 0,
               pre_offset_learning_rate: float = 1.0e6,
               name: str = 'PowerDecayWithOffset'):
    """Initialize configuration of the learning rate schedule.

    :param initial_learning_rate: The initial learning rate.
    :param power: The order of the polynomial.
    :param offset: The offset when computing the power decay.
    :param pre_offset_learning_rate: The maximum learning rate we'll use.
    :param name: Optional, name of learning rate schedule.
    """
    super(PowerDecayWithOffset, self).__init__()
    self._initial_learning_rate = initial_learning_rate
    self._power = power
    self._offset = offset
    self._pre_offset_learning_rate = pre_offset_learning_rate
    self._name = name

  def __call__(self, step: int):
    with tf.name_scope(self._name or 'PowerDecayWithOffset'):
      step = tf.cast(step, tf.float32)
      lr_after_offset = tf.math.pow(
          tf.math.maximum(step - self._offset, 1,0), self._power) * (
        self._initial_learning_rate)
      sign = tf.cast(step > self._offset, tf.float32)
      lr_combined = (1.0 - sign) * self._pre_offset_learning_rate + \
                    sign * lr_after_offset
      return tf.math.minimum(lr_combined, self._pre_offset_learning_rate)

  def get_config(self) -> Mapping[str, Any]:
    return {
      'initial_learning_rate': self._initial_learning_rate,
      'power': self._power,
      'offset': self._offset,
      'pre_offset_learning_rate': self._pre_offset_learning_rate,
      'name': self._name,
    }


class StepCosineDecayWithOffset(BaseSchedule):
  """Stepwise cosine learning rate decay with offset.

  Learning rate is equivalent to one or more cosine decay(s) starting and ending
  at each interval.

  *******

  Example:
    ::

      boundaries: [100000, 110000]
      values: [1.0, 0.5]
      lr_decayed_fn = (
        lr_schedule.StepCosineDecayWithOffset(
          boundaries,
          values))

    - from 0 to 100000 step, it will cosine decay from 1.0 to 0.5
    - from 100000 to 110000 step, it cosine decay from 0.5 to 0.0
  """

  def __init__(self,
               boundaries,
               values,
               offset: int = 0,
               name: str = 'StepCosineDecayWithOffset'):
    """Initialize configuration of the learning rate schedule.

    :param boundaries: A list of `Tensor`s or `int`s with strictly increasing
    entries, and with all elements having the same type as the optimizer step.
    :param values: A list of `Tensor`s or `float`s that specifies the values for
    the intervals defined by `boundaries`. It should have one more element than
    `boundaries`, and all elements should have the same type.
    :param offset: The offset when computing the power decay.
    :param name: Optional, name of learning rate schedule.
    """
    super(StepCosineDecayWithOffset, self).__init__()
    self._values = values
    self._boundaries = boundaries
    self._offset = offset
    self._name = name

    if len(self._values) < 1:
      raise ValueError('Expect non empty of values.')
    if len(self._boundaries) != len(self._values):
      raise ValueError('Boundaries length is equal to learning rate levels '
                       'length {} != {}'.format(len(boundaries), len(values)))

    self._total_steps = (
      [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)] +
      [0])

  def __call__(self, step: int):
    with tf.name_scope(self._name or 'StepCosineDecayWithOffset'):
      global_step = tf.cast(step - self._offset, tf.float32)
      lr_levels = self._values
      lr_steps = self._boundaries
      level_total_steps = self._total_steps
      num_levels = len(lr_levels)

      init_lr = lr_levels[0]
      next_init_lr = lr_levels[1] if num_levels > 1 else 0
      init_total_steps = level_total_steps[0]

      cosine_learning_rate = ((init_lr - next_init_lr) * (
          tf.cos(tf.constant(math.pi) * global_step  / init_total_steps) +
          1.0) / 2.0 + next_init_lr)
      learning_rate = cosine_learning_rate
      tf.compat.v1.logging.info('DEBUG lr %r next lr %r',
                                learning_rate, cosine_learning_rate)
      tf.compat.v1.logging.info('DEBUG lr %r nex lr %r init total step %r',
                                init_lr, next_init_lr, init_total_steps)

      for i in range(1, num_levels):
        next_init_lr = lr_levels[i]
        next_start_step = lr_steps[i]
        next_total_steps = level_total_steps[i]
        next_next_init_lr = lr_levels[i + 1] if num_levels > i + 1 else 0

        tf.compat.v1.logging.info(
            'DEBUG step %r nilr %r nss %r nts %r nnilr %r',
            step, next_init_lr, next_total_steps, next_next_init_lr)
        next_cosine_learning_rate = (
            (next_init_lr - next_next_init_lr) * (
            tf.cos(tf.constant(math.pi) * (global_step - next_start_step) /
                next_total_steps) + 1.0) / 2.0 + next_next_init_lr)
        learning_rate = tf.where(global_step >= next_start_step,
                                 next_cosine_learning_rate, learning_rate)
        tf.compat.v1.logging.info('DEBUG lr %r next lr %r',
                                  learning_rate, next_cosine_learning_rate)

      return learning_rate

  def get_config(self) -> Mapping[str, Any]:
    return {
      'boundaries': self._boundaries,
      'values': self._values,
      'offset': self._offset,
      'name': self._name,
    }
