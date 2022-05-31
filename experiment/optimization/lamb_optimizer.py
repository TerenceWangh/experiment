import re
import warnings

import numpy as np
import tensorflow as tf
from typing import Optional, Union, Callable, List

FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]


class LAMB(tf.keras.optimizers.Optimizer):
  """Optimizer that implements the Layer-wise Adaptive Moments (LAMB).

  See paper [Large Batch Optimization for Deep Learning: Training BERT
  in 76 minutes](https://arxiv.org/abs/1904.00962).

  reference: https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/lamb.py
  """

  def __init__(self,
               learning_rate: Union[FloatTensorLike, Callable] = 0.001,
               beta_1: FloatTensorLike = 0.9,
               beta_2: FloatTensorLike = 0.999,
               epsilon: FloatTensorLike = 1e-6,
               weight_decay: FloatTensorLike = 0.0,
               exclude_weight_decay: Optional[List[str]] = None,
               exclude_layer_adaptation: Optional[List[str]] = None,
               name: str = 'LAMB',
               **kwargs):
    """Construct a new LAMB optimizer.

    :param learning_rate: A `Tensor` or a floating point value. or a schedule
    that is a `tf.keras.optimizers.schedules.LearningRateSchedule` The learning
    rate.
    :param beta_1: A `float` value or a constant `float` tensor. The exponential
    decay rate for the 1st moment estimates.
    :param beta_2: A `float` value or a constant `float` tensor. The exponential
    decay rate for the 2nd moment estimates.
    :param epsilon: A small constant for numerical stability.
    :param weight_decay: weight decay.
    :param exclude_weight_decay: List of regex patterns of variables excluded
    from weight decay. Variables whose name contain a substring matching the
    pattern will be excluded.
    :param exclude_layer_adaptation: List of regex patterns of variables
    excluded from layer adaptation. Variables whose name contain a substring
    matching the pattern will be excluded.
    :param name: Optional name for the operations created when applying
    gradients. Defaults to "LAMB".
    :param kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`,
    `lr`, `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
    gradients by value, `decay` is included for backward compatibility to allow
    time inverse decay of learning rate. `lr` is included for backward
    compatibility, recommended to use `learning_rate` instead.
    """
    super(LAMB, self).__init__(name, **kwargs)

    # Just adding the square of the weights to the loss function is *not*
    # the correct way of using L2 regularization/weight decay with Adam,
    # since that will interact with the m and v parameters in strange ways.
    #
    # Instead we want to decay the weights in a manner that doesn't interact
    # with the m/v parameters.
    self._set_hyper('weight_decay', weight_decay)
    self._set_hyper('learning_rate', learning_rate)

    # This is learning rate decay for using keras learning rate schedule.
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('beta_1', beta_1)
    self._set_hyper('beta_2', beta_2)
    self._epsilon = epsilon or tf.backend_config.epsilon()
    self._exclude_weight_decay = exclude_weight_decay
    # exclude_from_layer_adaptation is set to exclude_from_weight_decay if
    # the arg is None.
    if exclude_layer_adaptation:
      self._exclude_layer_adaptation = exclude_layer_adaptation
    else:
      self._exclude_layer_adaptation = exclude_weight_decay

  def _create_slots(self, var_list):
    # Create slots for the first and second moments.
    # Separate for-loops to respect the ordering of slot variables from v1.
    for var in var_list:
      self.add_slots(var, 'm')
    for var in var_list:
      self.add_slots(var, 'v')

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(LAMB, self)._prepare_local(var_device, var_dtype, apply_state)
    local_step = tf.cast(self.iterations + 1, var_dtype)
    beta1_t = tf.identity(self._get_hyper('beta_1'), var_dtype)
    beta2_t = tf.identity(self._get_hyper('beta_2'), var_dtype)
    weight_decay = tf.identity(self._get_hyper('weight_decay'), var_dtype)
    beta1_p = tf.pow(beta1_t, local_step)
    beta2_p = tf.pow(beta2_t, local_step)
    apply_state[(var_device, var_dtype)].update(
        dict(
            weight_decay = weight_decay,
            epsilon = tf.convert_to_tensor(self._epsilon, var_dtype),
            beta1_t = beta1_t,
            beta1_p = beta1_p,
            one_minus_beta1_t = 1 - beta1_t,
            beta2_t = beta2_t,
            beta2_p = beta2_p,
            one_minus_beta2_t = 1 - beta2_t,
        )
    )

  def _resource_apply_dense(self, grad, var, apply_state = None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = (apply_state or {}).get(
        (var_device, var_dtype)
    ) or self._fallback_apply_state(var_device, var_dtype)

    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slots(var, 'm')
    m_scaled_g_values = grad * coefficients['one_minus_beta1_t']
    m_t = m * coefficients['beta1_t'] + m_scaled_g_values
    m_t = m.assign(m_t, use_locking=self._use_locking)
    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slots(var, 'v')
    v_scaled_g_values = grad * grad * coefficients['one_minus_beta2_t']
    v_t = v * coefficients['beta2_t'] + v_scaled_g_values
    v_t = v.assign(v_t, use_locking=self._use_locking)

    m_t_hat = m_t / (1.0 - coefficients['beta1_p'])
    v_t_hat = v_t / (1.0 - coefficients['beta2_p'])

    v_sqrt = tf.sqrt(v_t_hat)
    update = m_t_hat / (v_sqrt + coefficients['epsilon'])

    if self._do_use_weight_decay(var):
      update += coefficients['weight_decay'] * var

    ratio = 1.0
    if self._do_layer_adaptation(var):
      w_n = tf.norm(var,    ord=2)
      g_n = tf.norm(update, ord=2)
      ratio = tf.where(
          tf.greater(w_n, 0),
          tf.where(tf.greater(g_n, 0), (w_n / g_n), 1.0),
          1.0,
      )
    var_update = var - ratio * coefficients['lr_t'] * update
    return var.assign(var_update, use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices, apply_state):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = (apply_state or {}).get(
        (var_device, var_dtype)
    ) or self._fallback_apply_state(var_device, var_dtype)

    # m_t = beta1 * m + (1 - beta1) * g_t
    m = self.get_slots(var, 'm')
    m_scaled_g_values = grad * coefficients['one_minus_beta1_t']
    m_t = m.assign(m * coefficients['beta1_t'], use_locking=self._use_locking)
    with tf.control_dependencies([m_t]):
      m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

    # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
    v = self.get_slots(var, 'v')
    v_scaled_g_values = grad * grad * coefficients['one_minus_beta2_t']
    v_t = v.assign(v * coefficients['beta2_t'], use_locking=self._use_locking)
    with tf.control_dependencies([v_t]):
      v_t = self.resource_scatter_add(v, indices, v_scaled_g_values)

    m_t_hat = m_t / (1.0 - coefficients['beta1_p'])
    v_t_hat = v_t / (1.0 - coefficients['beta2_p'])

    v_sqrt = tf.sqrt(v_t_hat)
    update = m_t_hat / (v_sqrt + coefficients['epsilon'])

    if self._do_use_weight_decay(var):
      update += coefficients['weight_decay'] * var

    ratio = 1.0
    if self._do_layer_adaptation(var):
      w_n = tf.norm(var,    ord=2)
      g_n = tf.norm(update, ord=2)
      ratio = tf.where(
          tf.greater(w_n, 0),
          tf.where(tf.greater(g_n, 0), (w_n / g_n), 1.0),
          1.0
      )

    var_update = var.assign_sub(
        ratio * coefficients['lr_t'] * update, use_locking=self._use_locking)
    return tf.group(*[var_update, m_t, v_t])

  def get_config(self):
    config = super(LAMB, self).get_config()
    config.update({
      'learning_rate': self._serialize_hyperparameter('learning_rate'),
      'weight_decay': self._serialize_hyperparameter('weight_decay'),
      'decay': self._serialize_hyperparameter('decay'),
      'beta_1': self._serialize_hyperparameter('beta_1'),
      'beta_2': self._serialize_hyperparameter('beta_2'),
      'epsilon': self._epsilon,
      'exclude_weight_decay': self._exclude_weight_decay,
      'exclude_layer_adaptation': self._exclude_layer_adaptation,
    })
    return config

  def _do_use_weight_decay(self, var):
    for r in self._exclude_weight_decay:
      if re.search(r, var.name):
        return False
    return True

  def _do_layer_adaptation(self, var):
    for r in self._exclude_layer_adaptation:
      if re.search(r, var.name):
        return False
    return True
