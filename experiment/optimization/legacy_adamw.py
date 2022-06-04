"""Adam optimizer with weight decay that exactly matches the original BERT."""

from typing import Optional, List
import re
import tensorflow as tf


class AdamWeightDecay(tf.keras.optimizers.Adam):
  """Adam enables L2 weight decay and clip_by_global_norm on gradients.

  [Warning!]: Keras optimizer supports gradient clipping and has an AdamW
  implementation. Please consider evaluating the choice in Keras package.

  Just adding the square of the weights to the loss function is *not* the
  correct way of using L2 regularization/weight decay with Adam, since that will
  interact with the m and v parameters in strange ways.

  Instead we want to decay the weights in a manner that doesn't interact with
  the m/v parameters. This is equivalent to adding the square of the weights to
  the loss with plain (non-momentum) SGD
  """

  def __init__(self,
               learning_rate: float = 0.001,
               beta_1: float = 0.9,
               beta_2: float = 0.999,
               epsilon: float = 1e-7,
               amsgrad: bool = False,
               weight_decay: float = 0.0,
               include_weight_decay: Optional[List[str]] = None,
               exclude_weight_decay: Optional[List[str]] = None,
               gradient_clip_norm: float = 1.0,
               name: str = 'AdamWeightDecay',
               **kwargs):
    super(AdamWeightDecay, self).__init__(learning_rate, beta_1, beta_2,
                                          epsilon, amsgrad, name, **kwargs)
    self._weight_decay = weight_decay
    self._gradient_clip_norm = gradient_clip_norm
    self._include_weight_decay = include_weight_decay
    self._exclude_weight_decay = exclude_weight_decay

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype,
                                                apply_state)
    apply_state[(var_device, var_dtype)]['weight_decay'] = tf.constant(
        self._weight_decay, name='adam_weight_decay')

  def _decay_weights_op(self, var, learning_rate, apply_state):
    do_decay = self._do_use_weight_decay(var.name)
    if do_decay:
      return var.assign_sub(
          learning_rate * var *
          apply_state[(var.device, var.dtype.base_dtype)]['weight_decay'],
          use_locking=self._use_locking)
    return tf.no_op()

  def apply_gradients(self,
                      grads_and_vars,
                      name=None,
                      experimental_aggregate_gradients=True):
    grads, tvars = list(zip(*grads_and_vars))
    if experimental_aggregate_gradients and self.gradient_clip_norm > 0.0:
      # when experimental_aggregate_gradients = False, apply_gradients() no
      # longer implicitly all reduce gradients, users manually all reduce
      # gradient and passed the all reduced grads_and_vars. For now, the
      # clip_by_global_norm will be moved to before the explicit all reduce to
      # keep the math the same as TF 1 and pre TF 2.2 implementation.
      (grads, _) = tf.clip_by_global_norm(
          grads, clip_norm=self._gradient_clip_norm)
    return super(AdamWeightDecay, self).apply_gradients(
        zip(grads, tvars), name=name,
        experimental_aggregate_gradients=experimental_aggregate_gradients)

  def _get_learning_rate(self, var_device, var_dtype, apply_state):
    """Retrieves the learning rate with the given state."""
    if apply_state is None:
      return self._decayed_learning_rate_t[var_dtype], {}

    apply_state = apply_state or {}
    coefficients = apply_state.get((var_device, var_dtype))
    if coefficients is None:
      coefficients = self._fallback_apply_state(var_device, var_dtype)
      apply_state[(var_device, var_dtype)] = coefficients

    return coefficients['learning_rate_t'], dict(apply_state=apply_state)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    learning_rate_t, kwargs = self._get_learning_rate(
        var.device, var.dtype.base_dtype, apply_state)
    decay = self._decay_weights_op(var, learning_rate_t, apply_state)
    with tf.control_dependencies([decay]):
      return super(AdamWeightDecay, self)._resource_apply_dense(
          grad, var, **kwargs)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    learning_rate_t, kwargs = self._get_learning_rate(
        var.device, var.dtype.base_dtype, apply_state)
    decay = self._decay_weights_op(var, learning_rate_t, apply_state)
    with tf.control_dependencies([decay]):
      return super(AdamWeightDecay, self)._resource_apply_sparse(
          grad, var, indices, **kwargs)

  def get_config(self):
    config = super(AdamWeightDecay, self).get_config()
    config.update({
        'weight_decay': self._weight_decay,
    })
    return config

  def _do_use_weight_decay(self, param_name):
    if self._weight_decay == 0:
      return False

    if self._include_weight_decay:
      for r in self._include_weight_decay:
        if re.search(r, param_name) is not None:
          return True

    if self._exclude_weight_decay:
      for r in self._exclude_weight_decay:
        if re.search(r, param_name) is not None:
          return False

    return True
