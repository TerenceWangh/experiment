"""Functions and classes related to training performance."""

import tensorflow as tf


def configure_optimizer(optimizer,
                        use_float16=False,
                        loss_scale=None):
  """Configures optimizer object with performance options."""
  if use_float16:
    if loss_scale in (None, 'dynamic'):
      optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    else:
      # loss_scale is a number. We interpret that as a fixed loss scale.
      optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
          optimizer, dynamic=False, initial_scale=loss_scale)
  return optimizer


def set_mixed_precision_policy(dtype, loss_scale=None):
  """Sets the global `tf.keras.mixed_precision.Policy`."""
  assert loss_scale is None, (
      'The loss_scale argument must be None. The argument exists for '
      'historical reasons and will be removed soon.')
  if dtype == tf.float16:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
  elif dtype == tf.bfloat16:
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
  elif dtype == tf.float32:
    tf.keras.mixed_precision.set_global_policy('float32')
  else:
    raise ValueError('Unexpected dtype: %s' % dtype)
