import tensorflow as tf

def get_stochastic_depth_rate(init_rate: float, index: int, n: int):
  """Get drop connect rate for the ith block.

  Parameters
  ----------
  init_rate : float
    The initial drop rate.
  index : int
    The order of the current block.
  n

  Returns
  -------
  float or None
    Drop rate of the ith block.
  """
  if init_rate is not None:
    if init_rate < 0 or init_rate > 1:
      raise ValueError('Initial drop rate must be within 0 and 1.')
    rate = init_rate * float(index) / n
  else:
    rate = None
  return rate


@tf.keras.utils.register_keras_serializable(package='experiment')
class StochasticDepth(tf.keras.layers.Layer):
  """Creates a stochastic depth layer.

  This layer implements the Deep Networks with Stochastic Depth
  from https://arxiv.org/abs/1603.09382
  """
  def __init__(self, stochastic_depth_drop_rate: float, **kwargs):
    """Initializes a stochastic depth layer.

    Parameters
    ----------
    stochastic_depth_drop_rate : float
      The drop rate.
    kwargs : dict
      Additional keyword arguments to be passed.
    """
    super(StochasticDepth, self).__init__(**kwargs)
    self._drop_rate = stochastic_depth_drop_rate

  def get_config(self):
    config = super(StochasticDepth, self).get_config()
    config.update({
        'drop_rate': self._drop_rate,
    })
    return config

  def call(self, inputs, training=None):
    if training is None:
      training = tf.keras.backend.learning_phase()

    if not training or self._drop_rate is None or self._drop_rate == 0:
      return inputs

    keep_prob = 1.0 - self._drop_rate
    batch_size = tf.shape(inputs)[0]
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(
        [batch_size] + [1] * (inputs.shape.rank - 1), dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * binary_tensor
    return output
