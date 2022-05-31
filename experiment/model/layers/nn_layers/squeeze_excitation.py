from typing import Optional
import tensorflow as tf

from experiment.model import tf_utils

@tf.keras.utils.register_keras_serializable(package='experiment')
class SqueezeExcitation(tf.keras.layers.Layer):
  """Creates a squeeze and excitation layer.

  This block implements the Squeeze-and-Excitation block from
  https://arxiv.org/abs/1709.01507
  """

  def __init__(self,
               in_filters: int,
               out_filters: int,
               se_ratio: float,
               divisible_by: int = 1,
               use_3d_input: bool = False,
               kernel_initializer: str = 'VarianceScaling',
               kernel_regularizer: Optional[
                 tf.keras.regularizers.Regularizer] = None,
               bias_regularizer: Optional[
                 tf.keras.regularizers.Regularizer] = None,
               activation: str = 'relu',
               gating_activation: str = 'sigmoid',
               round_down_protect:bool = True,
               **kwargs):
    """Initializes a squeeze and excitation layer.

    Parameters
    ----------
    in_filters : int
      The number of input filters.
    out_filters : int
      The number of output filters.
    se_ratio : float
      se ratio for the squeeze and excitation layer.
    divisible_by : int, default 1
      Ensures all inner dimensions are divisible by this number.
    use_3d_input : bool, default False
      Whether the input is 3D image.
    kernel_initializer : str, default 'VarianceScaling'
      The kernel_initializer for convolutional layers.
    kernel_regularizer : tf.keras.regularizers.Regularizer, optional
      The kernel regularizer.
    bias_regularizer : tf.keras.regularizers.Regularizer, optional
      The bias regularizer.
    activation : str, default 'relu'
      The name of the activation function.
    gating_activation : str, default 'sigmoid'
      The name of the final gating function.
    round_down_protect : bool, default True
      Whether round down more than 10% will be allowed.
    """
    super(SqueezeExcitation, self).__init__(**kwargs)

    self._in_filters = in_filters
    self._out_filters = out_filters
    self._se_ratio = se_ratio
    self._divisible_by = divisible_by
    self._round_down_protect = round_down_protect
    self._use_3d_input = use_3d_input
    self._activation = activation
    self._gating_activation = gating_activation
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    if tf.keras.backend.image_data_format() == 'channels_last':
      if not use_3d_input:
        self._spatial_axis = [1, 2]
      else:
        self._spatial_axis = [1, 2, 3]
    else:
      if not use_3d_input:
        self._spatial_axis = [2, 3]
      else:
        self._spatial_axis = [2, 3, 4]
    self._activation_fn = tf_utils.get_activation(activation)
    self._gating_activation_fn = tf_utils.get_activation(gating_activation)

  def build(self, input_shape):
    num_reduced_filters = tf_utils.make_divisible(
        max(1, int(self._in_filters * self._se_ratio)),
        divisor=self._divisible_by,
        round_down_protect=self._round_down_protect)

    self._se_reduce = tf.keras.layers.Conv2D(
        filters=num_reduced_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=True,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
    )

    self._se_expand = tf.keras.layers.Conv2D(
        filters=self._out_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=True,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
    )

    super(SqueezeExcitation, self).build(input_shape)

  def get_config(self):
    config = super(SqueezeExcitation, self).get_config()
    config.update({
      'in_filters'        : self._in_filters,
      'out_filters'       : self._out_filters,
      'se_ratio'          : self._se_ratio,
      'divisible_by'      : self._divisible_by,
      'use_3d_input'      : self._use_3d_input,
      'kernel_initializer': self._kernel_initializer,
      'kernel_regularizer': self._kernel_regularizer,
      'bias_regularizer'  : self._bias_regularizer,
      'activation'        : self._activation,
      'gating_activation' : self._gating_activation,
      'round_down_protect': self._round_down_protect,
    })
    return config

  def call(self, inputs):
    x = tf.reduce_mean(inputs, self._spatial_axis, keepdims=True)
    x = self._activation_fn(self._se_reduce(x))
    x = self._gating_activation_fn(self._se_expand(x))
    return x * inputs
