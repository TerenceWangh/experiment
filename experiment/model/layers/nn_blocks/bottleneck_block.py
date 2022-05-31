from typing import Optional
import tensorflow as tf

from experiment.model import tf_utils

@tf.keras.utils.register_keras_serializable(package='experiment')
class BottleneckBlock(tf.keras.layers.Layer):
  """A standard bottleneck block."""

  def __init__(self,
               filters: int,
               strides: int,
               dilation_rate: int = 1,
               use_projection: bool = False,
               se_ratio: Optional[float] = None,
               resnetd_shortcut: bool = False,
               stochastic_depth_drop_rate: Optional[float] = None,
               kernel_initializer: str = 'VarianceScaling',
               kernel_regularizer: Optional[
                 tf.keras.regularizers.Regularizer] = None,
               bias_regularizer: Optional[
                 tf.keras.regularizers.Regularizer] = None,
               activation: str = 'relu',
               use_sync_bn: bool = False,
               norm_momentum: float = 0.99,
               norm_epsilon: float = 0.001,
               bn_trainable: bool = True,
               **kwargs):
    """Initializes a standard bottleneck block with BN after convolutions.

    Parameters
    ----------
    filters : int
      The number of filters for the first two convolutions. Note that the
      third and final convolution will use 4 times as many filters.
    strides : int
      The block stride. If greater than 1, this block will ultimately
      downsample the input.
    dilation_rate : int, default 1
      The  dilation_rate of convolutions
    use_projection : bool, default False
      Whether this block should use a projection shortcut (versus the default
      identity shortcut). This is usually `True` for the first block of a
      block group, which may change the number of filters and the resolution.
    se_ratio : float, optional
      The Ratio of the Squeeze-and-Excitation layer.
    resnetd_shortcut : bool, default False
      Whether applying the resnetd style modification to the shortcut
      connection.
    stochastic_depth_drop_rate : float, optional
      If not None, drop rate for the stochastic depth layer.
    kernel_initializer : str, default 'VarianceScaling'
      The kernel_initializer for convolutional layers.
    kernel_regularizer : tf.keras.regularizers.Regularizer, optional
      The kernel regularizer.
    bias_regularizer : tf.keras.regularizers.Regularizer, optional
      The bias regularizer.
    activation : str, default 'relu'
      The name of the activation function.
    use_sync_bn : bool, default False
      Whether using synchronized batch normalization.
    norm_momentum : float, default 0.99
      The normalization momentum for the moving average.
    norm_epsilon : float, default 0.001
      The added to variance to avoid dividing by zero.
    bn_trainable : bool, default True
      Whether batch norm layers should be trainable. Default to True.
    kwargs : dict
      The additional parameters.
    """
    super(BottleneckBlock, self).__init__(**kwargs)

    self._filters = filters
    self._strides = strides
    self._dilation_rate = dilation_rate
    self._use_projection = use_projection
    self._se_ratio = se_ratio
    self._resnetd_shortcut = resnetd_shortcut
    self._use_sync_bn = use_sync_bn
    self._activation = activation
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._kernel_initializer = kernel_initializer
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    if use_sync_bn:
      self._norm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf.keras.layers.BatchNormalization
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._bn_trainable = bn_trainable

  def build(self, input_shape):
    if self._use_projection:
      if self._resnetd_shortcut:
        self._shortcut0 = tf.keras.layers.AveragePooling2D(
            pool_size=2, strides=self._strides, padding='same')
        self._shortcut1 = tf.keras.layers.Conv2D(
            filters=self._filters * 4,
            kernel_size=1,
            strides=1,
            use_bias=False,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer)
      else:
        self._shortcut = tf.keras.layers.Conv2D(
            filters=self._filters * 4,
            kernel_size=1,
            strides=self._strides,
            use_bias=False,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer)

      self._norm0 = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon,
          trainable=self._bn_trainable)

    self._conv1 = tf.keras.layers.Conv2D(
        filters=self._filters,
        kernel_size=1,
        strides=1,
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm1 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        trainable=self._bn_trainable)
    self._activation1 = tf_utils.get_activation(
        self._activation, use_keras_layer=True)

    self._conv2 = tf.keras.layers.Conv2D(
        filters=self._filters,
        kernel_size=3,
        strides=self._strides,
        dilation_rate=self._dilation_rate,
        padding='same',
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm2 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        trainable=self._bn_trainable)
    self._activation2 = tf_utils.get_activation(
        self._activation, use_keras_layer=True)

    self._conv3 = tf.keras.layers.Conv2D(
        filters=self._filters * 4,
        kernel_size=1,
        strides=1,
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm3 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        trainable=self._bn_trainable)
    self._activation3 = tf_utils.get_activation(
        self._activation, use_keras_layer=True)

    if self._se_ratio and self._se_ratio > 0 and self._se_ratio <= 1:
      self._squeeze_excitation = nn_layers.SqueezeExcitation(
          in_filters=self._filters * 4,
          out_filters=self._filters * 4,
          se_ratio=self._se_ratio,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)
    else:
      self._squeeze_excitation = None

    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = nn_layers.StochasticDepth(
          self._stochastic_depth_drop_rate)
    else:
      self._stochastic_depth = None
    self._add = tf.keras.layers.Add()

    super(BottleneckBlock, self).build(input_shape)

  def get_config(self):
    config = super(BottleneckBlock, self).get_config()
    config.update({
      'filters'                   : self._filters,
      'strides'                   : self._strides,
      'dilation_rate'             : self._dilation_rate,
      'use_projection'            : self._use_projection,
      'se_ratio'                  : self._se_ratio,
      'resnetd_shortcut'          : self._resnetd_shortcut,
      'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
      'kernel_initializer'        : self._kernel_initializer,
      'kernel_regularizer'        : self._kernel_regularizer,
      'bias_regularizer'          : self._bias_regularizer,
      'activation'                : self._activation,
      'use_sync_bn'               : self._use_sync_bn,
      'norm_momentum'             : self._norm_momentum,
      'norm_epsilon'              : self._norm_epsilon,
      'bn_trainable'              : self._bn_trainable,
    })
    return config

  def call(self, inputs, training = False):
    shortcut = inputs
    if self._use_projection:
      if self._resnetd_shortcut:
        shortcut = self._shortcut0(shortcut)
        shortcut = self._shortcut1(shortcut)
      else:
        shortcut = self._shortcut(shortcut)
      shortcut = self._norm0(shortcut)

    x = self._conv1(inputs)
    x = self._norm1(x)
    x = self._activation1(x)

    x = self._conv2(x)
    x = self._norm2(x)
    x = self._activation2(x)

    x = self._conv3(x)
    x = self._norm3(x)

    if self._squeeze_excitation:
      x = self._squeeze_excitation(x)

    if self._stochastic_depth:
      x = self._stochastic_depth(x, training=training)

    x = self._add([x, shortcut])
    return self._activation3(x)
