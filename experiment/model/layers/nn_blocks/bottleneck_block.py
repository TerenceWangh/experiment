from typing import Optional

from absl import logging
import tensorflow as tf

from experiment.model.layers import nn_layers
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
    # pylint: disable=bad-whitespace
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
    # pylint: enable=bad-whitespace
    return config

  def call(self, inputs, training=False):
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


@tf.keras.utils.register_keras_serializable(package='experiment')
class InvertedBottleneckBlock(tf.keras.layers.Layer):
  """An inverted bottleneck block."""

  def __init__(self,
               in_filters,
               out_filters,
               expand_ratio,
               strides,
               kernel_size=3,
               se_ratio=None,
               stochastic_depth_drop_rate=None,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               se_inner_activation='relu',
               se_gating_activation='sigmoid',
               se_round_down_protect=True,
               expand_se_in_filters=False,
               depthwise_activation=None,
               use_sync_bn=False,
               dilation_rate=1,
               divisible_by=1,
               regularize_depthwise=False,
               use_residual=True,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               output_intermediate_endpoints=False,
               **kwargs):
    """Initializes an inverted bottleneck block with BN after convolutions.

    Arguments
    =========
    in_filters : int
        The number of filters of the input tensor.
    out_filters : int
        The number of filters of the output tensor.
    expand_ratio : int
        The expand_ratio for an inverted bottlenect block.
    strides : int
        The stride of the block. If greater than 1, this block will ultimately
        downsample the input.
    kernel_size : int, default 3
        The size of kernel for the depthwise convolution layer.
    se_ratio : float, optional
        If not None, se ratio for the squeeze and excitation layer.
    stochastic_depth_drop_rate : float, optional
        If not None, drop rate for the stochastic depth layer.
    kernel_initializer : str
        The kernel initializer for convolution layers.
    kernel_regularizer : tf.keras.regularizers.Regularizer, optional
        The regularizer for the kernel.
    bias_regularizer : tf.keras.regularizers.Regularizer, optional
        The regularizer for the bias.
    activation : str
        The name of the activation function.
    se_inner_activation : str
        The name of squeeze-excitation inner activation.
    se_gating_activation : str
        The name of squeeze-excitation gating activation.
    se_round_down_protect : bool
        Whether round down more than 10% will be allowed in SE layer.
    expand_se_in_filters : bool
        Whether or not to expand in_filter in squeeze and excitation layer.
    depthwise_activation : str
        The name of the activation function for depthwise only.
    use_sync_bn : bool
        If True, use synchronized batch normalization.
    dilation_rate : int
        The specification for the dilation rate to use for.
    divisible_by : int
        The value ensures all inner dimensions are divisible by this number.
    regularize_depthwise : bool
        Whether or not apply regularization on depthwise.
    use_residual : bool
        Whether to include residual connection between input and output.
    norm_momentum : float
        The normalization momentum for the moving average.
    norm_epsilon : float
        The value added to variance to avoid dividing by zero.
    output_intermediate_endpoints : bool
        Whether or not output the intermediate endpoints.
    kwargs : dict
        Additional keyword arguments to be passed.
    """
    super(InvertedBottleneckBlock, self).__init__(**kwargs)

    self._in_filters = in_filters
    self._out_filters = out_filters
    self._expand_ratio = expand_ratio
    self._strides = strides
    self._kernel_size = kernel_size
    self._se_ratio = se_ratio
    self._divisible_by = divisible_by
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._dilation_rate = dilation_rate
    self._use_sync_bn = use_sync_bn
    self._regularize_depthwise = regularize_depthwise
    self._use_residual = use_residual
    self._activation = activation
    self._se_inner_activation = se_inner_activation
    self._se_gating_activation = se_gating_activation
    self._depthwise_activation = depthwise_activation
    self._se_round_down_protect = se_round_down_protect
    self._kernel_initializer = kernel_initializer
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._expand_se_in_filters = expand_se_in_filters
    self._output_intermediate_endpoints = output_intermediate_endpoints

    if use_sync_bn:
      self._norm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf.keras.layers.BatchNormalization
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    if not depthwise_activation:
      self._depthwise_activation = activation
    if regularize_depthwise:
      self._depthsize_regularizer = kernel_regularizer
    else:
      self._depthsize_regularizer = None

  def build(self, input_shape):
    expand_filters = self._in_filters
    if self._expand_ratio > 1:
      # First 1x1 conv for channel expansion.
      expand_filters = tf_utils.make_divisible(
          self._in_filters * self._expand_ratio, self._divisible_by)
      expand_kernel = 1
      expand_stride = 1

      self._conv_0 = tf.keras.layers.Conv2D(
          filters=expand_filters,
          kernel_size=expand_kernel,
          strides=expand_stride,
          padding='same',
          use_bias=False,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)
      self._norm_0 = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon)
      self._activation_layer = tf_utils.get_activation(
          self._activation, use_keras_layer=True)

    self._conv_1 = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(self._kernel_size, self._kernel_size),
        strides=self._strides,
        padding='same',
        depth_multiplier=1,
        dilation_rate=self._dilation_rate,
        use_bias=False,
        depthwise_initializer=self._kernel_initializer,
        depthwise_regularizer=self._depthsize_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm_1 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)
    self._depthwise_activation_layer = tf_utils.get_activation(
        self._depthwise_activation, use_keras_layer=True)

    # Squeeze and excitation
    if self._se_ratio and self._se_ratio > 0 and self._se_ratio <= 1:
      logging.info('Use Squeeze and excitation.')
      in_filters = self._in_filters
      if self._expand_se_in_filters:
        in_filters = expand_filters
      self._squeeze_excitation = nn_layers.SqueezeExcitation(
          in_filters=in_filters,
          out_filters=expand_filters,
          se_ratio=self._se_ratio,
          divisible_by=self._divisible_by,
          round_down_protect=self._se_round_down_protect,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activation=self._se_inner_activation,
          gating_activation=self._se_gating_activation)
    else:
      self._squeeze_excitation = None

    # Last 1x1 conv
    self._conv_2 = tf.keras.layers.Conv2D(
        filters=self._out_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    self._norm_2 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)

    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = nn_layers.StochasticDepth(
          self._stochastic_depth_drop_rate)
    else:
      self._stochastic_depth = None
    self._add = tf.keras.layers.Add()

    super(InvertedBottleneckBlock, self).build(input_shape)

  def get_config(self):
    config = super(InvertedBottleneckBlock, self).get_config()
    config.update({
        'in_filters': self._in_filters,
        'out_filters': self._out_filters,
        'expand_ratio': self._expand_ratio,
        'strides': self._strides,
        'kernel_size': self._kernel_size,
        'se_ratio': self._se_ratio,
        'divisible_by': self._divisible_by,
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'se_inner_activation': self._se_inner_activation,
        'se_gating_activation': self._se_gating_activation,
        'se_round_down_protect': self._se_round_down_protect,
        'expand_se_in_filters': self._expand_se_in_filters,
        'depthwise_activation': self._depthwise_activation,
        'dilation_rate': self._dilation_rate,
        'use_sync_bn': self._use_sync_bn,
        'regularize_depthwise': self._regularize_depthwise,
        'use_residual': self._use_residual,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'output_intermediate_endpoints': self._output_intermediate_endpoints
    })
    return config

  def call(self, inputs, training=None):
    endpoints = {}
    shortcut = inputs
    if self._expand_ratio > 1:
      x = self._conv_0(inputs)
      x = self._norm_0(x)
      x = self._activation_layer(x)
    else:
      x = inputs

    x = self._conv_1(x)
    x = self._norm_1(x)
    x = self._depthwise_activation_layer(x)
    if self._output_intermediate_endpoints:
      endpoints['depthwise'] = x

    if self._squeeze_excitation:
      x = self._squeeze_excitation(x)

    x = self._conv_2(x)
    x = self._norm_2(x)

    if self._use_residual and self._in_filters == self._out_filters and \
        self._strides == 1:
      if self._stochastic_depth:
        x = self._stochastic_depth(x, training=training)
      x = self._add([x, shortcut])

    if self._output_intermediate_endpoints:
      return x, endpoints
    return x
