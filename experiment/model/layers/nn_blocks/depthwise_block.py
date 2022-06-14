from typing import Text, Optional

import tensorflow as tf

from experiment.model import tf_utils

@tf.keras.utils.register_keras_serializable(package='experiment')
class DepthwiseSeparableConvBlock(tf.keras.layers.Layer):
  """Create an depthwise separable convolution block with batch normalization.
  """

  def __init__(self,
               filters: int,
               kernel_size: int = 3,
               strides: int = 1,
               regularize_depthwise: bool = False,
               activation: Text = 'relu6',
               kernel_initializer: Text = 'VarianceScaling',
               kernel_regularizer: Optional[
                   tf.keras.regularizers.Regularizer] = None,
               dilation_rate: int = 1,
               use_sync_bn: bool = False,
               norm_momentum: float = 0.99,
               norm_epsilon: float = 0.001,
               **kwargs):
    """Initializes a convolution block with batch normalization.

    Arguments
    =========
    filters : int
        The number of filters for the first two convolutions. Note that the
        third and final convolution will use 4 times as many filters.
    kernel_size : int
        The specification for the height and width of the 2D convolution window.
    strides : int
        The block stride. If greater than 1, this block will ultimately
        downsample the input.
    regularize_depthwise : bool
        Whether apply regularization on depthwise.
    activation : str
        The name of the activation function.
    kernel_initializer : str
        The initializer for kernel.
    kernel_regularizer : str
        The regularizer for the kernel.
    dilation_rate : int or tuple or list of int
        Specifying the dilation rate to use for dilated convolution. Can be a
        single integer to specify the same value for all spatial dimensions.
    use_sync_bn : bool
        Whether use synchronized batch normalization.
    norm_momentum : float
        The normalization momentum for the moving average.
    norm_epsilon : float
        The value added to variance to avoid dividing by zero.
    kwargs : dict
        Additional keyword arguments to be passed.
    """
    super(DepthwiseSeparableConvBlock, self).__init__(**kwargs)
    self._filters = filters
    self._kernel_size = kernel_size
    self._strides = strides
    self._activation = activation
    self._regularize_depthwise = regularize_depthwise
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._dilation_rate = dilation_rate
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon

    if use_sync_bn:
      self._norm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf.keras.layers.BatchNormalization
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_fn = tf_utils.get_activation(activation)
    if regularize_depthwise:
      self._depthsize_regularizer = kernel_regularizer
    else:
      self._depthsize_regularizer = None

  def get_config(self):
    config = super(DepthwiseSeparableConvBlock, self).get_config()
    config.update({
        'filters': self._filters,
        'strides': self._strides,
        'regularize_depthwise': self._regularize_depthwise,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
    })
    return config

  def build(self, input_shape):
    self._dw_conv_0 = tf.keras.layers.DepthwiseConv2D(
        kernel_size=self._kernel_size,
        strides=self._strides,
        padding='same',
        depth_multiplier=1,
        dilation_rate=self._dilation_rate,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._depthsize_regularizer,
        use_bias=False)
    self._norm_0 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)
    self._conv_1 = tf.keras.layers.Conv2D(
        filters=self._filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer)
    self._norm_1 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)

    super(DepthwiseSeparableConvBlock, self).build(input_shape)

  def call(self, inputs, training=None):
    x = self._dw_conv_0(inputs)
    x = self._norm_0(x)
    x = self._activation_fn(x)

    x = self._conv_1(x)
    x = self._norm_1(x)
    return self._activation_fn(x)
