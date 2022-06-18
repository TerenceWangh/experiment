from typing import Optional
import tensorflow as tf

from experiment.model import tf_utils
from experiment.model.layers import nn_layers
from experiment.ops import spatial_transform_ops


@tf.keras.utils.register_keras_serializable(package='experiment')
def pyramid_feature_fusion(inputs, target_level):
  """Fuses all feature maps in the feature pyramid at the target level.

  Parameters
  ----------
  inputs : dict
      A dictionary containing the feature pyramid. The size of the input tensor
      needs to be fixed.
  target_level : int
      The target feature level for feature fusion.

  Returns
  -------
  tf.Tensor
      The float tensor with shape [batch_size, feature_height, feature_width,
      feature channel].
  """
  # Convert keys to int.
  pyramid_feats = {int(k): v for k, v in inputs.items()}
  min_level, max_level = min(pyramid_feats.keys()), max(pyramid_feats.keys())

  resampled_feats = []
  for level in range(min_level, max_level + 1):
    feat = pyramid_feats[level]
    if level != target_level:
      target_size = list(feat.shape[1:3])
      target_size[0] *= 2**(level - target_level)
      target_size[1] *= 2**(level - target_level)
      target_size = tf.cast(target_size, tf.int32)
      # cast feat to float32 so the resize op can be run on TPU
      feat = tf.cast(feat, tf.float32)
      feat = tf.image.resize(
          feat, size=target_size, method=tf.image.ResizeMethod.BILINEAR)
      # cast it back to be compatible with the rest operations
      feat = tf.cast(feat, pyramid_feats[level].dtype)
    resampled_feats.append(feat)

  return tf.math.add_n(resampled_feats)


class PanopticFPNFusion(tf.keras.Model):
  """Creates a Panoptic FPN feature Fusion layer.

  This implements feature fusion for semantic segmentation head from the paper:
  Alexander Kirillov, Ross Girshick, Kaiming He and Piotr Dollar.
  Panoptic Feature Pyramid Networks.
  (https://arxiv.org/pdf/1901.02446.pdf)
  """

  def __init__(self,
               min_level: int = 2,
               max_level: int = 5,
               target_level: int = 2,
               num_filters: int = 128,
               num_fpn_filters: int = 256,
               activation: str = 'relu',
               kernel_regularizer: Optional[
                   tf.keras.regularizers.Regularizer] = None,
               bias_regularizer: Optional[
                   tf.keras.regularizers.Regularizer] = None,
               **kwargs):
    """Initializes panoptic FPN feature fusion layer.

    Parameters
    ----------
    min_level : int, default 2
        The minimum level to use in feature fusion.
    max_level : int, default 5
        The maximum level to use in feature fusion.
    target_level : int, default 2
        The target feature level for feature fusion.
    num_filters : int, default 128
        The number of filters in conv2d layers.
    num_fpn_filters : int, default 256
        The number of filters in the FPN outputs
    activation : str, default 'relu'
        The activation function.
    kernel_regularizer : tf.keras.regularizers.Regularizer, optional
        The regularizer for kernel of conv layers.
    bias_regularizer : tf.keras.regularizers.Regularizer, optional
        The regularizer for bias of conv layers.
    kwargs : dict
        Additional keyword arguments to be passed.
    """
    if target_level > max_level:
      raise ValueError('target_level should be less than max_level.')

    self._min_level = min_level
    self._max_level = max_level
    self._target_level = target_level
    self._num_filters = num_filters
    self._num_fpn_filters = num_fpn_filters
    self._activation = activation
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    self._activation_op = tf_utils.get_activation(self._activation)
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._norm_axis = -1
    else:
      self._norm_axis = 1

    inputs = self._build_inputs()
    fused_features = self._call(inputs)
    self._output_specs = {str(target_level): fused_features.get_shape()}
    super(PanopticFPNFusion, self).__init__(
        inputs=inputs, outputs=fused_features,
        **kwargs)

  def get_config(self):
    config = super(PanopticFPNFusion, self).get_config()
    config.update({
        'min_level': self._min_level,
        'max_level': self._max_level,
        'target_level': self._target_level,
        'num_filters': self._num_filters,
        'num_fpn_filters': self._num_fpn_filters,
        'activation': self._activation,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
    })
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    return self._output_specs

  def _build_inputs(self):
    inputs = {}
    for level in range(self._min_level, self._max_level + 1):
      inputs[str(level)] = tf.keras.Input(
          shape=[None, None, self._num_fpn_filters])
    return inputs

  def _call(self, inputs):
    upscaled_features = []
    for level in range(self._min_level, self._max_level + 1):
      num_conv_layers = max(1, level - self._target_level)
      x = inputs[str(level)]
      for i in range(num_conv_layers):
        x = tf.keras.layers.Conv2D(
            filters=self._num_filters,
            kernel_size=3,
            padding='same',
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer)(x)
        x = nn_layers.GroupNormalization(groups=32, axis=self._norm_axis)(x)
        x = self._activation_op(x)
        if level != self._target_level:
          x = spatial_transform_ops.nearest_upsampling(x, scale=2)
      upscaled_features.append(x)
    fused_features = tf.math.add_n(upscaled_features)
    return fused_features
