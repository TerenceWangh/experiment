"""Contains definitions of segmentation heads."""

from typing import List, Union, Optional, Mapping, Tuple, Any
import tensorflow as tf

from experiment.model import tf_utils
from experiment.model.layers import nn_layers
from experiment.ops import spatial_transform_ops


class MaskScoring(tf.keras.Model):
  """Creates a mask scoring layer.

  This implements mask scoring layer from the paper:
  Zhaojin Huang, Lichao Huang, Yongchao Gong, Chang Huang, Xinggang Wang.
  Mask Scoring R-CNN.
  (https://arxiv.org/pdf/1903.00241.pdf)
  """

  def __init__(self,
               num_classes: int,
               fc_input_size: List[int],
               num_convs: int = 3,
               num_filters: int = 256,
               fc_dims: int = 1024,
               num_fcs: int = 2,
               activation: str = 'relu',
               use_sync_bn: bool = False,
               norm_momentum: float = 0.99,
               norm_epsilon: float = 0.001,
               kernel_regularizer: Optional[
                   tf.keras.regularizers.Regularizer] = None,
               bias_regularizer: Optional[
                 tf.keras.regularizers.Regularizer] = None,
               **kwargs):
    """Initializes mask scoring layer.

    Parameters
    ----------
    num_classes : int
        The number of classes.
    fc_input_size : list of int
        The input size of the fully connected layers.
    num_convs : int, default 3
        The number of conv layers.
    num_filters : int, default 256
        The number of filters for conv layers.
    fc_dims : int, default 1024
        The number of filters for each fully connected layers.
    num_fcs : int, default 2
        The number of fully connected layers.
    activation : str, default 'relu'
        The activation function.
    use_sync_bn : bool, False
        Whether or not to use sync batch normalization.
    norm_momentum : float, default 0.99
        The momentum in BatchNorm.
    norm_epsilon : float, default 0.001
        The epsilon value in BatchNorm.
    kernel_regularizer : tf.keras.regularizers.Regularizer, optional
        The regularizer for kernel of conv layers.
    bias_regularizer : tf.keras.regularizers.Regularizer, optional
        The regularizer for bias of conv layers.
    kwargs : dict
        Additional keyword arguments to be passed.
    """
    super(MaskScoring, self).__init__(**kwargs)

    self._num_classes = num_classes
    self._num_convs = num_convs
    self._num_filters = num_filters
    self._fc_input_size = fc_input_size
    self._fc_dims = fc_dims
    self._num_fcs = num_fcs
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._activation = activation
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_layer = tf_utils.get_activation(self._activation)

  def build(self, input_shape: Union[tf.TensorShape, List[tf.TensorShape]]):
    conv_op = tf.keras.layers.Conv2D
    conv_kwargs = {
        'filters': self._num_filters,
        'kernel_size': 3,
        'padding': 'same',
        'kernel_initializer': tf.keras.initializers.VarianceScaling(
            scale=2, mode='fan_out', distribution='untruncated_normal'),
        'bias_initializer': tf.zeros_initializer(),
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
    }

    if self._use_sync_bn:
      bn_op = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      bn_op = tf.keras.layers.BatchNormalization
    bn_kwargs = {
        'axis': self._bn_axis,
        'momentum': self._norm_momentum,
        'epsilon': self._norm_epsilon,
    }

    self._convs = []
    self._conv_norms = []
    for i in range(self._num_convs):
      conv_name = 'mask-scoring_{}'.format(i)
      if 'kernel_initializer' in conv_kwargs:
        conv_kwargs['kernel_initializer'] = tf_utils.clone_initializer(
            conv_kwargs['kernel_initializer'])
      self._convs.append(conv_op(name=conv_name, **conv_kwargs))
      bn_name = 'mask-scoring-bn_{}'.format(i)
      self._conv_norms.append(bn_op(name=bn_name, **bn_kwargs))

    self._fcs = []
    self._fc_norms = []
    for i in range(self._num_fcs):
      fc_name = 'mask-scoring-fc_{}'.format(i)
      self._fcs.append(
          tf.keras.layers.Dense(
              name=fc_name,
              units=self._fc_dims,
              kernel_initializer=tf.keras.initializers.VarianceScaling(
                  scale=1/3.0, mode='fan_out', distribution='uniform'),
              kernel_regularizer=self._kernel_regularizer,
              bias_regularizer=self._bias_regularizer))
      bn_name = 'mask-scoring-fc-bn_{}'.format(i)
      self._fc_norms.append(bn_op(name=bn_name, **bn_kwargs))

    self._classifier = tf.keras.layers.Dense(
        name='iou-scores',
        units=self._num_classes,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)

    super(MaskScoring, self).build(input_shape)

  def call(self, inputs: tf.Tensor, training: Optional[bool] = None):
    """Forward pass mask scoring head.

    Parameters
    ==========
    inputs : tf.Tensor
        The tensor with shape [batch_size, width, size, num_classes],
        representing the segmentation logits.
    training : bool, optional
        Whether it is in `training` mode.

    Returns
    =======
    mask_scores : tf.Tensor
        The predicted mask scores [batch_size, num_classes].
    """
    x = tf.stop_gradient(inputs)
    for conv, bn in zip(self._convs, self._conv_norms):
      x = conv(x)
      x = bn(x)
      x = self._activation_layer(x)

    # Cast feat to float32 so the resize op can be run on TPU
    x = tf.cast(x, tf.float32)
    x = tf.image.resize(x, size=self._fc_input_size,
                        method=tf.image.ResizeMethod.BILINEAR)
    # Cast it back to be compatible with the rest operations
    x = tf.cast(x, inputs.dtype)

    _, h, w, filters = x.get_shape().as_list()
    x = tf.reshape(x, [-1, h * w * filters])

    for fc, bn in zip(self._fcs, self._fc_norms):
      x = fc(x)
      x = bn(x)
      x = self._activation_layer(x)

    ious = self._classifier(x)
    return ious

  def get_config(self) -> Mapping[str, Any]:
    config = super(MaskScoring, self).get_config()
    config.update({
        'num_classes': self._num_classes,
        'num_convs': self._num_convs,
        'num_filters': self._num_filters,
        'fc_input_size': self._fc_input_size,
        'fc_dims': self._fc_dims,
        'num_fcs': self._num_fcs,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'activation': self._activation,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
    })
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)


@tf.keras.utils.register_keras_serializable(package='experiment')
class SegmentationHead(tf.keras.layers.Layer):
  """Creates a segmentation head."""

  def __init__(self,
               num_classes: int,
               level: Union[int, str],
               num_convs: int = 2,
               num_filters: int = 256,
               use_depthwise_convolution: bool = False,
               prediction_kernel_size: int = 1,
               upsample_factor: int = 1,
               feature_fusion: Optional[str] = None,
               decoder_min_level: Optional[int] = None,
               decoder_max_level: Optional[int] = None,
               low_level: int = 2,
               low_level_num_filters: int = 48,
               num_decoder_filters: int = 256,
               activation: str = 'relu',
               use_sync_bn: bool = False,
               norm_momentum: float = 0.99,
               norm_epsilon: float = 0.001,
               kernel_regularizer: Optional[
                 tf.keras.regularizers.Regularizer] = None,
               bias_regularizer: Optional[
                 tf.keras.regularizers.Regularizer] = None,
               **kwargs):
    """Initializes a segmentation head.

    Parameters
    ----------
    num_classes : int
        The number of mask classification categories. Note that the number
        does not include background class.
    level : int or str
        The level to use to build segmentation head.
    num_convs : int, default 2
        The number of stacked convolution before that the last prediction layer.
    num_filters : int, default 256
        The number to specify the number of filters used.
    use_depthwise_convolution : bool, default False
        Whether use depthwise separable convolutions.
    prediction_kernel_size : int, default 1
        The number of kernel size of the prediction layer.
    upsample_factor : int, default 1
        The number of the upsampling factor to generate finer mask.
    feature_fusion : str, optional
        One of the constants in nn_layers.FeatureFusion, namely
        `deeplabv3plus`, `pyramid_fusion`, `panoptic_fpn_fusion`,
        `deeplabv3plus_sum_to_merge`, or None. If `deeplabv3plus`, features from
        decoder_features[level] will be fused with low level feature maps from
        backbone. If `pyramid_fusion`, multiscale features will be resized and
        fused at the target level.
    decoder_min_level : int, optional
        The minimum level from decoder to use in feature fusion. It is only used
        when feature_fusion is set to `panoptic_fpn_fusion`.
    decoder_max_level : int, optional
        The maximum level from decoder to use in feature fusion. It is only used
        when feature_fusion is set to `panoptic_fpn_fusion`.
    low_level : int, default 2
        The backbone level to be used for feature fusion. It is used when
        feature_fusion is set to `deeplabv3plus` or
        `deeplabv3plus_sum_to_merge`.
    low_level_num_filters : int, default 48
        The reduced number of filters for the low level features before fusing
        it with higher level features. It is only used when feature_fusion is
        set to `deeplabv3plus` or `deeplabv3plus_sum_to_merge`.
    num_decoder_filters : int, default 256
        The number of filters in the decoder outputs.
        It is only used when feature_fusion is set to `panoptic_fpn_fusion`.
    activation : str, default 'relu'
        The activation function.
    use_sync_bn : bool, default False
        Whether to use synchronized batch normalization across different
        replicas.
    norm_momentum : float, default 0.99
        The normalization momentum for the moving average.
    norm_epsilon : float, default 0.001
        The normalization epsilon added to variance to avoid dividing by zero.
    kernel_regularizer : tf.keras.regularizers.Regularizer, optional
        The regularizer for kernel of conv layers.
    bias_regularizer : tf.keras.regularizers.Regularizer, optional
        The regularizer for bias of conv layers.
    kwargs : dict
        Additional keyword arguments to be passed.
    """
    super(SegmentationHead, self).__init__(**kwargs)

    self._num_classes = num_classes
    self._level = level
    self._num_convs = num_convs
    self._num_filters = num_filters
    self._use_depthwise_convolution = use_depthwise_convolution
    self._prediction_kernel_size = prediction_kernel_size
    self._upsample_factor = upsample_factor
    self._feature_fusion = feature_fusion
    self._decoder_min_level = decoder_min_level
    self._decoder_max_level = decoder_max_level
    self._low_level = low_level
    self._low_level_num_filters = low_level_num_filters
    self._num_decoder_filters = num_decoder_filters
    self._activation = activation
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_layer = tf_utils.get_activation(activation)

  def build(self, input_shape: Union[tf.TensorShape, List[tf.TensorShape]]):
    use_depthwise_convolution = self._use_depthwise_convolution
    conv_op = tf.keras.layers.Conv2D
    if self._use_sync_bn:
      bn_op = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      bn_op = tf.keras.layers.BatchNormalization
    bn_kwargs = {
        'axis': self._bn_axis,
        'momentum': self._norm_momentum,
        'epsilon': self._norm_epsilon,
    }

    if self._feature_fusion in ['deeplabv3plus', 'deeplabv3plus_sum_to_merge']:
      # Deeplabv3+ feature fusion layers.
      self._dlv3p_conv = conv_op(
          kernel_size=1,
          padding='same',
          use_bias=False,
          kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
          kernel_regularizer=self._kernel_regularizer,
          name='segmentation_head_deeplabv3p_fusion_conv',
          filters=self._low_level_num_filters)
      self._dlv3p_norm = bn_op(
          name='segmentation_head_deeplabv3p_fusion_norm', **bn_kwargs)
    elif self._feature_fusion in ['panoptic_fpn_fusion']:
      self._panoptic_fpn_fusion = nn_layers.PanopticFPNFusion(
          min_level=self._decoder_min_level,
          max_level=self._decoder_max_level,
          target_level=self._level,
          num_filters=self._num_filters,
          num_fpn_filters=self._num_decoder_filters,
          activation=self._activation_layer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)

    # Segmentation head layers.
    self._convs = []
    self._norms = []
    for i in range(self._num_convs):
      if use_depthwise_convolution:
        self._convs.append(
            tf.keras.layers.DepthwiseConv2D(
                name='segmentation_head_depthwise_conv_{}'.format(i),
                kernel_size=3,
                padding='same',
                use_bias=False,
                depthwise_initializer=tf.keras.initializers.RandomNormal(
                    stddev=0.01),
                depthwise_regularizer=self._kernel_regularizer,
                depth_multiplier=1))
        norm_name = 'segmentation_head_depthwise_norm_{}'.format(i)
        self._norms.append(bn_op(name=norm_name, **bn_kwargs))
      conv_name = 'segmentation_head_conv_{}'.format(i)
      self._convs.append(
          conv_op(
              name=conv_name,
              filters=self._num_filters,
              kernel_size=3 if not use_depthwise_convolution else 1,
              padding='same',
              use_bias=False,
              kernel_initializer=tf.keras.initializers.RandomNormal(
                  stddev=0.01),
              kernel_regularizer=self._kernel_regularizer))
      norm_name = 'segmentation_head_norm_{}'.format(i)
      self._norms.append(bn_op(name=norm_name, **bn_kwargs))

    self._classifier = conv_op(
        name='segmentation_output',
        filters=self._num_classes,
        kernel_size=self._prediction_kernel_size,
        padding='same',
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)
    
    super(SegmentationHead, self).build(input_shape)

  def call(self, inputs: Tuple[Union[tf.Tensor, Mapping[str, tf.Tensor]],
                               Union[tf.Tensor, Mapping[str, tf.Tensor]]]):
    """Forward pass of the segmentation head.

    It supports both a tuple of 2 tensors or 2 dictionaries. The first is
    backbone endpoints, and the second is decoder endpoints. When inputs are
    tensors, they are from a single level of feature maps. When inputs are
    dictionaries, they contain multiple levels of feature maps, where the key
    is the index of feature map.

    Parameters
    ----------
    inputs : tuple of tf.Tensor
        A tuple of 2 feature map tensors of shape
        [batch, height_l, width_l, channels] or 2 dictionaries of tensors:
        - key: A `str` of the level of the multilevel features.
        - values: A `tf.Tensor` of the feature map tensors, whose shape is
            [batch, height_l, width_l, channels].
        The first is backbone endpoints, and the second is decoder endpoints.

    Returns
    -------
    tf.Tensor
        The segmentation mask scores predicted from input features.
    """
    backbone_output = inputs[0]
    decoder_output = inputs[1]
    if self._feature_fusion in ['deeplabv3plus', 'deeplabv3plus_sum_to_merge']:
      # deeplabv3+ feature fusion
      x = decoder_output[str(self._level)] if isinstance(
          decoder_output, dict) else decoder_output
      y = backbone_output[str(self._low_level)] if isinstance(
          backbone_output, dict) else backbone_output
      y = self._dlv3p_norm(self._dlv3p_conv(y))
      y = self._activation_layer(y)

      x = tf.image.resize(
          x, tf.shape(y)[1:3], method=tf.image.ResizeMethod.BILINEAR)
      x = tf.cast(x, dtype=y.dtype)
      if self._feature_fusion == 'deeplabv3plus':
        x = tf.concat([x, y], axis=self._bn_axis)
      else:
        x = tf.keras.layers.Add()([x, y])
    elif self._feature_fusion in ['pyramid_fusion']:
      if not isinstance(decoder_output, dict):
        raise ValueError('Only support dictionary decoder_output.')
      x = nn_layers.pyramid_feature_fusion(decoder_output, self._level)
    elif self._feature_fusion in ['panoptic_fpn_fusion']:
      x = self._panoptic_fpn_fusion(decoder_output)
    else:
      x = decoder_output[str(self._level)] if isinstance(
          decoder_output, dict) else decoder_output

    for conv, norm in zip(self._convs, self._norms):
      x = conv(x)
      x = norm(x)
      x = self._activation_layer(x)
    if self._upsample_factor > 1:
      x = spatial_transform_ops.nearest_upsampling(
          x, scale=self._upsample_factor)

    return self._classifier(x)

  def get_config(self):
    config = super(SegmentationHead, self).get_config()
    config.update({
        'num_classes': self._num_classes,
        'level': self._level,
        'num_convs': self._num_convs,
        'num_filters': self._num_filters,
        'use_depthwise_convolution': self._use_depthwise_convolution,
        'prediction_kernel_size': self._prediction_kernel_size,
        'upsample_factor': self._upsample_factor,
        'feature_fusion': self._feature_fusion,
        'decoder_min_level': self._decoder_min_level,
        'decoder_max_level': self._decoder_max_level,
        'low_level': self._low_level,
        'low_level_num_filters': self._low_level_num_filters,
        'num_decoder_filters': self._num_decoder_filters,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
    })
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)
