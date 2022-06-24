"""Contains definitions of instance prediction heads."""

from typing import List, Union, Optional
import tensorflow as tf

from experiment.model import tf_utils


@tf.keras.utils.register_keras_serializable(package='experiment')
class DetectionHead(tf.keras.layers.Layer):
  """Creates a detection head."""

  def __init__(self,
               num_classes: int,
               num_convs: int = 0,
               num_filters: int = 256,
               use_separable_conv: bool = False,
               num_fcs: int = 2,
               fc_dims: int = 1024,
               class_agnostic_bbox_pred: bool = False,
               activation: str = 'relu',
               use_sync_bn: bool = False,
               norm_momentum: float = 0.99,
               norm_epsilon: float = 0.001,
               kernel_regularizer: Optional[
                   tf.keras.regularizers.Regularizer] = None,
               bias_regularizer: Optional[
                   tf.keras.regularizers.Regularizer] = None,
               **kwargs):
    """Initializes a detection head.

    Parameters
    ----------
    num_classes : int
        The number of classes for detection.
    num_convs : int, default 0
        The number of the intermediate convolution layers before the FC layers.
    num_filters : int, default 256
        The number of filters of the intermediate convolution layers.
    use_separable_conv : bool, default False
        Whether the separable convolution layers is used.
    num_fcs : int, default 2
        The number of FC layers before predictions.
    fc_dims : int, default 1024
        The number of dimension of the FC layers.
    class_agnostic_bbox_pred : bool, default False
        Whether bboxes should be predicted for every class or not.
    activation : str, default 'relu'
        The activation function.
    use_sync_bn : bool, default False
        Whether to use synchronized batch normalization across different
        replicas.
    norm_momentum : float, default 0.99
        The normalization momentum for the moving average.
    norm_epsilon : float, default 0.001
        The value added to variance to avoid dividing by zero.
    kernel_regularizer : tf.keras.regularizers.Regularizer, optional
        The regularizer for kernel of conv layers.
    bias_regularizer : tf.keras.regularizers.Regularizer, optional
        The regularizer for bias of conv layers.
    kwargs : dict
        Additional keyword arguments to be passed.
    """
    super(DetectionHead, self).__init__(**kwargs)

    self._num_classes = num_classes
    self._num_convs = num_convs
    self._num_filters = num_filters
    self._use_separable_conv = use_separable_conv
    self._num_fcs = num_fcs
    self._fc_dims = fc_dims
    self._class_agnostic_bbox_pred = class_agnostic_bbox_pred
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
    conv_kwargs = {
        'filters': self._num_filters,
        'kernel_size': 3,
        'padding': 'same',
    }
    if self._use_separable_conv:
      conv_op = tf.keras.layers.SeparableConv2D
      conv_kwargs.update({
          'depthwise_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'pointwise_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'bias_initializer': tf.zeros_initializer(),
          'depthwise_regularizer': self._kernel_regularizer,
          'pointwise_regularizer': self._kernel_regularizer,
          'bias_regularizer': self._bias_regularizer,
      })
    else:
      conv_op = tf.keras.layers.Conv2D
      conv_kwargs.update({
          'kernel_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'bias_initializer': tf.zeros_initializer(),
          'kernel_regularizer': self._kernel_regularizer,
          'bias_regularizer': self._bias_regularizer,
      })

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
      conv_name = 'detection-conv_{}'.format(i)
      for initializer_name in ['kernel_initializer',
                               'depthwise_initializer',
                               'pointwise_initializer']:
        if initializer_name in conv_kwargs:
          conv_kwargs[initializer_name] = tf_utils.clone_initializer(
              conv_kwargs[initializer_name])
      self._convs.append(conv_op(name=conv_name, **conv_kwargs))
      bn_name = 'detection-conv-bn_{}'.format(i)
      self._conv_norms.append(bn_op(name=bn_name, **bn_kwargs))

    self._fcs = []
    self._fc_norms = []
    for i in range(self._num_fcs):
      fc_name = 'detection-fc_{}'.format(i)
      self._fcs.append(
          tf.keras.layers.Dense(
              units=self._fc_dims,
              kernel_initializer=tf.keras.initializers.VarianceScaling(
                  scale=1.0/3.0, mode='fan_out', distribution='uniform'),
              kernel_regularizer=self._kernel_regularizer,
              bias_regularizer=self._bias_regularizer,
              name=fc_name))
      bn_name = 'detection-fc-bn_{}'.format(i)
      self._fc_norms.append(bn_op(name=bn_name, **bn_kwargs))

    self._classifier = tf.keras.layers.Dense(
        units=self._num_classes,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        name='detection-scores')

    if self._class_agnostic_bbox_pred:
      num_box_outputs = 4
    else:
      num_box_outputs = self._num_classes * 4

    self._box_regressor = tf.keras.layers.Dense(
        units=num_box_outputs,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        name='detection-bboxes')

    super(DetectionHead, self).build(input_shape)

  def call(self, inputs: tf.Tensor, training: Optional[bool] = None):
    """Forward pass of box and class branches for the Mask-RCNN model.

    Parameters
    ----------
    inputs : tf.Tensor
        The tensor of the shape [batch_size, num_instance, roi_height,
        roi_width, roi_channels], representing the ROI features.
    training : bool, optional
        Whether it is in `training` mode.

    Returns
    -------
    tf.Tensor
        class_outputs: [batch_size, num_rois, num_classes], representing the
        class predictions
        box_outputs: [batch_size, num_rois, num_classes * 4], representing the
        box predictions.
    """
    roi_features = inputs
    _, num_rois, height, width, filters = roi_features.get_shape().as_list()

    x = tf.reshape(roi_features, [-1, height, width, filters])
    for conv, bn in zip(self._convs, self._conv_norms):
      x = conv(x)
      x = bn(x)
      x = self._activation_layer(x)

    _, _, _, filters = x.get_shape().as_list()
    x = tf.reshape(x, [-1, num_rois, height * width * filters])

    for fc, bn in zip(self._fcs, self._fc_norms):
      x = fc(x)
      x = bn(x)
      x = self._activation_layer(x)

    classes = self._classifier(x)
    boxes = self._box_regressor(x)
    return classes, boxes

  def get_config(self):
    config = super(DetectionHead, self).get_config()
    config.update({
        'num_classes': self._num_classes,
        'num_convs': self._num_convs,
        'num_filters': self._num_filters,
        'use_separable_conv': self._use_separable_conv,
        'num_fcs': self._num_fcs,
        'fc_dims': self._fc_dims,
        'class_agnostic_bbox_pred': self._class_agnostic_bbox_pred,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
    })
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)


@tf.keras.utils.register_keras_serializable(package='experiment')
class MaskHead(tf.keras.layers.Layer):
  """Create a mask head"""

  def __init__(self,
               num_classes: int,
               upsample_factor: int = 2,
               num_convs: int = 4,
               num_filters: int = 256,
               use_separable_conv: bool = False,
               class_agnostic: bool = False,
               activation: str = 'relu',
               use_sync_bn: bool = False,
               norm_momentum: float = 0.99,
               norm_epsilon: float = 0.001,
               kernel_regularizer: Optional[
                   tf.keras.regularizers.Regularizer] = None,
               bias_regularizer: Optional[
                   tf.keras.regularizers.Regularizer] = None,
               **kwargs):
    """Initializes a detection head.

    Parameters
    ----------
    num_classes : int
        The number of classes for detection.
    upsample_factor : int default 2
        The upsample factor to generate the final predicted masks.
        It should be >= 1.
    num_convs : int, default 4
        The number of the intermediate convolution layers before the FC layers.
    num_filters : int, default 256
        The number of filters of the intermediate convolution layers.
    use_separable_conv : bool, default False
        Whether the separable convolution layers is used.
    class_agnostic : bool, default False
        Whether use a single channel mask head that is shared between all
        classes.
    activation : str, default 'relu'
        The activation function.
    use_sync_bn : bool, default False
        Whether to use synchronized batch normalization across different
        replicas.
    norm_momentum : float, default 0.99
        The normalization momentum for the moving average.
    norm_epsilon : float, default 0.001
        The value added to variance to avoid dividing by zero.
    kernel_regularizer : tf.keras.regularizers.Regularizer, optional
        The regularizer for kernel of conv layers.
    bias_regularizer : tf.keras.regularizers.Regularizer, optional
        The regularizer for bias of conv layers.
    kwargs : dict
        Additional keyword arguments to be passed.
    """
    super(MaskHead, self).__init__(**kwargs)

    self._num_classes = num_classes
    self._upsample_factor = upsample_factor
    self._num_convs = num_convs
    self._num_filters = num_filters
    self._use_separable_conv = use_separable_conv
    self._activation = activation
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._class_agnostic = class_agnostic

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_layer = tf_utils.get_activation(activation)

  def build(self, input_shape: Union[tf.TensorShape, List[tf.TensorShape]]):
    """Creates the variables of the head."""
    conv_kwargs = {
        'filters': self._num_filters,
        'kernel_size': 3,
        'padding': 'same',
    }
    if self._use_separable_conv:
      conv_op = tf.keras.layers.SeparableConv2D
      conv_kwargs.update({
          'depthwise_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'pointwise_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'bias_initializer': tf.zeros_initializer(),
          'depthwise_regularizer': self._kernel_regularizer,
          'pointwise_regularizer': self._kernel_regularizer,
          'bias_regularizer': self._bias_regularizer,
      })
    else:
      conv_op = tf.keras.layers.Conv2D
      conv_kwargs.update({
          'kernel_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'bias_initializer': tf.zeros_initializer(),
          'kernel_regularizer': self._kernel_regularizer,
          'bias_regularizer': self._bias_regularizer,
      })

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
      conv_name = 'mask-conv_{}'.format(i)
      for initializer_name in ['kernel_initializer',
                               'depthwise_initializer',
                               'pointwise_initializer']:
        if initializer_name in conv_kwargs:
          conv_kwargs[initializer_name] = tf_utils.clone_initializer(
              conv_kwargs[initializer_name])
      self._convs.append(conv_op(name=conv_name, **conv_kwargs))
      bn_name = 'mask-conv-bn_{}'.format(i)
      self._conv_norms.append(bn_op(name=bn_name, **bn_kwargs))

    self._deconv = tf.keras.layers.Conv2DTranspose(
        filters=self._num_filters,
        kernel_size=self._upsample_factor,
        strides=self._upsample_factor,
        padding='valid',
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2, mode='fan_out', distribution='untruncated_normal'),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        name='mask-upsampling')
    self._deconv_bn = bn_op(name='mask-deconv-bn', **bn_kwargs)

    if self._class_agnostic:
      num_filters = 1
    else:
      num_filters = self._num_classes
    conv_kwargs.update({
        'filters': num_filters,
        'kernel_size': 1,
        'padding': 'valid',
    })
    self._mask_regressor = conv_op(name='mask-logits', **conv_kwargs)
    
    super(MaskHead, self).build(input_shape)

  def call(self, inputs: List[tf.Tensor], training: Optional[bool] = None):
    """Forward pass of mask branch for the Mask-RCNN model.

    Parameters
    ----------
    inputs : list of tf.Tensor
        inputs[0]: A `tf.Tensor` of shape [batch_size, num_instances,
          roi_height, roi_width, roi_channels], representing the ROI features.
        inputs[1]: A `tf.Tensor` of shape [batch_size, num_instances],
          representing the classes of the ROIs.
    training : bool, optional
        Whether it is in `training` mode.

    Returns
    -------
    tf.Tensor
        [batch_size, num_instances, roi_height * upsample_factor,
        roi_width * upsample_factor], representing the mask predictions.
    """
    roi_features, roi_classes = inputs
    batch_size, num_rois, height, width, filters = \
        roi_features.get_shape().as_list()

    if batch_size is None:
      batch_size = tf.shape(roi_features)[0]

    x = tf.reshape(roi_features, [-1, height, width, filters])
    for conv, bn in zip(self._convs, self._conv_norms):
      x = conv(x)
      x = bn(x)
      x = self._activation_layer(x)

    x = self._deconv(x)
    x = self._deconv_bn(x)
    x = self._activation_layer(x)

    logits = self._mask_regressor(x)

    mask_height = height * self._upsample_factor
    mask_width = width * self._upsample_factor

    if self._class_agnostic:
      logits = tf.reshape(logits, [-1, num_rois, mask_height, mask_width, 1])
    else:
      logits = tf.reshape(
          logits,
          [-1, num_rois, mask_height, mask_width, self._num_classes])

    batch_indices = tf.tile(
        tf.expand_dims(tf.range(batch_size), axis=1), [1, num_rois])
    mask_indices = tf.tile(
        tf.expand_dims(tf.range(num_rois), axis=0), [batch_size, 1])

    if self._class_agnostic:
      class_gather_indices = tf.zeros_like(roi_classes, dtype=tf.int32)
    else:
      class_gather_indices = tf.cast(roi_classes, dtype=tf.int32)

    gather_indices = tf.stack(
        [batch_indices, mask_indices, class_gather_indices], axis=2)
    mask_outputs = tf.gather_nd(
        tf.transpose(logits, [0, 1, 4, 2, 3]), gather_indices)
    return mask_outputs

  def get_config(self):
    config = super(MaskHead, self).get_config()
    config.update({
        'num_classes': self._num_classes,
        'upsample_factor': self._upsample_factor,
        'num_convs': self._num_convs,
        'num_filters': self._num_filters,
        'use_separable_conv': self._use_separable_conv,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'class_agnostic': self._class_agnostic,
    })
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
