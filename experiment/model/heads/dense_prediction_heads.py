"""Contains definitions of dense prediction heads."""

from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import tensorflow as tf

from experiment.model import tf_utils


@tf.keras.utils.register_keras_serializable(package='experiment')
class RetinaNetHead(tf.keras.layers.Layer):
  """RetinaNet head."""

  def __init__(self,
               min_level: int,
               max_level: int,
               num_classes: int,
               num_anchors_per_location: int,
               num_convs: int = 4,
               num_filters: int = 256,
               attribute_heads: Optional[List[Dict[str, Any]]] = None,
               use_separable_conv: bool = False,
               activation: str = 'relu',
               use_sync_bn: bool = False,
               norm_momentum: float = 0.99,
               norm_epsilon: float = 0.001,
               kernel_regularizer: Optional[
                   tf.keras.regularizers.Regularizer] = None,
               bias_regularizer: Optional[
                   tf.keras.regularizers.Regularizer] = None,
               num_params_per_anchor: int = 4,
               **kwargs):
    """Initializes a RetinaNet head.

    Parameters
    ----------
    min_level : int
        The number of minimum feature level.
    max_level : int
        The number of maximum feature level.
    num_classes : int
        The number of classes to predict.
    num_anchors_per_location : int
        The number of number of anchors per pixel location.
    num_convs : int, default 4
        The number that represents the number of the intermediate conv layers
        before the prediction.
    num_filters : int, default 256
        The number that represents the number of filters of the intermediate
        conv layers.
    attribute_heads : List[Dict[str, Any]], optional
        If not None, a list that contains a dict for each additional attribute
        head. Each dict consists of 3 key-value pairs: `name`, `type`
        ('regression' or 'classification'), and `size` (number of predicted
        values for each instance).
    use_separable_conv : bool, default False
        Whether the separable convolution layers is used.
    activation : str, default 'relu'
        Which activation is used, e.g. 'relu', 'swish', etc.
    use_sync_bn : bool, default False
        Whether to use synchronized batch normalization across different
        replicas.
    norm_momentum : float, default 0.99
        The normalization momentum for the moving average.
    norm_epsilon : float, default 0.001
        The value added to variance to avoid dividing by zero.
    kernel_regularizer : tf.keras.regularizers.Regularizer, optional
        A `tf.keras.regularizers.Regularizer` object for Conv2D.
    bias_regularizer : tf.keras.regularizers.Regularizer, optional
        A `tf.keras.regularizers.Regularizer` object for bias.
    num_params_per_anchor : int, default 4
        Number of parameters required to specify an anchor box. For example,
        `num_params_per_anchor` would be 4 for axis-aligned anchor boxes
        specified by their y-centers, x-centers, heights, and widths.
    kwargs : dict
        Additional keyword arguments to be passed.
    """
    super(RetinaNetHead, self).__init__(**kwargs)
    self._min_level = min_level
    self._max_level = max_level
    self._num_classes = num_classes
    self._num_anchors_per_location = num_anchors_per_location
    self._num_convs = num_convs
    self._num_filters = num_filters
    self._attribute_heads = attribute_heads
    self._use_separable_conv = use_separable_conv
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._num_params_per_anchor = num_params_per_anchor

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation = tf_utils.get_activation(activation)

  def build(self, input_shape: Union[tf.TensorShape, List[tf.TensorShape]]):
    """Creates the variables of the head."""
    if self._use_separable_conv:
      conv_op = tf.keras.layers.SeparableConv2D
    else:
      conv_op = tf.keras.layers.Conv2D

    if self._use_sync_bn:
      bn_op = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      bn_op = tf.keras.layers.BatchNormalization

    conv_kwargs = {
        'filters': self._num_filters,
        'kernel_size': 3,
        'padding': 'same',
        'bias_initializer': tf.zeros_initializer(),
        'bias_regularizer': self._bias_regularizer,
    }

    if not self._use_separable_conv:
      conv_kwargs.update({
          'kernel_initializer': tf.keras.initializers.RandomNormal(stddev=0.01),
          'kernel_regularizer': self._kernel_regularizer,
      })

    bn_kwargs = {
        'axis': self._bn_axis,
        'momentum': self._norm_momentum,
        'epsilon': self._norm_epsilon,
    }

    # Classifier net
    self._cls_convs = []
    self._cls_norms = []
    for level in range(self._min_level, self._max_level + 1):
      this_level_cls_norms = []
      for i in range(self._num_convs):
        if level == self._min_level:
          cls_conv_name = 'classnet-conv_{}'.format(i)
          if 'kernel_initializer' in conv_kwargs:
            conv_kwargs['kernel_initializer'] = tf_utils.clone_initializer(
                conv_kwargs['kernel_initializer'])
          self._cls_convs.append(conv_op(name=cls_conv_name, **conv_kwargs))
        cls_norm_name = 'classnet-conv-norm_{}_{}'.format(level, i)
        this_level_cls_norms.append(bn_op(name=cls_norm_name, **bn_kwargs))
      self._cls_norms.append(this_level_cls_norms)

    classifier_kwargs = {
        'filters': self._num_classes * self._num_anchors_per_location,
        'kernel_size': 3,
        'padding': 'same',
        'bias_initializer': tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
        'bias_regularizer': self._bias_regularizer,
    }
    if not self._use_separable_conv:
      classifier_kwargs.update({
          'kernel_initializer': tf.keras.initializers.RandomNormal(stddev=1e-5),
          'kernel_regularizer': self._kernel_regularizer,
      })
    self._classifier = conv_op(name='scores', **classifier_kwargs)

    # Box net
    self._box_convs = []
    self._box_norms = []
    for level in range(self._min_level, self._max_level + 1):
      this_level_box_norms = []
      for i in range(self._num_convs):
        if level == self._min_level:
          box_conv_name = 'boxnet-conv_{}'.format(i)
          if 'kernel_initializer' in conv_kwargs:
            conv_kwargs['kernel_initializer'] = tf_utils.clone_initializer(
                conv_kwargs['kernel_initializer'])
          self._box_convs.append(conv_op(name=box_conv_name, **conv_kwargs))
        box_norm_name = 'boxnet-conv-norm_{}_{}'.format(level, i)
        this_level_box_norms.append(bn_op(name=box_norm_name, **bn_kwargs))
      self._box_norms.append(this_level_box_norms)

    box_regressor_kwargs = {
        'filters': self._num_params_per_anchor * self._num_anchors_per_location,
        'kernel_size': 3,
        'padding': 'same',
        'bias_initializer': tf.zeros_initializer(),
        'bias_regularizer': self._bias_regularizer,
    }
    if not self._use_separable_conv:
      box_regressor_kwargs.update({
          'kernel_initializer': tf.keras.initializers.RandomNormal(stddev=1e-5),
          'kernel_regularizer': self._kernel_regularizer,
      })
    self._box_regressor = conv_op(name='boxes', **box_regressor_kwargs)

    # Attribute learning nets
    if self._attribute_heads:
      self._attribute_predictors = {}
      self._attribute_convs = {}
      self._attribute_norms = {}

      for attribute_config in self._attribute_heads:
        attribute_name = attribute_config['name']
        attribute_type = attribute_config['type']
        attribute_size = attribute_config['size']
        attribute_convs_i = []
        attribute_norms_i = []

        # Build conv and norm layers.
        for level in range(self._min_level, self._max_level + 1):
          this_level_attribute_norms = []
          for i in range(self._num_convs):
            if level == self._min_level:
              attribute_conv_name = '{}-conv_{}'.format(attribute_name, i)
              if 'kernel_initializer' in conv_kwargs:
                conv_kwargs['kernel_initializer'] = tf_utils.clone_initializer(
                    conv_kwargs['kernel_initializer'])
              attribute_convs_i.append(conv_op(name=attribute_conv_name,
                                               **conv_kwargs))
            attribute_norm_name = '{}-conv-norm_{}_{}'.format(
                attribute_name, level, i)
            this_level_attribute_norms.append(bn_op(name=attribute_norm_name,
                                                    **bn_kwargs))
          attribute_norms_i.append(this_level_attribute_norms)
        self._attribute_convs[attribute_name] = attribute_convs_i
        self._attribute_norms[attribute_name] = attribute_norms_i

        # Build the final prediction layer
        attribute_predictor_kwargs = {
            'filters': attribute_size * self._num_anchors_per_location,
            'kernel_size': 3,
            'padding': 'same',
            'bias_regularizer': self._bias_regularizer,
        }

        if attribute_type == 'regression':
          attribute_predictor_kwargs.update({
              'bias_initializer': tf.zeros_initializer(),
          })
        elif attribute_type == 'classification':
          attribute_predictor_kwargs.update({
              'bias_initializer': tf.constant_initializer(
                  -np.log((1 - 0.01) / 0.01))
          })
        else:
          raise ValueError(
              'Attribute head type {} not supported.'.format(attribute_type))

        if not self._use_separable_conv:
          attribute_predictor_kwargs.update({
              'kernel_initializer': tf.keras.initializers.RandomNormal(
                  stddev=1e-5),
              'kernel_regularizer': self._kernel_regularizer,
          })

        self._attribute_predictors[attribute_name] = conv_op(
            name='{}_attributes'.format(attribute_name),
            **attribute_predictor_kwargs)

    super(RetinaNetHead, self).build(input_shape)

  def call(self, features: Mapping[str, tf.Tensor]):
    """Forward pass of the RetinaNet head.

    Parameters
    =========
    features : dict of tf.Tensor
        key : str
            The level of the multilevel features.
        values : tf.Tensor
            The feature map tensors, whose shape is [batch, height, width, chs].

    Returns
    =======
    dict
        scores: The scores of the prediction.
        boxes: The coordinates of the predictions.
        attributes: A dict of (attribute_name, attribute_prediction)
    """
    scores, boxes = {}, {}
    if self._attribute_heads:
      attributes = {
          attribute['name']: {} for attribute in self._attribute_heads
      }
    else:
      attributes = {}

    for i, level in enumerate(range(self._min_level, self._max_level + 1)):
      this_level_features = features[str(level)]

      # classification net
      x = this_level_features
      for conv, norm in zip(self._cls_convs, self._cls_norms[i]):
        x = conv(x)
        x = norm(x)
        x = self._activation(x)
      scores[str(level)] = self._classifier(x)

      # box net
      x = this_level_features
      for conv, norm in zip(self._box_convs, self._box_norms[i]):
        x = conv(x)
        x = norm(x)
        x = self._activation(x)
      boxes[str(level)] = self._box_regressor(x)

      # attribute net
      if self._attribute_heads:
        for attribute in self._attribute_heads:
          attribute_name = attribute['name']
          x = this_level_features
          for conv, norm in zip(self._attribute_convs[attribute_name],
                                self._attribute_norms[attribute_name][i]):
            x = conv(x)
            x = norm(x)
            x = self._activation(x)
          attributes[attribute_name][str(level)] = self._attribute_predictors[
              attribute_name](x)

    return scores, boxes, attributes

  def get_config(self):
    return {
        'min_level': self._min_level,
        'max_level': self._max_level,
        'num_classes': self._num_classes,
        'num_anchors_per_location': self._num_anchors_per_location,
        'num_convs': self._num_convs,
        'num_filters': self._num_filters,
        'attribute_heads': self._attribute_heads,
        'use_separable_conv': self._use_separable_conv,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'num_params_per_anchor': self._num_params_per_anchor,
    }

  @classmethod
  def from_config(cls, config):
    return cls(**config)


@tf.keras.utils.register_keras_serializable(package='experiment')
class RPNHead(tf.keras.layers.Layer):
  """Creates a Region Proposal Network (RPN) head."""

  def __init__(self,
               min_level: int,
               max_level: int,
               num_anchors_per_location: int,
               num_convs: int = 1,
               num_filters: int = 256,
               use_separable_conv: bool = False,
               activation: str = 'relu',
               use_sync_bn: bool = False,
               norm_momentum: float = 0.99,
               norm_epsilon: float = 0.001,
               kernel_regularizer: Optional[
                   tf.keras.regularizers.Regularizer] = None,
               bias_regularizer: Optional[
                   tf.keras.regularizers.Regularizer] = None,
               **kwargs):
    """Initializes a Region Proposal Network head.

    Parameters
    ----------
    min_level : int
        The number of minimum feature level.
    max_level : int
        The number of maximum feature level.
    num_anchors_per_location : int
        The number of number of anchors per pixel location.
    num_convs : int, default 4
        The number that represents the number of the intermediate conv layers
        before the prediction.
    num_filters : int, default 256
        The number that represents the number of filters of the intermediate
        conv layers.
    use_separable_conv : bool, default False
        Whether the separable convolution layers is used.
    activation : str, default 'relu'
        Which activation is used, e.g. 'relu', 'swish', etc.
    use_sync_bn : bool, default False
        Whether to use synchronized batch normalization across different
        replicas.
    norm_momentum : float, default 0.99
        The normalization momentum for the moving average.
    norm_epsilon : float, default 0.001
        The value added to variance to avoid dividing by zero.
    kernel_regularizer : tf.keras.regularizers.Regularizer, optional
        A `tf.keras.regularizers.Regularizer` object for Conv2D.
    bias_regularizer : tf.keras.regularizers.Regularizer, optional
        A `tf.keras.regularizers.Regularizer` object for bias.
    kwargs : dict
        Additional keyword arguments to be passed.
    """
    super(RPNHead, self).__init__(**kwargs)
    self._min_level = min_level
    self._max_level = max_level
    self._num_anchors_per_location = num_anchors_per_location
    self._num_convs = num_convs
    self._num_filters = num_filters
    self._use_separable_conv = use_separable_conv
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation = tf_utils.get_activation(activation)

  def build(self, input_shape):
    conv_kwargs = {
        'filters': self._num_filters,
        'kernel_size': 3,
        'padding': 'same',
        'bias_initializer': tf.zeros_initializer(),
        'bias_regularizer': self._bias_regularizer,
    }
    if self._use_separable_conv:
      conv_op = tf.keras.layers.SeparableConv2D
    else:
      conv_op = tf.keras.layers.Conv2D
      conv_kwargs.update({
          'kernel_initializer': tf.keras.initializers.RandomNormal(stddev=0.01),
          'kernel_regularizer': self._kernel_regularizer,
      })

    bn_kwrags = {
        'axis': self._bn_axis,
        'momentum': self._norm_momentum,
        'epsilon': self._norm_epsilon,
    }
    if self._use_sync_bn:
      bn_op = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      bn_op = tf.keras.layers.BatchNormalization

    self._convs = []
    self._norms = []
    for level in range(self._min_level, self._max_level + 1):
      this_level_norms = []
      for i in range(self._num_convs):
        if level == self._min_level:
          conv_name = 'rpn-conv_{}'.format(i)
          if 'kernel_initializer' in conv_kwargs:
            conv_kwargs['kernel_initializer'] = tf_utils.clone_initializer(
                conv_kwargs['kernel_initializer'])
          self._convs.append(conv_op(name=conv_name, **conv_kwargs))
        norm_name = 'rpn-conv-norm_{}_{}'.format(level, i)
        this_level_norms.append(bn_op(name=norm_name, **bn_kwrags))
      self._norms.append(this_level_norms)

    classifier_kwargs = {
        'filters': self._num_anchors_per_location,
        'kernel_size': 1,
        'padding': 'same',
        'bias_initializer': tf.zeros_initializer(),
        'bias_regularizer': self._bias_regularizer,
    }
    box_regressor_kwargs = {
        'filters': 4 * self._num_anchors_per_location,
        'kernel_size': 1,
        'padding': 'valid',
        'bias_initializer': tf.zeros_initializer(),
        'bias_regularizer': self._bias_regularizer,
    }
    if not self._use_separable_conv:
      classifier_kwargs.update({
          'kernel_initializer': tf.keras.initializers.RandomNormal(stddev=1e-5),
          'kernel_regularizer': self._kernel_regularizer,
      })
      box_regressor_kwargs.update({
          'kernel_initializer': tf.keras.initializers.RandomNormal(stddev=1e-5),
          'kernel_regularizer': self._kernel_regularizer,
      })

    self._classifier = conv_op(name='rpn-scores', **classifier_kwargs)
    self._box_regressor = conv_op(name='rpn-boxes', **box_regressor_kwargs)

    super(RPNHead, self).build(input_shape)

  def call(self, features: Mapping[str, tf.Tensor]):
    """Forward pass of the RPN head.

    Parameters
    =========
    features : dict of tf.Tensor
        key : str
            The level of the multilevel features.
        values : tf.Tensor
            The feature map tensors, whose shape is [batch, height, width, chs].

    Returns
    =======
    dict
        scores: The scores of the prediction.
        boxes: The coordinates of the predictions.
    """
    scores, boxes = {}, {}
    for i, level in enumerate(range(self._min_level, self._max_level + 1)):
      x = features[str(level)]
      for conv, norm in zip(self._convs, self._norms[i]):
        x = conv(x)
        x = norm(x)
        x = self._activation(x)
      scores[str(level)] = self._classifier(x)
      boxes[str(level)] = self._box_regressor(x)
    return scores, boxes

  def get_config(self):
    return {
        'min_level': self._min_level,
        'max_level': self._max_level,
        'num_anchors_per_location': self._num_anchors_per_location,
        'num_convs': self._num_convs,
        'num_filters': self._num_filters,
        'use_separable_conv': self._use_separable_conv,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
    }

  @classmethod
  def from_config(cls, config):
    return cls(**config)
