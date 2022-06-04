"""Build classification models."""

from typing import Optional

import tensorflow as tf

layers = tf.keras.layers
regularizers = tf.keras.regularizers

@tf.keras.utils.register_keras_serializable(package='experiment')
class ClassificationModel(tf.keras.Model):
  """A classification class builder."""

  def __init__(self,
               backbone: tf.keras.Model,
               num_classes: int,
               input_specs: tf.keras.layers.InputSpec = layers.InputSpec(
                   shape=[None, None, None, 3]),
               dropout_rate: float = 0.0,
               kernel_initializer: str = 'random_uniform',
               kernel_regularizer: Optional[regularizers.Regularizer] = None,
               bias_regularizer: Optional[regularizers.Regularizer] = None,
               add_head_batch_norm: bool = False,
               use_sync_bn: bool = False,
               norm_momentum: float = 0.99,
               norm_epsilon: float = 0.001,
               skip_logits_layer: bool = False,
               **kwargs) -> None:
    """Classification initialization function.

    Arguments
    =========
    backbone : tf.keras.Model
        A backbone network.
    num_classes : int
        The number of classes in classification task.
    input_specs : tf.keras.layers.InputSpec
        The specs of the input tensor.
    dropout_rate : float, default 0.0
        Rate for dropout regularization.
    kernel_initializer : str, default 'randim_uniform'
        The kernel initializer for the dense layer.
    kernel_regularizer : tf.keras.regularizers.Regularizer, optional
        The regularizer of kernel.
    bias_regularizer : tf.keras.regularizers.Regularizer, optional
        The regularizer of bias.
    add_head_batch_norm : bool, default False
        Whether to add a batch normalization layer before pool.
    use_sync_bn : bool default False
        Whether to use synchronized batch normalization.
    norm_momentum : float, default 0.99
        The normalization momentum for the moving average.
    norm_epsilon : float, default 0.001
        The small float added to variance to avoid dividing by zero.
    skip_logits_layer : bool, default False
        Whether to skip the prediction layer.
    **kwargs : dict
        The keyword arguments to be passed.
    """
    if use_sync_bn:
      norm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      norm = tf.keras.layers.BatchNormalization
    axis = -1 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    inputs = tf.keras.Input(shape=input_specs.shape[1:], name=input_specs.name)
    endpoints = backbone(inputs)
    x = endpoints[max(endpoints.keys())]

    if add_head_batch_norm:
      x = norm(axis=axis, momentum=norm_momentum, epsilon=norm_epsilon)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    if not skip_logits_layer:
      x = tf.keras.layers.Dropout(dropout_rate)(x)
      x = tf.keras.layers.Dense(
          num_classes,
          kernel_initializer=kernel_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer)(x)

    super(ClassificationModel, self).__init__(
        inputs=inputs, outputs=x, **kwargs)

    self._config_dict = {
        'backbone': backbone,
        'num_classes': num_classes,
        'input_specs': input_specs,
        'dropout_rate': dropout_rate,
        'kernel_initializer': kernel_initializer,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
        'add_head_batch_norm': add_head_batch_norm,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
    }
    self._input_specs = input_specs
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._backbone = backbone
    self._norm = norm
