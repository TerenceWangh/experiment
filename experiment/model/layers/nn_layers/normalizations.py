from typing import Optional
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="experiment")
class GroupNormalization(tf.keras.layers.Layer):
  """Group normalization layer.

  Source: "Group Normalization" (Yuxin Wu & Kaiming He, 2018)
  https://arxiv.org/pdf/1803.08494.pdf

  Group Normalization divides the channels into groups and computes
  within each group the mean and variance for normalization.
  Empirically, its accuracy is more stable than batch norm in a wide
  range of small batch sizes, if learning rate is adjusted linearly
  with batch sizes.

  Relation to Layer Normalization:
  If the number of groups is set to 1, then this operation becomes identical to
  Layer Normalization.

  Relation to Instance Normalization:
  If the number of groups is set to the input dimension (number of groups is
  equal to number of channels), then this operation becomes identical to
  Instance Normalization.
  """

  def __init__(self,
               groups: int = 32,
               axis: int = -1,
               epsilon: float = 1e-3,
               center: bool = True,
               scale: bool = True,
               beta_initializer: str = 'zeros',
               gamma_initializer: str = 'ones',
               beta_regularizer: Optional[
                   tf.keras.regularizers.Regularizer] = None,
               gamma_regularizer: Optional[
                   tf.keras.regularizers.Regularizer] = None,
               beta_constraint: Optional[
                   tf.keras.constraints.Constraint] = None,
               gamma_constraint: Optional[
                   tf.keras.constraints.Constraint] = None,
               **kwargs):
    """Initializes group normalization.

    Parameters
    ----------
    groups : int, default 32
        The number of groups for Group Normalization. Can be range [1, N] where
        N is the input dimension. The input dimension must be divisible by the
        number of groups.
    axis : int, default -1
        The axis that should be normalized.
    epsilon : float 1e-3
        Small value added to variance to avoid dividing by zero.
    center : bool, default True
        Whether add offset of `beta` to normalized tensor.
    scale : bool, default True
        Whether multiply by `gamma`.
    beta_initializer : str, default 'zeros'
        Initializer for the beta weight.
    gamma_initializer : str, default 'ones'
        Initializer for the gamma weight.
    beta_regularizer : tf.keras.regularizers.Regularizer, optional
        The regularizer for the beta weight.
    gamma_regularizer : tf.keras.regularizers.Regularizer, optional
        The regularizer for the gamma weight.
    beta_constraint : tf.keras.constraints.Constraint, optional
        The constraint for the beta weight.
    gamma_constraint : tf.keras.constraints.Constraint, optional
        The constraint for the gamma weight.
    kwargs : dict
        Additional keyword arguments to be passed.
    """
    super(GroupNormalization, self).__init__(**kwargs)

    self._supports_masking = True
    self._groups = groups
    self._axis = axis
    self._epsilon = epsilon
    self._center = center
    self._scale = scale
    self._beta_initializer = tf.keras.initializers.get(beta_initializer)
    self._gamma_initializer = tf.keras.initializers.get(gamma_initializer)
    self._beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
    self._gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
    self._beta_constraint = tf.keras.constraints.get(beta_constraint)
    self._gamma_constraint = tf.keras.constraints.get(gamma_constraint)
    self._check_axis()

  def build(self, input_shape):
    self._check_if_input_shape_is_none(input_shape)
    self._set_number_of_groups_for_instance_norm(input_shape)
    self._check_size_of_dimensions(input_shape)
    self._create_input_spec(input_shape)

    self._add_gamma_weight(input_shape)
    self._add_beta_weight(input_shape)

    super(GroupNormalization, self).build(input_shape)

  def call(self, inputs):
    input_shape = tf.keras.backend.int_shape(inputs)
    tensor_input_shape = tf.shape(inputs)

    reshaped_inputs, group_shape = self._reshape_into_groups(
        inputs, input_shape, tensor_input_shape)
    normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

    is_instance_norm = (input_shape[self._axis] // self._groups) == 1
    if not is_instance_norm:
      outputs = tf.reshape(normalized_inputs, tensor_input_shape)
    else:
      outputs = normalized_inputs

    return outputs

  def get_config(self):
    config = super(GroupNormalization, self).get_config()
    config.update({
        'group': self._groups,
        'axis': self._axis,
        'epsilon': self._epsilon,
        'center': self._center,
        'scale': self._scale,
        'beta_initializer': tf.keras.initializers.serialize(
            self._beta_initializer),
        'gamma_initializer': tf.keras.initializers.serialize(
            self._gamma_initializer),
        'beta_regularizer': tf.keras.regularizers.serialize(
            self._beta_regularizer),
        'gamma_regularizer': tf.keras.regularizers.serialize(
            self._gamma_regularizer),
        'beta_constraint': tf.keras.constraints.serialize(
            self._beta_constraint),
        'gamma_constraint': tf.keras.constraints.serialize(
            self._gamma_constraint),
    })
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):
    group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
    is_instance_norm = (input_shape[self._axis] // self._groups) == 1
    if is_instance_norm:
      return inputs, group_shape

    group_shape[self._axis] = input_shape[self._axis] // self._groups
    group_shape.insert(self._axis, self._groups)
    group_shape = tf.stack(group_shape)
    reshaped_inputs = tf.reshape(inputs, group_shape)
    return reshaped_inputs, group_shape

  def _apply_normalization(self, reshaped_inputs, input_shape):
    group_shape = tf.keras.backend.int_shape(reshaped_inputs)
    group_reduction_axes = list(range(1, len(group_shape)))
    is_instance_norm = (input_shape[self._axis] // self._groups) == 1
    if is_instance_norm:
      axis = -1 if self._axis == -1 else self._axis - 1
    else:
      axis = -2 if self._axis == -1 else self._axis - 1
    group_reduction_axes.pop(axis)

    mean, variance = tf.nn.moments(
        reshaped_inputs, group_reduction_axes, keepdims=True)

    gamma, beta = self._get_reshaped_weights(input_shape)
    normalized_inputs = tf.nn.batch_normalization(
        reshaped_inputs,
        mean=mean,
        variance=variance,
        scale=gamma,
        offset=beta,
        variance_epsilon=self._epsilon)
    return normalized_inputs

  def _get_reshaped_weights(self, input_shape):
    broadcast_shape = self._create_broadcast_shape(input_shape)
    gamma, beta = None, None
    if self._scale:
      gamma = tf.reshape(self._gamma, broadcast_shape)
    if self._center:
      beta = tf.reshape(self._beta, broadcast_shape)
    return gamma, beta

  def _check_if_input_shape_is_none(self, input_shape):
    dim = input_shape[self._axis]
    if dim is None:
      raise ValueError(
          'Axis {} of input tensor should have a defined dimension but the '
          'layer received an input with shape {}.'.format(
              self._axis, input_shape))

  def _set_number_of_groups_for_instance_norm(self, input_shape):
    dim = input_shape[self._axis]
    if self._groups == -1:
      self._groups = dim

  def _check_size_of_dimensions(self, input_shape):
    dim = input_shape[self._axis]
    if dim < self._groups:
      raise ValueError(
          'Number of groups ({}) cannot be more than number of channels '
          '({}).'.format(self._groups, dim))
    if dim % self._groups != 0:
      raise ValueError(
          'Number of groups ({}) must be a multiple of the number of channels '
          '({}).'.format(self._groups, dim))

  def _check_axis(self):
    if self._axis == 0:
      raise ValueError(
          'You are trying to normalize your batch axis. Do you want to use'
          'tf.layer.batch_normalization instead.')

  def _create_input_spec(self, input_shape):
    dim = input_shape[self._axis]
    self._input_spec = tf.keras.layers.InputSpec(
        ndim=len(input_shape), axes={self._axis: dim})

  def _add_gamma_weight(self, input_shape):
    dim = input_shape[self._axis]
    shape = (dim,)
    if self._scale:
      self._gamma = self.add_weight(
          shape=shape,
          name='gamma',
          initializer=self._gamma_initializer,
          regularizer=self._gamma_regularizer,
          constraint=self._gamma_constraint,
      )
    else:
      self._gamma = None

  def _add_beta_weight(self, input_shape):
    dim = input_shape[self._axis]
    shape = (dim,)
    if self._center:
      self._beta = self.add_weight(
          shape=shape,
          name='beta',
          initializer=self._beta_initializer,
          regularizer=self._beta_regularizer,
          constraint=self._beta_constraint,
      )
    else:
      self._beta = None

  def _create_broadcast_shape(self, input_shape):
    broadcast_shape = [1] * len(input_shape)
    is_instance_norm = (input_shape[self._axis] // self._groups) == 1
    if is_instance_norm:
      broadcast_shape[self._axis] = self._groups
    else:
      broadcast_shape[self._axis] = input_shape[self._axis] // self._groups
      broadcast_shape.insert(self._axis, self._groups)
    return broadcast_shape
