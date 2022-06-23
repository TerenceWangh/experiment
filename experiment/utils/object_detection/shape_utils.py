"""Utils used to manipulate tensor shapes."""

import tensorflow as tf


def assert_shape_equal(shape1, shape2):
  """Asserts that shape1 and shape2 are equal.

  If the shapes are static, raises a ValueError when the shapes
  mismatch.

  Parameters
  ----------
  shape1: array_like
      a list containing shape of the first tensor.
  shape2: array_like
      a list containing shape of the second tensor.

  Returns
  -------
      Either a tf.no_op() when shapes are all static and a tf.assert_equal() op
      when the shapes are dynamic.

  Raises
  ------
      ValueError: When shapes are both static and unequal.
  """
  if (all(isinstance(dim, int) for dim in shape1) and
      all(isinstance(dim, int) for dim in shape2)):
    if shape1 != shape2:
      raise ValueError('Unequal shapes {}, {}'.format(shape1, shape2))
    else:
      return tf.no_op()
  else:
    return tf.assert_equal(shape1, shape2)


def combined_static_and_dynamic_shape(tensor: tf.Tensor):
  """Returns a list containing static and dynamic values for the dimensions.

  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.

  Parameters
  ----------
  tensor: tf.Tensor.
      The input tensor.

  Returns
  -------
      A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  s_tensor_shape = tensor.shape.as_list()
  d_tensor_shape = tf.shape(input=tensor)
  combined_shape = []
  for index, dim in enumerate(s_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(d_tensor_shape[index])
  return combined_shape


def pad_or_clip_nd(tensor, output_shape):
  """Pad or Clip given tensor to the output shape.

  Parameters
  ----------
  tensor: tf.Tensor.
      Input tensor to pad or clip.
  output_shape: array_like.
      The size to pad or clip each dimension of the input tensor.

  Returns
  -------
      Input tensor padded and clipped to the output shape.
  """
  tensor_shape = tf.shape(input=tensor)
  clip_size = [
      tf.where(tensor_shape[i] - shape > 0, shape, -1)
      if shape is not None else -1 for i, shape in enumerate(output_shape)
  ]
  clipped_tensor = tf.slice(
      tensor, begin=tf.zeros(len(clip_size), dtype=tf.int32), size=clip_size)

  # Pad tensor if the shape of clipped tensor is smaller than the expected
  # shape.
  clipped_tensor_shape = tf.shape(input=clipped_tensor)
  trailing_paddings = [
      shape - clipped_tensor_shape[i] if shape is not None else 0
      for i, shape in enumerate(output_shape)
  ]
  paddings = tf.stack(
      [tf.zeros(len(trailing_paddings), dtype=tf.int32), trailing_paddings],
      axis=1)
  padded_tensor = tf.pad(tensor=clipped_tensor, paddings=paddings)
  output_static_shape = [
      dim if not isinstance(dim, tf.Tensor) else None for dim in output_shape
  ]
  padded_tensor.set_shape(output_static_shape)
  return padded_tensor
