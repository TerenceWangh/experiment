"""Common TF utilities."""

from typing import Optional, Tuple, Dict, Callable, Union
import six
import tensorflow as tf

from experiment.model import activations


# Type annotations
States = Dict[str, tf.Tensor]
Activation = Union[str, Callable]


def is_special_none_tensor(tensor):
  """Checks if a tensor is a special None Tensor."""
  return tensor.shape.ndims == 0 and tensor.dtype == tf.int32


def get_activation(identifier, use_keras_layer=False):
  """Maps a identifier to a Python function, e.g., "relu" => `tf.nn.relu`.

  It checks string first and if it is one of customized activation not in TF,
  the corresponding activation will be returned. For non-customized activation
  names and callable identifiers, always fallback to tf.keras.activations.get.

  Prefers using keras layers when use_keras_layer=True. Now it only supports
  'relu', 'linear', 'identity', 'swish'.

  :param identifier: String name of the activation function or callable.
  :param use_keras_layer: If True, use keras layer if identifier is
      allow-listed.
  :return: A Python function corresponding to the activation function or a keras
      activation layer when use_keras_layer=True.
  """
  if isinstance(identifier, six.string_types):
    identifier = str(identifier).lower()
    if use_keras_layer:
      # pylint: disable=bad-whitespace
      keras_layer_allow_list = {
          'relu'          : 'relu',
          'linear'        : 'linear',
          'identity'      : 'linear',
          'swish'         : 'swish',
          'sigmoid'       : 'sigmoid',
          'relu6'         : tf.nn.relu6,
          'hard_swish'    : activations.hard_swish,
          'hard_sigmoid'  : activations.hard_sigmoid,
      }
      # pylint: enable=bad-whitespace
      if identifier in keras_layer_allow_list:
        return tf.keras.layers.Activation(keras_layer_allow_list[identifier])

    # pylint: disable=bad-whitespace
    name_to_fn = {
        'gelu'        : activations.gelu,
        'simple_swish': activations.simple_swish,
        'hard_swish'  : activations.hard_swish,
        'relu6'       : activations.relu6,
        'hard_sigmoid': activations.hard_sigmoid,
        'identity'    : activations.identity,
    }
    # pylint: enable=bad-whitespace
    if identifier in name_to_fn:
      return tf.keras.activations.get(name_to_fn[identifier])
  return tf.keras.activations.get(identifier)


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  :param tensor: A tf.Tensor object to find the shape of.
  :param expected_rank: (optional) int. The expected rank of `tensor`. If this
      is specified and the `tensor` has a different rank, and exception will be
      thrown.
  :param name: Optional name of the tensor for the error message.
  :return: A list of dimensions of the shape of tensor. All static dimensions
      will be returned as python integers, and dynamic dimensions will be
      returned as tf.Tensor scalars.
  """
  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for index, dim in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dynamic_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dynamic_shape[index]
  return shape


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  :param tensor: A tf.Tensor to check the rank of.
  :param expected_rank: Python integer or list of integers, expected rank.
  :param name: Optional name of the tensor for the error message.
  :raise ValueError: If the expected shape doesn't match the actual shape.
  """
  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    raise ValueError(
        'For the tensor `{}`, the actual rank `{}` (shape = {}) is not equal '
        'to the expected tensor rank `{}`'.format(
            name, actual_rank, str(tensor.shape), str(expected_rank)))


def safe_mean(losses):
  """Computes a safe mean of the losses

  :param losses: `Tensor` whose elements contain individual loss measurements.
  :return: A scalar representing the mean of `losses`. If `num_present` is zero,
      then zero is returned.
  """
  total = tf.reduce_sum(losses)
  num_elements = tf.cast(tf.size(losses), dtype=losses.dtype)
  return tf.math.divide_no_nan(total, num_elements)


def get_replica_id():
  """Gets replica id depending on the environment."""
  context = tf.distribute.get_replica_context()
  if context is not None:
    return context.replica_id_in_sync_group
  else:
    raise RuntimeError('Unknown replica context. The `get_replica_id` method '
                       'replies on TF 2.x tf.distribute API.')


def cross_replica_concat(value, axis, name='cross_replica_concat'):
  """Concatenates the given `value` across GPU cores, along `axis`.

  In general, each core ("replica") will pass a
  replica-specific value as `value` (corresponding to some element of a
  data-parallel computation taking place across replicas).

  The resulting concatenated `Tensor` will have the same shape as `value` for
  all dimensions except `axis`, where it will be larger by a factor of the
  number of replicas. It will also have the same `dtype` as `value`.

  The position of a given replica's `value` within the resulting concatenation
  is determined by that replica's replica ID.

  For example:
  With `value` for replica 0 given as
      0 0 0
      0 0 0
  and `value` for replica 1 given as
      1 1 1
      1 1 1

  the resulting concatenation along axis 0 will be
      0 0 0
      0 0 0
      1 1 1
      1 1 1
  and this result will be identical across all replicas.

  Note that this API only works in TF2 with `tf.distribute`.

  :param value: The `Tensor` to concatenate across replicas. Each replica will
      have a different value for this `Tensor`, and these replica-specific
      values will be concatenated.
  :param axis: The axis along which to perform the concatenation as a Python
      integer (not a `Tensor`). E.g., `axis=0` to concatenate along the batch
      dimension.
  :param name: A name for the operation (used to create a name scope).
  :return: The result of concatenating `value` along `axis` across replicas.
  :raise RuntimeError: when the batch (0-th) dimension is None.
  """
  with tf.name_scope(name):
    context = tf.distribute.get_replica_context()
    # Typically this could be hit only if the tensor is derived from a dataset
    # with finite epochs and drop_remainder=False, where the last batch could
    # of different batch size and then the dim-0 is of dynamic shape.
    if value.shape.as_list()[0] is None:
      raise RuntimeError('{} has unknown batch.'.format(value))
    return context.all_gather(value, axis=axis)


def make_divisible(value: float, divisor: int,
                   min_value: Optional[float] = None,
                   round_down_protect: bool = True) -> int:
  """This is to ensure that all layers have channels that are divisible by 8.

  Parameters
  ----------
  value : float
      The original value.
  divisor : int
      The divisor that need to be checked upon.
  min_value : float, optional
      The minimum value threshold.
  round_down_protect : bool, default True
      Whether round down more than 10% will be allowed.

  Returns
  -------
  int
      The adjusted value in `int` that is divisible against divisor.
  """
  if min_value is None:
    min_value = divisor
  new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if round_down_protect and new_value < 0.9 * value:
    new_value += divisor
  return new_value


def round_filters(filters: int,
                  multiplier: float,
                  divisor: int = 8,
                  min_depth: Optional[int] = None,
                  round_down_protect: bool = True,
                  skip: bool = False) -> int:
  """Rounds number of filters based on width multiplier."""
  if skip or not multiplier:
    return filters

  new_f = make_divisible(value=filters * multiplier,
                         divisor=divisor,
                         min_value=min_depth,
                         round_down_protect=round_down_protect)
  return int(new_f)


def get_padding_for_kernel_size(kernel_size) -> Tuple[int]:
  """Compute padding size given kernel size."""
  if kernel_size == 7:
    return (3, 3)
  if kernel_size == 3:
    return (1, 1)
  raise ValueError('Padding for kernel size {} not known.'.format(kernel_size))


def clone_initializer(initializer):
  # Keras initializer is going to be stateless, which mean reusing the same
  # initializer will produce same init value when the shapes are the same.
  if isinstance(initializer, tf.keras.initializers.Initializer):
    return initializer.__class__.from_config(initializer.get_config())
  # When the input is string/dict or other serialized configs, caller will
  # create a new keras Initializer instance based on that, and we don't need to
  # do anything
  return initializer
