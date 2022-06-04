"""Class to subsample minibatches by balancing positives and negatives.

Subsamples minibatches based on a pre-specified positive fraction in range
[0,1]. The class presumes there are many more negatives than positive examples:
if the desired batch_size cannot be achieved with the pre-specified positive
fraction, it fills the rest with negative examples. If this is not sufficient
for obtaining the desired batch_size, it returns fewer examples.

The main function to call is Subsample(self, indicator, labels). For convenience
one can also call SubsampleWeights(self, weights, labels) which is defined in
the minibatch_sampler base class.

When is_static is True, it implements a method that guarantees static shapes.
It also ensures the length of output of the subsample is always batch_size, even
when number of examples set to True in indicator is less than batch_size.
"""
import tensorflow as tf
from experiment.utils.object_detection import shape_utils


def indices_to_dense_vector(indices,
                            size,
                            indices_value=1.,
                            default_value=0,
                            dtype=tf.float32):
  """Creates dense vector with indices set to specific value and rest to zeros.

  This function exists because it is unclear if it is safe to use
    ```tf.sparse_to_dense(indices, [size], 1, validate_indices=False)```
  with indices which are not ordered.
  This function accepts a dynamic size (e.g. tf.shape(tensor)[0])

  Parameters
  ----------
  indices : tf.Tensor
      1d Tensor with integer indices which are to be set to indices_values.
  size : int
      The size of output Tensor.
  indices_value : float, default 1.0
      The values of elements specified by indices in the output vector.
  default_value : int, default 0
      The values of other elements in the output vector.
  dtype : tf.dtypes, default tf.float32
      The data type.

  Returns
  -------
  tf.Tensor
      The dense 1D Tensor of shape [size] with indices set to indices_values and
      the rest set to default_value.
  """
  size = tf.cast(size, dtype=tf.int32)
  zeros = tf.ones([size], dtype=dtype) * default_value
  values = tf.ones_like(indices, dtype=dtype) * indices_value
  return tf.dynamic_stitch(
      [tf.range(size), tf.cast(indices, dtype=tf.int32)], [zeros, values])


def matmul_gather_on_zeroth_axis(params, indices, scope=None):
  """Matrix multiplication based implementation of tf.gather on zeroth axis.

  Parameters
  ----------
  params : tf.Tensor
      The tensor from which to gather values. Must be at least rank 1.
  indices : tf.Tensor
      Must be one of the following types: int32, int64. Must be in range
      [0, params.shape[0])
  scope : str, optional
      The name for the operation.

  Returns
  -------
  tf.Tensor
      Has the same type as params. Values from params gathered from indices
      given by indices, with shape `indices.shape + params.shape[1:]`.
  """
  with tf.name_scope(scope or 'MatMulGather'):
    params_shape = shape_utils.combined_static_and_dynamic_shape(params)
    indices_shape = shape_utils.combined_static_and_dynamic_shape(indices)
    params2d = tf.reshape(params, [params_shape[0], -1])
    indicator_matrix = tf.one_hot(indices, params_shape[0])
    gathered_result_flattened = tf.matmul(indicator_matrix, params2d)
    return tf.reshape(
        gathered_result_flattened, tf.stack(indices_shape + params_shape[1:]))
