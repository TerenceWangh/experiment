"""Data loader utils."""
from typing import Dict

import tensorflow as tf
from experiment.ops import preprocess_ops


def process_source_id(source_id: tf.Tensor) -> tf.Tensor:
  """Processes source_id to the right format.

  Parameters
  ==========
  source_id : tf.Tensor
      The source ID.

  Returns
  =======
  tf.Tensor
      The formatted source ID.
  """
  if source_id.dtype == tf.string:
    source_id = tf.strings.to_number(source_id, tf.int64)

  with tf.control_dependencies([source_id]):
    source_id = tf.cond(
        pred=tf.equal(tf.size(input=source_id), 0),
        true_fn=lambda: tf.cast(tf.constant(-1), tf.int64),
        false_fn=lambda: tf.identity(source_id))
  return source_id


def pad_groundtruths_to_fixed_size(
    groundtruths: Dict[str, tf.Tensor], size: int) -> Dict[str, tf.Tensor]:
  """Pads the first dimension of groundtruths labels to the fixed size.

  Parameters
  ----------
  groundtruths: dict
      A  dictionary of {`str`: `tf.Tensor`} that contains groundtruth
      annotations of `boxes`, `is_crowds`, `areas` and `classes`.
  size : int
      The expected size of the first dimension of padded tensors.

  Returns
  -------
  dict
      The same keys as input and padded tensors as values.
  """
  groundtruths['boxes'] = preprocess_ops.clip_or_pad_to_fixed_size(
      groundtruths['boxes'], size, -1)
  groundtruths['is_crowds'] = preprocess_ops.clip_or_pad_to_fixed_size(
      groundtruths['is_crowds'], size, 0)
  groundtruths['areas'] = preprocess_ops.clip_or_pad_to_fixed_size(
      groundtruths['areas'], size, -1)
  groundtruths['classes'] = preprocess_ops.clip_or_pad_to_fixed_size(
      groundtruths['classes'], size, -1)
  if 'attributes' in groundtruths:
    for k, v in groundtruths['attributes'].items():
      groundtruths['attributes'][k] = preprocess_ops.clip_or_pad_to_fixed_size(
          v, size, -1)
  return groundtruths
