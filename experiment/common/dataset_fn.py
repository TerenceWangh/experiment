"""Utility library for picking an appropriate dataset function."""
import functools
from typing import Any, Callable, Type, Union

import tensorflow as tf

PossibleDatasetType = Union[Type[tf.data.Dataset], Callable[[tf.Tensor], Any]]


def pick_dataset_fn(file_type: str) -> PossibleDatasetType:
  """Get the dataset function by the given `file_type`.

  :param file_type: `tfrecord` or `tfrecord_compressed`.
  :return: The dataset function.
  """
  if file_type == 'tfrecord':
    return tf.data.TFRecordDataset
  if file_type == 'tfrecord_compressed':
    return functools.partial(tf.data.TFRecordDataset,
                             compression_type='GZIP')
  raise ValueError('Unrecognized file_type: {}'.format(file_type))
