"""Provides a utility class for managing summary writing."""

import os
import tensorflow as tf


class SummaryManager:
  """A utility class for managing summary writing."""

  def __init__(self, summary_dir, summary_fn, global_step=None):
    """Initializes the `SummaryManager` instance.

    :param summary_dir: The directory in which to write summaries. If `None`,
        all summary writing operations provided by this class are no-ops.
    :param summary_fn: A callable defined accepting `name`, `value`, and `step`
        parameters, making calls to `tf.summary` functions to write summaries.
    :param global_step: A `tf.Variable` containing the global step value.
    """
    self._enable = summary_dir is not None
    self._summary_dir = summary_dir
    self._summary_fn = summary_fn
    self._summary_writers = {}

    if global_step is None:
      self._global_step = tf.summary.experimental.get_step()
    else:
      self._global_step = global_step

  def summary_writer(self, relative_path=''):
    """Returns the underlying summary writer for a specific subdirectory.

    :param relative_path: The current path in which to write summaries, relative
        to the summary directory. By default it is empty, which corresponds to
        the root directory.
    :return: The underlying summary.
    """
    if self._summary_writers and relative_path in self._summary_writers:
      return self._summary_writers[relative_path]

    if self._enable:
      self._summary_writers[relative_path] = tf.summary.create_file_writer(
          os.path.join(self._summary_dir, relative_path))
    else:
      self._summary_writers[relative_path] = tf.summary.create_noop_writer()
    return self._summary_writers[relative_path]

  def flush(self):
    """Flushes the underlying summary writer."""
    if self._enable:
      tf.nest.map_structure(tf.summary.flush, self._summary_writers)

  def write_summaries(self, summary_dict):
    """Writes summaries for the given dictionary of values.

    This recursively creates subdirectories for any nested dictionaries
    provided in `summary_dict`, yielding a hierarchy of directories which will
    then be reflected in the TensorBoard UI as different colored curves.

    For example, users may evaluate on multiple datasets and return
    `summary_dict` as a nested dictionary:

      {
        "dataset1": {
          "loss": loss1,
          "accuracy": accuracy1,
        },
        "dataset2": {
          "loss": loss2,
          "accuracy": accuracy2,
        },
      }

    This will create two subdirectories, "dataset1" and "dataset2", inside the
    summary root directory. Each directory will contain event files including
    both "loss" and "accuracy" summaries.

    :param summary_dict: A dictionary of values. If any value in `summary_dict`
        is itself a dictionary, then the function will create a subdirectory
        with name given by the corresponding key. This is performed recursively.
        Leaf values are then summarized using the summary writer instance
        specific to the parent relative path.
    """
    if not self._enable:
      return
    self._write_summaries(summary_dict)

  def _write_summaries(self, summary_dict, relative_path=''):
    for name, value in summary_dict.items():
      if isinstance(value, dict):
        self._write_summaries(
            value, relative_path=os.path.join(relative_path, name))
      else:
        with self.summary_writer(relative_path).as_default():
          self._summary_fn(name, value, step=self._global_step)
