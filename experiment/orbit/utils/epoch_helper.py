"""Provides a utility class for training in epochs."""

import tensorflow as tf


class EpochHelper:
  """A helper class handle bookkeeping of epochs in custom training loops."""

  def __init__(self, epoch_steps: int, global_step: tf.Variable):
    """Initializes the `EpochHelper` instance.

    :param epoch_steps: An integer indicating how many steps are in an epoch.
    :param global_step: A `tf.Variable` providing the current global step.
    """
    self._epoch_steps = epoch_steps
    self._global_step = global_step
    self._current_epoch = None
    self._epoch_start_step = None
    self._in_epoch = False

  def epoch_begin(self):
    """Returns whether a new epoch should begin."""
    if self._in_epoch:
      return False
    current_step = self._global_step.numpy()
    self._epoch_start_step = current_step
    self._current_epoch = current_step // self._epoch_steps
    self._in_epoch = True
    return True

  def epoch_end(self):
    """Returns whether the current epoch should end."""
    if not self._in_epoch:
      raise ValueError('`epoch_end` can only be called inside an epoch.')
    current_step = self._global_step.numpy()
    epoch = current_step // self._epoch_steps

    if epoch > self._current_epoch:
      self._in_epoch = False
      return True
    return False

  @property
  def batch_index(self):
    """Index of the next batch within the current epoch."""
    return self._global_step.numpy() - self._epoch_start_step

  @property
  def current_epoch(self):
    return self._current_epoch
