"""Abstraction of multi-task model."""
from typing import Text, Dict
import tensorflow as tf


class MultiTaskBaseModel(tf.Module):
  """Base class that holds multi-task model computation."""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._sub_tasks = self._instantiate_sub_tasks()

  def _instantiate_sub_tasks(self) -> Dict[Text, tf.keras.Model]:
    """Abstract function that sets up the computation for each sub-task.

    :return: A map from task name (as string) to a tf.keras.Model object that
        represents the sub-task in the multi-task pool.
    """
    raise NotImplementedError(
        '_instantiate_sub_task_models() is not implemented.')

  @property
  def sub_task(self):
    """Fetch a map of task name (string) to task model (tf.keras.Model)."""
    return self._sub_tasks

  def initialize(self):
    """Optional function that loads a pre-train checkpoint."""
    return
