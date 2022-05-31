"""Gaussian error linear unit."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Text')
def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415

  :param x: float Tensor to perform activation.
  :return: `x` with the GELU activation applied.
  """
  return tf.keras.activations.gelu(x, approximate=True)
