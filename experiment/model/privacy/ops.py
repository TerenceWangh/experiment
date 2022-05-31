from typing import List, Tuple
import tensorflow as tf


def clip_l2_norm(grads_vars: List[Tuple[tf.Tensor, tf.Tensor]],
                 l2_norm_clip: float) -> List[Tuple[tf.Tensor, tf.Tensor]]:
  """Clip gradients by global norm."""
  gradients = []
  variables = []
  for g, v in grads_vars:
    gradients.append(g)
    variables.append(v)
  clipped_gradients = tf.clip_by_global_norm(gradients, l2_norm_clip)[0]
  return list(zip(clipped_gradients, variables))


def add_noise(grads_vars: List[Tuple[tf.Tensor, tf.Tensor]],
              noise_stddev: float) -> List[Tuple[tf.Tensor, tf.Tensor]]:
  """Add noise to gradients"""
  ret = []
  for g, v in grads_vars:
    noise = tf.random.normal(tf.shape(g), stddev=noise_stddev)
    ret.append((g + noise, v))
  return ret
