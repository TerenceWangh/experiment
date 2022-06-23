"""Contains definitions of box sampler."""

import tensorflow as tf

from experiment.ops import sampling_ops


@tf.keras.utils.register_keras_serializable(package='experiment')
class BoxSampler(tf.keras.layers.Layer):
  """Creates a BoxSampler to sample positive and negative boxes."""

  def __init__(self,
               num_samples: int = 512,
               foreground_fraction: float = 0.25,
               **kwargs):
    """Initializes a box sampler.

    Parameters
    ----------
    num_samples : int, default 512
        The number of sampled boxes per image.
    foreground_fraction : float, default 0.25
        The percentage of boxes should be sampled from the positive examples.
    kwargs : dict
        Additional keyword arguments passed to Layer.
    """
    super(BoxSampler, self).__init__(**kwargs)
    self._num_samples = num_samples
    self._foreground_fraction = foreground_fraction

  def call(self,
           positive_matches: tf.Tensor,
           negative_matches: tf.Tensor,
           ignored_matches: tf.Tensor):
    """Samples and selects positive and negative instances.

    Parameters
    ----------
    positive_matches : tf.Tensor
        The tensor of shape of [batch, N] where N is the number of instances.
        For each element, `True` means the instance corresponds to a positive
        example.
    negative_matches : tf.Tensor
        The  tensor of shape of [batch, N] where N is the number of instances.
        For each element, `True` means the instance corresponds to a negative
        example.
    ignored_matches : tf.Tensor
        The tensor of shape of [batch, N] where N is the number of instances.
        For each element, `True` means the instance should be ignored.

    Returns
    -------
    tf.Tensor
        A `tf.tensor` of shape of [batch_size, K], storing the indices of the
        sampled examples, where K is `num_samples`.
    """
    sample_candidates = tf.logical_and(
        tf.logical_or(positive_matches, negative_matches),
        tf.logical_not(ignored_matches))
    sampler = sampling_ops.BalancedPositiveNegativeSampler(
        positive_fraction=self._foreground_fraction, is_static=True)
    batch_size = sample_candidates.shape[0]
    sampled_indicators = []
    for i in range(batch_size):
      sampled_indicator = sampler.subsample(
          sample_candidates[i],
          self._num_samples,
          positive_matches[i])
      sampled_indicators.append(sampled_indicator)
    sampled_indicators = tf.stack(sampled_indicators)
    _, selected_indices = tf.nn.top_k(
        tf.cast(sampled_indicators, dtype=tf.int32),
        k=self._num_samples, sorted=True)

    return selected_indices

  def get_config(self):
    config = super(BoxSampler, self).get_config()
    config.update({
        'num_samples': self._num_samples,
        'foreground_fraction': self._foreground_fraction,
    })
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)
