"""Base minibatch sampler module.

The job of the minibatch_sampler is to subsample a minibatch based on some
criterion.

The main function call is:
    subsample(indicator, batch_size, **params).
Indicator is a 1d boolean tensor where True denotes which examples can be
sampled. It returns a boolean indicator where True denotes an example has been
sampled..

Subclasses should implement the Subsample function and can make use of the
@staticmethod SubsampleIndicator.
"""
from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

from experiment.utils.object_detection import ops


class MinibatchSampler:
  """Abstract base class for subsampling minibatches."""

  __metaclass__ = ABCMeta

  def __init__(self):
    """Constructs a minibatch sampler."""

  @abstractmethod
  def subsample(self, indicator, batch_size, **params):
    """Subsample indicator vector.

    Given a boolean indicator vector with M elements set to `True`, the function
    assigns all but `num_samples` of these previously `True` elements to
    `False`. If `num_samples` is greater than M, the original indicator vector
    is returned.

    Parameters
    ----------
    indicator: tf.Tensor.
        A 1-dimensional boolean tensor indicating which elements are
        allowed to be sampled and which are not.
    batch_size: int
        Desired batch size.
    params: dict
        Additional keyword arguments for specific implementations of the
        MinibatchSampler.

    Returns
    -------
        Boolean tensor of shape [N] whose True entries have been sampled.
        If sum(indicator) >= batch_size, sum(is_sampled) = batch_size
    """

  @staticmethod
  def subsample_indicator(indicator, num_samples):
    """Subsample indicator vector.

    Given a boolean indicator vector with M elements set to `True`, the function
    assigns all but `num_samples` of these previously `True` elements to
    `False`. If `num_samples` is greater than M, the original indicator vector
    is returned.

    Parameters
    ----------
    indicator: tf.Tensor
        A 1-dimensional boolean tensor indicating which elements are
        allowed to be sampled and which are not.
    num_samples: int
        The number of samples.

    Returns
    -------
        A boolean tensor with the same shape as input (indicator) tensor
    """
    indices = tf.where(indicator)
    indices = tf.random.shuffle(indices)
    indices = tf.reshape(indices, [-1])

    num_samples = tf.minimum(tf.size(input=indices), num_samples)
    selected_indices = tf.slice(indices, [0], tf.reshape(num_samples, [1]))
    selected_indicator = ops.indices_to_dense_vector(
        selected_indices, tf.shape(input=indicator)[0])
    return tf.equal(selected_indicator, 1)
