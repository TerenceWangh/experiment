"""Region Similarity Calculators for BoxLists.

Region Similarity Calculators compare a pairwise measure of similarity
between the boxes in two BoxLists.
"""
from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf


def area(box_list, scope = None):
  """Computes area of boxes.

  Parameters
  ----------
  box_list: BoxList
      BoxList holding N boxes
  scope: str, optional
      Name scope of the function.

  Returns
  -------
  tf.Tensor
      A tensor with shape [N] representing box areas.
  """
  with tf.name_scope(scope or 'Area'):
    ymin, xmin, ymax, xmax = tf.split(
        value=box_list.get(), num_or_size_splits=4, axis=1)
    return tf.squeeze((ymax - ymin) * (xmax - xmin), [1])


def intersection(box_list1, box_list2, scope = None):
  """Compute pairwise intersection areas between boxes.

  Parameters
  ----------
  box_list1: BoxList
      BoxList holding N boxes.
  box_list2: BoxList
      BoxList holding N boxes.
  scope: str, optional
      Name scope of the function.

  Returns
  -------
  tf.Tensor
      A tensor with shape [N, M] representing pairwise intersections.
  """
  with tf.name_scope(scope or 'Intersection'):
    ymin1, xmin1, ymax1, xmax1 = tf.split(
        value=box_list1.get(), num_or_size_splits=4, axis=1)
    ymin2, xmin2, ymax2, xmax2 = tf.split(
        value=box_list2.get(), num_or_size_splits=4, axis=1)
    all_pairs_min_ymax = tf.minimum(ymax1, tf.transpose(a=ymax2))
    all_pairs_max_ymin = tf.maximum(ymin1, tf.transpose(a=ymin2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(xmax1, tf.transpose(a=xmax2))
    all_pairs_max_xmin = tf.maximum(xmin1, tf.transpose(a=xmin2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths

def iou(box_list1, box_list2, scope = None):
  """Computes pairwise intersection-over-union between box collections.

  Parameters
  ----------
  box_list1: BoxList
      BoxList holding N boxes.
  box_list2: BoxList
      BoxList holding N boxes.
  scope: str, optional
      Name scope of the function.

  Returns
  -------
  tf.Tensor:
      A tensor with shape [N, M] representing pairwise iou scores.
  """
  with tf.name_scope(scope or 'IoU'):
    intersections = intersection(box_list1, box_list2)
    areas1 = area(box_list1)
    areas2 = area(box_list2)
    unions = (
        tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0), tf.zeros_like(intersections),
        tf.truediv(intersections, unions))


class RegionSimilarityCalculator:
  """Abstract base class for region similarity calculator."""
  __metaclass__ = ABCMeta

  def compare(self, box_list1, box_list2, scope = None):
    """Computes matrix of pairwise similarity between BoxLists.

    This op (to be overridden) computes a measure of pairwise similarity between
    the boxes in the given BoxLists. Higher values indicate more similarity.
    Note that this method simply measures similarity and does not explicitly
    perform a matching.

    Parameters
    ----------
    box_list1: BoxList
        BoxList holding N boxes.
    box_list2: BoxList
        BoxList holding N boxes.
    scope: str, optional
        Name scope of the function.

    Returns
    -------
    tf.Tensor
        A (float32) tensor of shape [N, M] with pairwise similarity score.
    """
    with tf.name_scope(scope or 'RegionSimilarity'):
      return self._compare(box_list1, box_list2)

  @abstractmethod
  def _compare(self, box_list1, box_list2):
    pass


class IouSimilarity(RegionSimilarityCalculator):
  """Class to compute similarity based on Intersection over Union (IOU) metric.

  This class computes pairwise similarity between two BoxLists based on IOU.
  """

  def _compare(self, box_list1, box_list2):
    return iou(box_list1, box_list2)
