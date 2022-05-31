"""Region Similarity Calculators."""

import tensorflow as tf


def area(box):
  """Computes area of boxes.

  Parameters
  ----------
  box: a float Tensor with [num_of_boxes, 4], or [batch_size, num_of_boxes, 4].
      The input box to calculate area.

  Returns
  -------
      a float Tensor with [num_of_boxes], or [batch_size, num_of_boxes]
  """
  with tf.name_scope('Area'):
    y_min, x_min, y_max, x_max = tf.split(value=box, num_or_size_splits=4,
                                          axis=-1)
  return tf.squeeze((y_max - y_min) * (x_max - x_min), axis=-1)


def intersection(gt_boxes, boxes):
  """Compute pairwise intersection areas between boxes.

  Notes
  -----
  B: batch_size

  N: number of ground truth boxes.

  M: number of anchor boxes.

  Parameters
  ----------
  gt_boxes: a float Tensor with [N, 4], or [B, N, 4]
      ground truth boxes.
  boxes: a float Tensor with [M, 4], or [B, M, 4]
      the input boxes.

  Returns
  -------
      a float Tensor with shape [N, M] or [B, N, M] representing pairwise
      intersections.
  """
  with tf.name_scope('Intersection'):
    y_min1, x_min1, y_max1, x_max1 = tf.split(
      value=gt_boxes, num_or_size_splits=4, axis=-1)
    y_min2, x_min2, y_max2, x_max2 = tf.split(
      value=boxes, num_or_size_splits=4, axis=-1)

    boxes_rank = len(boxes.shape)
    perm = [1, 0] if boxes_rank == 2 else [0, 2, 1]
    # [N, M] or [B, N, M]
    y_min_max = tf.minimum(y_max1, tf.transpose(y_max2, perm))
    y_max_min = tf.maximum(y_min1, tf.transpose(y_min2, perm))
    x_min_max = tf.minimum(x_max1, tf.transpose(x_max2, perm))
    x_max_min = tf.maximum(x_min1, tf.transpose(x_min2, perm))

    intersect_heights = y_min_max - y_max_min
    intersect_widths = x_min_max - x_max_min
    zeros_t = tf.cast(0, intersect_heights.dtype)
    intersect_heights = tf.maximum(zeros_t, intersect_heights)
    intersect_widths = tf.maximum(zeros_t, intersect_widths)
    return intersect_heights * intersect_widths


def iou(gt_boxes, boxes):
  """Computes pairwise intersection-over-union between box collections.

  Notes
  -----
  B: batch_size

  N: number of ground truth boxes.

  M: number of anchor boxes.

  Parameters
  ----------
  gt_boxes: a float Tensor with [N, 4], or [B, N, 4]
      ground truth boxes.
  boxes: a float Tensor with [M, 4], or [B, M, 4]
      the input boxes.

  Returns
  -------
      a Tensor with shape [N, M] representing pairwise iou scores.
  """
  with tf.name_scope('IOU'):
    intersections = intersection(gt_boxes, boxes)
    gt_boxes_areas = area(gt_boxes)
    boxes_areas = area(boxes)
    boxes_rank = len(boxes_areas.shape)
    boxes_axis = 1 if (boxes_rank == 2) else 0
    gt_boxes_areas = tf.expand_dims(gt_boxes_areas, -1)
    boxes_areas = tf.expand_dims(boxes_areas, boxes_axis)
    unions = gt_boxes_areas + boxes_areas
    unions = unions - intersections
    return tf.where(
      tf.equal(intersections, 0.0), tf.zeros_like(intersections),
      tf.truediv(intersections, unions))


class IoUSimilarity:
  """Class to compute similarity based on Intersection over Union (IOU) metric.
  """

  def __init__(self, mask_value = -1):
    """Constructor IoUSimilarity.

    Parameters
    ----------
    mask_value: int, default -1.
        Value to be set.
    """
    self._mask_value = mask_value


  def __call__(self, boxes1, boxes2, boxes1_masks = None, boxes2_masks = None):
    """Compute pairwise IOU similarity between ground truth boxes and anchors.

    B: batch_size.

    N: Number of ground truth boxes.

    M: Number of anchor boxes.

    Parameters
    ----------
    boxes1: a float Tensor with M or B * M boxes.
        The ground truth boxes.
    boxes2: a float Tensor with N or B * N boxes
        The rank must be less than or equal to rank of `boxes_1`.
    boxes1_masks: a boolean Tensor with M or B * M boxes, optional.
    boxes2_masks: a boolean Tensor with N or B * N boxes. optional.

    Returns
    -------
        A Tensor with shape [M, N] or [B, M, N] representing pairwise iou
        scores, anchor per row and ground_truth_box per column.
    """
    boxes1 = tf.cast(boxes1, tf.float32)
    boxes2 = tf.cast(boxes2, tf.float32)

    boxes1_rank = len(boxes1.shape)
    boxes2_rank = len(boxes2.shape)

    if boxes1_rank < 2 or boxes1_rank > 3:
      raise ValueError(
        '`ground_truth_boxes` must be rank 2 or 3, got {}'.format(boxes1_rank))
    if boxes2_rank < 2 or boxes2_rank > 3:
      raise ValueError(
        '`anchors` must be rank 2 or 3, got {}'.format(boxes2_rank))
    if boxes1_rank < boxes2_rank:
      raise ValueError(
        '`ground_truth_boxes` is unbatched while `anchors` is batched is not a '
        'valid use case, got ground_truth_box rank {}, and anchors rank '
        '{}'.format(boxes1_rank, boxes2_rank))

    result = iou(boxes1, boxes2)
    if boxes1_masks is None and boxes2_masks is None:
      return result

    background_mask = None
    mask_value_t = tf.cast(self._mask_value, result.dtype) * \
                   tf.ones_like(result)
    perm = [1, 0] if boxes2_rank == 2 else [0, 2, 1]
    if boxes1_masks is not None and boxes2_masks is not None:
      background_mask = tf.logical_or(boxes1_masks,
                                      tf.transpose(boxes2_masks, perm))
    elif boxes1_masks is not None:
      background_mask = boxes1_masks
    else:
      background_mask = tf.logical_or(
          tf.zeros(tf.shape(boxes2)[:-1], dtype=tf.bool),
          tf.transpose(boxes2_masks, perm))
    return tf.where(background_mask, mask_value_t, result)
