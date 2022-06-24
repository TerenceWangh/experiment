"""Contains definitions of RoI sampler."""

import tensorflow as tf

from experiment.model.layers import box_sampler
from experiment.ops import box_matcher
from experiment.ops import iou_similarity
from experiment.ops import target_gather


@tf.keras.utils.register_keras_serializable(package='experiment')
class RoISampler(tf.keras.layers.Layer):
  """Samples RoIs and assigns targets to the sampled RoIs."""

  def __init__(self,
               mix_gt_boxes: bool = True,
               num_sampled_rois: int = 512,
               foreground_fraction: float = 0.25,
               foreground_iou_threshold: float = 0.5,
               background_iou_high_threshold: float = 0.5,
               background_iou_low_threshold: float = 0,
               skip_subsampling: bool = False,
               **kwargs):
    """Initializes a RoI sampler.

    Parameters
    ----------
    mix_gt_boxes : bool, default True
        Whether to mix the groundtruth boxes with proposed ROIs.
    num_sampled_rois : int, default 512
        The number of sampled ROIs per image.
    foreground_fraction : float, default 0.25
        What percentage of proposed ROIs should be sampled from the foreground
        boxes.
    foreground_iou_threshold : float, default 0.5
        The IoU threshold for a box to be considered as positive
        (if >= `foreground_iou_threshold`).
    background_iou_high_threshold : float, default 0.5
        The IoU threshold for a box to be considered as negative (if overlap in
        [`background_iou_low_threshold`, `background_iou_high_threshold`]).
    background_iou_low_threshold : float, default 0
        The IoU threshold for a box to be considered as negative (if overlap in
        [`background_iou_low_threshold`, `background_iou_high_threshold`])
    skip_subsampling : bool, default False
        Whether we want to skip the sampling procedure than balances the fg/bg
        classes. Used for upper frcnn layers in cascade RCNN.
    kwargs : dict
        Additional keyword arguments passed to Layer.
    """
    self._mix_gt_boxes = mix_gt_boxes
    self._num_sampled_rois = num_sampled_rois
    self._foreground_fraction = foreground_fraction
    self._foreground_iou_threshold = foreground_iou_threshold
    self._background_iou_high_threshold = background_iou_high_threshold
    self._background_iou_low_threshold = background_iou_low_threshold
    self._skip_subsampling = skip_subsampling

    self._sim_calc = iou_similarity.IoUSimilarity()
    self._box_matcher = box_matcher.BoxMatcher(
        thresholds=[
            background_iou_low_threshold,
            background_iou_high_threshold,
            foreground_iou_threshold
        ],
        indicators=[-3, -1, -2, 1])
    self._target_gather = target_gather.TargetGather()
    self._sampler = box_sampler.BoxSampler(
        num_sampled_rois, foreground_fraction)
    super(RoISampler, self).__init__(**kwargs)

  def call(self, boxes: tf.Tensor, gt_boxes: tf.Tensor, gt_classes: tf.Tensor):
    """Assigns the proposals with groundtruth classes and performs subsmpling.

    Given `proposed_boxes`, `gt_boxes`, and `gt_classes`, the function uses the
    following algorithm to generate the final `num_samples_per_image` RoIs.
      1. Calculates the IoU between each proposal box and each gt_boxes.
      2. Assigns each proposed box with a groundtruth class and box by choosing
         the largest IoU overlap.
      3. Samples `num_samples_per_image` boxes from all proposed boxes, and
         returns box_targets, class_targets, and RoIs.

    Parameters
    ----------
    boxes : tf.Tensor
        A `tf.Tensor` of shape of [batch_size, N, 4]. N is the number of
        proposals before groundtruth assignment. The last dimension is the
        box coordinates w.r.t. the scaled images in [ymin, xmin, ymax, xmax]
        format.
    gt_boxes : tf.Tensor
        A `tf.Tensor` of shape of [batch_size, MAX_NUM_INSTANCES, 4].
        The coordinates of gt_boxes are in the pixel coordinates of the scaled
        image. This tensor might have padding of values -1 indicating the
        invalid box coordinates.
    gt_classes : tf.Tensor
        A `tf.Tensor` with a shape of [batch_size, MAX_NUM_INSTANCES].
        This tensor might have paddings with values of -1 indicating the invalid
        classes.

    Returns
    -------
    tf.Tensor
        A `tf.Tensor` of shape of [batch_size, K, 4], representing
        the coordinates of the sampled RoIs, where K is the number of the
        sampled RoIs, i.e. K = num_samples_per_image.
    tf.Tensor
        A `tf.Tensor` of shape of [batch_size, K, 4], storing the box
        coordinates of the matched groundtruth boxes of the samples RoIs.
    tf.Tensor
        A `tf.Tensor` of shape of [batch_size, K], storing the
        classes of the matched groundtruth boxes of the sampled RoIs.
    tf.Tensor
        A `tf.Tensor` of shape of [batch_size, K], storing the indices of the
        sampled groudntruth boxes in the original `gt_boxes` tensor, i.e.,
        gt_boxes[sampled_gt_indices[:, i]] = sampled_gt_boxes[:, i].
    """
    gt_boxes = tf.cast(gt_boxes, dtype=boxes.dtype)
    if self._mix_gt_boxes:
      boxes = tf.concat([boxes, gt_boxes], axis=1)

    boxes_invalid_mask = tf.less(
        tf.reduce_max(boxes, axis=-1, keepdims=True), 0.0)
    gt_invalid_mask = tf.less(
        tf.reduce_max(gt_boxes, axis=-1, keepdims=True), 0.0)
    similarity_matrix = self._sim_calc(
        boxes, gt_boxes, boxes_invalid_mask, gt_invalid_mask)
    matched_gt_indices, match_indicators = self._box_matcher(similarity_matrix)
    positive_matches = tf.greater_equal(match_indicators, 0)
    negative_matches = tf.equal(match_indicators, -1)
    ignored_matches = tf.equal(match_indicators, -2)
    invalid_matches = tf.equal(match_indicators, -3)

    background_mask = tf.expand_dims(
        tf.logical_or(negative_matches, invalid_matches), -1)
    gt_classes = tf.expand_dims(gt_classes, axis=-1)
    matched_gt_classes = self._target_gather(
        gt_classes, matched_gt_indices, background_mask)
    matched_gt_classes = tf.where(
        background_mask, tf.zeros_like(matched_gt_classes), matched_gt_classes)
    matched_gt_boxes = self._target_gather(
        gt_boxes, matched_gt_indices, tf.tile(background_mask, [1, 1, 4]))
    matched_gt_boxes = tf.where(
        background_mask, tf.zeros_like(matched_gt_boxes), matched_gt_boxes)
    matched_gt_indices = tf.where(
        tf.squeeze(background_mask, -1),
        tf.multiply(tf.ones_like(matched_gt_indices), -1),
        matched_gt_indices)

    if self._skip_subsampling:
      matched_gt_classes = tf.squeeze(matched_gt_classes, axis=-1)
      return boxes, matched_gt_boxes, matched_gt_classes, matched_gt_indices

    indices = self._sampler(
        positive_matches, negative_matches, ignored_matches)
    rois = self._target_gather(boxes, indices)
    gt_boxes = self._target_gather(matched_gt_boxes, indices)
    gt_classes = tf.squeeze(self._target_gather(
        matched_gt_classes, indices), axis=-1)
    gt_indices = tf.squeeze(self._target_gather(
        tf.expand_dims(matched_gt_indices, -1), indices), axis=-1)
    return rois, gt_boxes, gt_classes, gt_indices

  def get_config(self):
    config = super(RoISampler, self).get_config()
    config.update({
        'mix_gt_boxes': self._mix_gt_boxes,
        'num_sampled_rois': self._num_sampled_rois,
        'foreground_fraction': self._foreground_fraction,
        'foreground_iou_threshold': self._foreground_iou_threshold,
        'background_iou_high_threshold': self._background_iou_high_threshold,
        'background_iou_low_threshold': self._background_iou_low_threshold,
        'skip_subsampling': self._skip_subsampling,
    })
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)
