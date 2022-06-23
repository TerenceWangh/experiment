"""Contains definitions of RoI generator."""

from typing import Optional, Mapping

import tensorflow as tf

from experiment.ops import box_ops, nms


def _multi_level_props_rois(raw_boxes: Mapping[str, tf.Tensor],
                            raw_scores: Mapping[str, tf.Tensor],
                            anchor_boxes: Mapping[str, tf.Tensor],
                            image_shape: tf.Tensor,
                            pre_nms_top_k: int = 2000,
                            pre_nms_score_threshold: float = 0.0,
                            pre_nms_min_size_threshold: float = 0.0,
                            nms_iou_threshold: float = 0.7,
                            num_proposals: int = 1000,
                            use_batched_nms: bool = False,
                            decode_boxes: bool = True,
                            clip_boxes: bool = True,
                            apply_sigmoid_to_scores: bool = True):
  """Proposes RoIs given a group of candidates from different FPN levels.

  The following describes the steps:
    1. For each individual level:
      a. Apply sigmoid transform if specified.
      b. Decode boxes if specified.
      c. Clip boxes if specified.
      d. Filter small boxes and those fall outside image if specified.
      e. Apply pre-NMS filtering including pre-NMS top k and score thresholding.
      f. Apply NMS.
    2. Aggregate post-NMS boxes from each level.
    3. Apply an overall top k to generate the final selected RoIs.

  Parameters
  ----------
  raw_boxes : dict
      FPN levels and values representing box tenors of shape
      [batch_size, feature_h, feature_w, num_anchors * 4].
  raw_scores : dict
      FPN levels and values representing logits tensors of shape
      [batch_size, feature_h, feature_w, num_anchors].
  anchor_boxes : dict
      FPN levels and values representing anchor box tensors of shape
      [batch_size, feature_h * feature_w * num_anchors, 4].
  image_shape : tf.Tensor
      A `tf.Tensor` of shape [batch_size, 2] where the last dimension
      are [height, width] of the scaled image.
  pre_nms_top_k : int, default 2000
      The top scoring RPN proposals *per level* to keep before applying NMS.
  pre_nms_score_threshold : float, default 0.0
      The value between 0 and 1 representing the minimal box score to keep
      before applying NMS. This is often used as a pre-filtering step for
      better performance.
  pre_nms_min_size_threshold : float, default 0.0
      The value representing the minimal box size in each side (w.r.t. the
      scaled image) to keep before applying NMS. This is often used as a
      pre-filtering step for better performance.
  nms_iou_threshold : float, default 0.7
      The value between 0 and 1 representing the IoU threshold used for NMS.
      If 0.0, no NMS is applied.
  num_proposals : int, default 1000
      The top scoring RPN proposals *in total* to keep after applying NMS.
  use_batched_nms : bool, default False
      Whether NMS is applied in batch using
      `tf.image.combined_non_max_suppression`.
  decode_boxes : bool, default True
      Whether `raw_boxes` needs to be decoded using `anchor_boxes`. If False,
      use `raw_boxes` directly and ignore `anchor_boxes`.
  clip_boxes : bool, default True
      Whether boxes are first clipped to the scaled image size before applying
      NMS. If False, no clipping is applied and `image_shape` is ignored.
  apply_sigmoid_to_scores : bool, default True
      Whether apply sigmoid to `raw_scores` before applying NMS.

  Returns
  -------
  tf.Tensor
      The box coordinates of the selected proposals w.r.t. the scaled image with
      the shape of [batch_size, num_proposals, 4].
  tf.Tensor
      The scores of the selected proposals with the shape of
      [batch_size, num_proposals, 1].
  """
  with tf.name_scope('multi_level_propose_rois'):
    rois, roi_scores = [], []
    image_shape = tf.expand_dims(image_shape, axis=1)
    for level in sorted(raw_scores.keys()):
      with tf.name_scope('level_{}'.format(level)):
        _, fh, fw, n_anchors = raw_scores[level].get_shape().as_list()
        num_boxes = fh * fw * n_anchors
        this_level_scores = tf.reshape(raw_scores[level], [-1, num_boxes])
        this_level_boxes = tf.reshape(raw_boxes[level], [-1, num_boxes, 4])
        this_level_anchors = tf.cast(tf.reshape(anchor_boxes[level],
                                                [-1, num_boxes, 4]),
                                     dtype=this_level_scores.dtype)

        if apply_sigmoid_to_scores:
          this_level_scores = tf.sigmoid(this_level_scores)

        if decode_boxes:
          this_level_boxes = box_ops.decode_boxes(
              this_level_boxes, this_level_anchors)

        if clip_boxes:
          this_level_boxes = box_ops.clip_boxes(
              this_level_boxes, image_shape)

        if pre_nms_min_size_threshold > 0.0:
          this_level_boxes, this_level_scores = box_ops.filter_boxes(
              this_level_boxes,
              this_level_scores,
              image_shape,
              pre_nms_min_size_threshold)

        this_level_pre_nms_top_k = min(num_boxes, pre_nms_top_k)
        this_level_post_nms_top_k = min(num_boxes, num_proposals)
        if nms_iou_threshold > 0.0:
          if use_batched_nms:
            this_level_rois, this_level_roi_scores, _, _ = (
                tf.image.combined_non_max_suppression(
                    tf.expand_dims(this_level_boxes, axis=2),
                    tf.expand_dims(this_level_scores, axis=-1),
                    max_output_size_per_class=this_level_pre_nms_top_k,
                    max_total_size=this_level_post_nms_top_k,
                    iou_threshold=nms_iou_threshold,
                    score_threshold=pre_nms_score_threshold,
                    pad_per_class=False,
                    clip_boxes=False))
          else:
            if pre_nms_score_threshold > 0.0:
              this_level_boxes, this_level_scores = (
                  box_ops.filter_boxes_by_scores(
                      this_level_boxes,
                      this_level_scores,
                      pre_nms_score_threshold
                  ))
            this_level_boxes, this_level_scores = box_ops.top_k_boxes(
                this_level_boxes, this_level_scores, k=this_level_pre_nms_top_k)
            this_level_roi_scores, this_level_rois = (
                nms.sorted_non_max_suppression_padded(
                    this_level_scores,
                    this_level_boxes,
                    max_output_size=this_level_post_nms_top_k,
                    iou_threshold=nms_iou_threshold))
        else:
          this_level_rois, this_level_roi_scores = box_ops.top_k_boxes(
              this_level_boxes, this_level_scores,
              k=this_level_post_nms_top_k)

        rois.append(this_level_rois)
        roi_scores.append(this_level_roi_scores)

    all_rois = tf.concat(rois, axis=1)
    all_roi_scores = tf.concat(roi_scores, axis=1)

    with tf.name_scope('top_k_rois'):
      _, num_valid_rois = all_roi_scores.get_shape().as_list()
      overall_top_k = min(num_valid_rois, num_proposals)

      selected_rois, selected_roi_scores = box_ops.top_k_boxes(
          all_rois, all_roi_scores, k=overall_top_k)

    return selected_rois, selected_roi_scores


@tf.keras.utils.register_keras_serializable(package='experiment')
class MultiLevelRoIGenerator(tf.keras.layers.Layer):
  """Proposes RoIs for the second stage processing."""

  def __init__(self,
               pre_nms_top_k: int = 2000,
               pre_nms_score_threshold: float = 0.0,
               pre_nms_min_size_threshold: float = 0.0,
               nms_iou_threshold: float = 0.7,
               num_proposals: int = 1000,
               test_pre_nms_top_k: int = 1000,
               test_pre_nms_score_threshold: float = 0.0,
               test_pre_nms_min_size_threshold: float = 0.0,
               test_nms_iou_threshold: float = 0.7,
               test_num_proposals: int = 1000,
               use_batched_nms: bool = False,
               **kwargs):
    """Initializes a ROI generator.

    The ROI generator transforms the raw predictions from RPN to ROIs.

    Parameters
    ----------
    pre_nms_top_k : int, default 2000
        The number of top scores proposals to be kept before applying NMS.
    pre_nms_score_threshold : float, default 0.0
        The score threshold to apply before applying NMS. Proposals whose scores
        are below this threshold are thrown away.
    pre_nms_min_size_threshold : float, default 0.0
        The threshold of each side of the box (w.r.t. the scaled image).
        Proposals whose sides are below this threshold are thrown away.
    nms_iou_threshold : float, default 0.7
        The NMS IoU threshold in [0, 1].
    num_proposals : int, default 1000
        The final number of proposals to generate.
    test_pre_nms_top_k : int, default 1000
        The number of top scores proposals to be kept before applying NMS in
        testing.
    test_pre_nms_score_threshold : float, default 0.0
        The score threshold to apply before applying NMS in testing. Proposals
        whose scores are below this threshold are thrown away.
    test_pre_nms_min_size_threshold : float, default 0.0
        The threshold of each side of the box (w.r.t. the scaled image) in
        testing. Proposals whose sides are below this threshold are thrown away.
    test_nms_iou_threshold : float, default 0.7
        The NMS IoU threshold with range of [0, 1] in testing.
    test_num_proposals : int, default 1000
        The final number of proposals to generate in testing.
    use_batched_nms : bool, default False
        Whether or not use `tf.image.combined_non_max_suppression`.
    kwargs : dict
        Additional keyword arguments passed to Layer.
    """
    self._pre_nms_top_k = pre_nms_top_k
    self._pre_nms_score_threshold = pre_nms_score_threshold
    self._pre_nms_min_size_threshold = pre_nms_min_size_threshold
    self._nms_iou_threshold = nms_iou_threshold
    self._num_proposals = num_proposals
    self._test_pre_nms_top_k = test_pre_nms_top_k
    self._test_pre_nms_score_threshold = test_pre_nms_score_threshold
    self._test_pre_nms_min_size_threshold = test_pre_nms_min_size_threshold
    self._test_nms_iou_threshold = test_nms_iou_threshold
    self._test_num_proposals = test_num_proposals
    self._use_batched_nms = use_batched_nms
    super(MultiLevelRoIGenerator, self).__init__(**kwargs)

  def call(self,
           raw_boxes: Mapping[str, tf.Tensor],
           raw_scores: Mapping[str, tf.Tensor],
           anchor_boxes: Mapping[str, tf.Tensor],
           image_shape: tf.Tensor,
           training: Optional[bool] = None):
    """Proposes RoIs given a group of candidates from different FPN levels.

    The following describes the steps:
      1. For each individual level:
        a. Apply sigmoid transform if specified.
        b. Decode boxes if specified.
        c. Clip boxes if specified.
        d. Filter small boxes and those fall outside image if specified.
        e. Apply pre-NMS filtering including pre-NMS top k and score
           thresholding.
        f. Apply NMS.
      2. Aggregate post-NMS boxes from each level.
      3. Apply an overall top k to generate the final selected RoIs.

    Parameters
    ----------
    raw_boxes : dict
        FPN levels with box tensors of shape
        [batch, feature_h, feature_w, num_anchors * 4].
    raw_scores : dict
        FPN levels with logit tensors of shape
        [batch, feature_h, feature_w, num_anchors].
    anchor_boxes : dict
        FPN levels with anchor box tensors of shape
        [batch, feature_h * feature_w * num_anchors, 4].
    image_shape : tf.Tensor
        A `tf.Tensor` of shape [batch, 2] where the last dimension
        are [height, width] of the scaled image.
    training : bool, optional
        Whether it is in training mode.

    Returns
    -------
    tf.Tensor
        The proposed ROIs in The scaled image coordinate with shape
        [batch, num_proposals, 4].
    tf.Tensor
        The cores of the proposed ROIs with shape [batch, num_proposals].
    """
    roi_boxes, roi_scores = _multi_level_props_rois(
        raw_boxes, raw_scores, anchor_boxes, image_shape,
        pre_nms_top_k=self._pre_nms_top_k if training \
            else self._test_pre_nms_top_k,
        pre_nms_score_threshold=self._pre_nms_score_threshold if training \
            else self._test_pre_nms_score_threshold,
        pre_nms_min_size_threshold=
            self._pre_nms_min_size_threshold if training \
            else self._test_pre_nms_min_size_threshold,
        nms_iou_threshold=self._nms_iou_threshold if training \
            else self._test_nms_iou_threshold,
        num_proposals=self._num_proposals if training \
            else self._test_num_proposals,
        use_batched_nms=self._use_batched_nms,
        decode_boxes=True,
        clip_boxes=True,
        apply_sigmoid_to_scores=True)
    return roi_boxes, roi_scores
  
  def get_config(self):
    config = super(MultiLevelRoIGenerator, self).get_config()
    config.update({
        'pre_nms_top_k': self._pre_nms_top_k,
        'pre_nms_score_threshold': self._pre_nms_score_threshold,
        'pre_nms_min_size_threshold': self._pre_nms_min_size_threshold,
        'nms_iou_threshold': self._nms_iou_threshold,
        'num_proposals': self._num_proposals,
        'test_pre_nms_top_k': self._test_pre_nms_top_k,
        'test_pre_nms_score_threshold': self._test_pre_nms_score_threshold,
        'test_pre_nms_min_size_threshold':
            self._test_pre_nms_min_size_threshold,
        'test_nms_iou_threshold': self._test_nms_iou_threshold,
        'test_num_proposals': self._test_num_proposals,
        'use_batched_nms': self._use_batched_nms,
    })
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
