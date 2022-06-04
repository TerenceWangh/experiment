"""Box matcher implementation."""

import tensorflow as tf


class BoxMatcher:
  """Matcher based on highest value.

  This class computes matches from a similarity matrix. Each column is matched
  to a single row.

  To support object detection target assignment this class enables setting both
  positive_threshold (upper threshold) and negative_threshold (lower thresholds)
  defining three categories of similarity which define whether examples are
  positive, negative, or ignored, for example:
  (1) thresholds=[negative_threshold, positive_threshold], and
      indicators=[negative_value, ignore_value, positive_value]: The similarity
      metrics below negative_threshold will be assigned with negative_value,
      the metrics between negative_threshold and positive_threshold will be
      assigned ignore_value, and the metrics above positive_threshold will be
      assigned positive_value.
  (2) thresholds=[negative_threshold, positive_threshold], and
      indicators=[ignore_value, negative_value, positive_value]: The similarity
      metric below negative_threshold will be assigned with ignore_value,
      the metrics between negative_threshold and positive_threshold will be
      assigned negative_value, and the metrics above positive_threshold will be
      assigned positive_value.
  """

  def __init__(self, thresholds, indicators, force_match_for_each_col=False):
    """Construct BoxMatcher.

    Parameters
    ----------
    thresholds : array_like
        Classify boxes into different buckets. The list needs to be sorted,
        and will be prepended with -Inf and appended with +Inf.
    indicators : array_like
        Assign for each bucket. len(`indicators` must equal to
        len(`threshold`) + 1.
    force_match_for_each_col : bool, default True
        Ensures that each column is matched to at least one row (which is not
        guaranteed otherwise if the positive_threshold is high). Defaults to
        False. If True, all force matched row will be assigned to
        `indicators[-1]`.

    Raises
    ------
    ValueError:
        If `threshold` not sorted, or len(indicators) != len(threshold) + 1
    """
    if not all([lo <= hi for lo, hi in zip(thresholds[:-1], thresholds[1:])]):
      raise ValueError('`threshold` must be sorted, got {}.'.format(thresholds))
    self._indicators = indicators
    if len(indicators) != len(thresholds) + 1:
      raise ValueError(
          'len(`indicators`) must be len(`thresholds`) + 1, got indicators {}, '
          'threshold {}.'.format(indicators, thresholds))

    thresholds = thresholds[:]
    thresholds.insert(0, -float('inf'))
    thresholds.append(float('inf'))
    self._thresholds = thresholds
    self._force_match_for_each_col = force_match_for_each_col

  def __call__(self, similarity_matrix):
    """Tries to match each column of the similarity matrix to a row.

    Parameters
    ----------
    similarity_matrix: tf.Tensor
        A float tensor of shape [N, M] representing any similarity metric.

    Returns
    -------
        A integer tensor of shape [N] with corresponding match indices for each
        of M columns, for positive match, the match result will be the
        corresponding row index, for negative match, the match will be
        `negative_value`, for ignored match, the match result will be
        `ignore_value`.
    """
    squeeze_result = False
    if len(similarity_matrix.shape) == 2:
      squeeze_result = True
      similarity_matrix = tf.expand_dims(similarity_matrix, axis=0)

    static_shape = similarity_matrix.shape.as_list()
    num_rows = static_shape[1] or tf.shape(similarity_matrix)[1]
    batch_size = static_shape[0] or tf.shape(similarity_matrix)[0]

    def _match_when_rows_are_empty():
      """Performs matching when the rows of similarity matrix are empty.

      When the rows are empty, all detections are false positives. So we return
      a tensor of -1's to indicate that the columns do not match to any rows.

      Returns
      -------
          Int32 tensor indicating the row each column matches to.
      """
      with tf.name_scope('empty_gt_boxes'):
        matches = tf.zeros([batch_size, num_rows], dtype=tf.int32)
        match_labels = tf.ones([batch_size, num_rows], dtype=tf.int32)
        return matches, match_labels

    def _match_when_rows_are_non_empty():
      """Performs matching when the rows of similarity matrix are non empty.

      Returns
      -------
          Int32 tensor indicating the row each column matches to.
      """
      # Matches for each column
      with tf.name_scope('non_empty_gt_boxes'):
        matches = tf.argmax(similarity_matrix, axis=-1, output_type=tf.int32)

        # Get logical indices of ignored and unmatched columns as tf.int64
        matched_values = tf.reduce_max(similarity_matrix, axis=-1)
        matched_indicators = tf.zeros([batch_size, num_rows], tf.int32)
        matched_type = matched_values.dtype

        for (index, lo, hi) in zip(
            self._indicators, self._thresholds[:-1], self._thresholds[1:]):
          lo_threshold = tf.cast(lo, matched_type)
          hi_threshold = tf.cast(hi, matched_type)
          mask = tf.logical_and(
              tf.greater_equal(matched_values, lo_threshold),
              tf.less(matched_values, hi_threshold))
          matched_indicators = self._set_values_using_indicator(
              matched_indicators, mask, index)

        if self._force_match_for_each_col:
          # [batch_size, M], for each col (gt_box), find the best matching
          # row (anchor).
          force_match_column_ids = tf.argmax(
              input=similarity_matrix, axis=1, output_type=tf.int32)
          # [batch_size, M, N]
          force_match_column_indicators = tf.one_hot(
              force_match_column_ids, depth=num_rows)
          # [batch_size, N], for each row (anchor), find the largest column
          # index for ground truth box
          force_match_column_ids = tf.argmax(
              input=force_match_column_indicators, axis=1, output_type=tf.int32)
          # [batch_size, N]
          force_match_column_mask = tf.cast(
              tf.reduce_max(force_match_column_indicators, axis=1), tf.bool)
          # [batch_size, N]
          final_matches = tf.where(force_match_column_mask,
                                   force_match_column_ids,
                                   matches)
          final_matches_indicators = tf.where(
              force_match_column_mask,
              self._indicators[-1] * tf.ones([batch_size, num_rows],
                                             dtype=tf.int32),
              matched_indicators)
          return final_matches, final_matches_indicators
        else:
          return matches, matched_indicators

    num_gt_boxes = similarity_matrix.shape.as_list()[-1] or \
                  tf.shape(similarity_matrix)[-1]
    result_match, result_matched_indicators = tf.cond(
        pred=tf.greater(num_gt_boxes, 0),
        true_fn=_match_when_rows_are_non_empty,
        false_fn=_match_when_rows_are_empty)

    if squeeze_result:
      result_match = tf.squeeze(result_match, axis=0)
      result_matched_indicators = tf.squeeze(result_matched_indicators, axis=0)

    return result_match, result_matched_indicators

  def _set_values_using_indicator(self, x, indicator, value):
    """Set the indicated fields of x to value.

    Parameters
    ----------
    x: tf.Tensor.
    indicator: bool
      Same shape as x.
    value: scalar.
        The value to set.

    Returns
    -------
        modified tensor.
    """
    indicator = tf.cast(indicator, x.dtype)
    return tf.add(tf.multiply(x, 1 - indicator), value * indicator)
