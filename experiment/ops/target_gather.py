"""Definition of target gather, which gathers targets from indices."""

import tensorflow as tf


class TargetGather:
  """Target gather for dense object detector."""

  def __call__(self, labels, match_indices, mask=None, mask_value=0.0):
    """Labels anchors with ground truth inputs.

    B: batch_size

    N: number of ground truth boxes.

    Parameters
    ----------
    labels : tf.Tensor
        An integer tensor with shape [N, dims] or [B, N, ...].
        The ground truth labels.
    match_indices : tf.Tensor
        An integer tensor with shape [M] or [B, M].
        The matched label index.
    mask : tf.Tensor
        An boolean tensor with shape [M, dims] or [B, M,...], optional.
        The matched labels.
    mask_value : float, default 0.0
        The value to fill in for mask.

    Returns
    -------
    tf.Tensor
      An integer Tensor with shape [M] or [B, M]

    Raises
    ------
    ValueError
        If `labels` is higher than rank 3.
    """
    if len(labels.shape) <= 2:
      return self._gather_unbatched(labels, match_indices, mask, mask_value)
    elif len(labels.shape) == 3:
      return self._gather_batched(labels, match_indices, mask, mask_value)
    else:
      raise ValueError('`TargetGather` does not support `labels` with rank '
                       'larger than 3, got {}.'.format(len(labels.shape)))

  def _gather_unbatched(self, labels, match_indices, mask, mask_value):
    """Gather based on unbatched labels and boxes."""
    num_gt_boxes = tf.shape(labels)[0]

    def _assign_when_rows_empty():
      if len(labels.shape) > 1:
        mask_shape = [match_indices.shape[0], labels.shape[-1]]
      else:
        mask_shape = [match_indices.shape[0]]
      return tf.cast(mask_value, labels.dtype) * tf.ones(
          mask_shape, dtype=labels.dtype)

    def _assign_when_rows_not_empty():
      targets = tf.gather(labels, match_indices)
      if mask is None:
        return targets
      else:
        masked_targets = tf.cast(mask_value, labels.dtype) * tf.ones_like(
            mask, dtype=labels.dtype)
        return tf.where(mask, masked_targets, targets)

    return tf.cond(tf.greater(num_gt_boxes, 0),
                   _assign_when_rows_not_empty,
                   _assign_when_rows_empty)

  def _gather_batched(self, labels, match_indices, mask, mask_value):
    """Gather based on batched labels."""
    batch_size = labels.shape[0]
    if batch_size == 1:
      if mask is not None:
        result = self._gather_unbatched(
            tf.squeeze(labels, axis=0),
            tf.squeeze(match_indices, axis=0),
            tf.squeeze(mask, axis=0),
            mask_value)
      else:
        result = self._gather_unbatched(
            tf.squeeze(labels, axis=0),
            tf.squeeze(match_indices, axis=0),
            None,
            mask_value)
      return tf.expand_dims(result, axis=0)
    else:
      indices_shape = tf.shape(match_indices)
      indices_dtype = match_indices.dtype
      batch_indices = tf.expand_dims(
          tf.range(
              indices_shape[0], dtype=indices_dtype), axis=-1) * tf.ones(
                  [1, indices_shape[-1]], dtype=indices_dtype)
      gather_nd_indices = tf.stack(
          [batch_indices, match_indices], axis=-1)
      targets = tf.gather_nd(labels, gather_nd_indices)
      if mask is None:
        return targets
      else:
        masked_targets = tf.cast(mask_value, labels.dtype) * tf.ones_like(
            mask, dtype=labels.dtype)
        return tf.where(mask, masked_targets, targets)
