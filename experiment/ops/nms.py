"""Tensorflow implementation of non max suppression."""
import tensorflow as tf

from experiment.ops import box_ops

NMS_TILE_SIZE = 512


def _self_suppression(iou, _, iou_sum):
  batch_size = tf.shape(iou)[0]
  can_suppress_others = tf.cast(
      tf.reshape(tf.reduce_max(iou, 1) <= 0.5, [batch_size, -1, 1]), iou.dtype)
  iou_suppressed = tf.reshape(
      tf.cast(tf.reduce_max(can_suppress_others * iou, 1) <= 0.5, iou.dtype),
      [batch_size, -1, 1]) * iou
  iou_sum_new = tf.reduce_sum(iou_suppressed, [1, 2])
  return [
      iou_suppressed,
      tf.reduce_any(iou_sum - iou_sum_new > 0.5),
      iou_sum_new
  ]


def _cross_suppression(boxes, box_slice, iou_threshold, inner_idx):
  batch_size = tf.shape(boxes)[0]
  new_slice = tf.slice(boxes, [0, inner_idx * NMS_TILE_SIZE, 0],
                       [batch_size, NMS_TILE_SIZE, 4])
  iou = box_ops.bbox_overlap(new_slice, box_slice)
  ret_slice = tf.expand_dims(
      tf.cast(tf.reduce_all(iou < iou_threshold, [1]), box_slice.dtype),
      2) * box_slice
  return boxes, ret_slice, iou_threshold, inner_idx + 1


def _suppression_loop_body(boxes, iou_threshold, output_size, idx):
  """Process boxes in the range [idx*NMS_TILE_SIZE, (idx+1)*NMS_TILE_SIZE).

  Parameters
  ----------
  boxes : tf.Tensor
      A tensor with a shape of [batch_size, anchors, 4].
  iou_threshold : float
      The threshold for deciding whether boxes overlap too much with respect
      to IoU.
  output_size : tf.Tensor
      An int32 tensor of size [batch_size]. Representing the number
      of selected boxes for each batch.
  idx : int
      An integer scalar representing induction variable.

  Returns
  -------
  boxes : tf.Tensor
      updated boxes.
  iou_threshold : float
      pass down iou_threshold to the next iteration.
  output_size : int
      the updated output_size.
  idx : int
      the updated induction variable.
  """
  num_tiles = tf.shape(boxes)[1] // NMS_TILE_SIZE
  batch_size = tf.shape(boxes)[0]

  # Iterates over tiles that can possibly suppress the current tile.
  box_slice = tf.slice(boxes, [0, idx * NMS_TILE_SIZE, 0],
                       [batch_size, NMS_TILE_SIZE, 4])
  _, box_slice, _, _ = tf.while_loop(
      lambda _boxes, _box_slice, _threshold, inner_idx: inner_idx < idx,
      _cross_suppression, [boxes, box_slice, iou_threshold,
                           tf.constant(0)])

  # Iterates over the current tile to compute self-suppression.
  iou = box_ops.bbox_overlap(box_slice, box_slice)
  mask = tf.expand_dims(
      tf.reshape(tf.range(NMS_TILE_SIZE), [1, -1]) > tf.reshape(
          tf.range(NMS_TILE_SIZE), [-1, 1]), 0)
  iou *= tf.cast(tf.logical_and(mask, iou >= iou_threshold), iou.dtype)
  suppressed_iou, _, _ = tf.while_loop(
      lambda _iou, loop_condition, _iou_sum: loop_condition, _self_suppression,
      [iou, tf.constant(True),
       tf.reduce_sum(iou, [1, 2])])
  suppressed_box = tf.reduce_sum(suppressed_iou, 1) > 0
  box_slice *= tf.expand_dims(1.0 - tf.cast(suppressed_box, box_slice.dtype), 2)

  # Uses box_slice to update the input boxes.
  mask = tf.reshape(
      tf.cast(tf.equal(tf.range(num_tiles), idx), boxes.dtype), [1, -1, 1, 1])
  boxes = tf.tile(
      tf.expand_dims(box_slice, [1]), [1, num_tiles, 1, 1]) * mask + \
          tf.reshape(boxes, [batch_size, num_tiles, NMS_TILE_SIZE, 4]) * \
          (1 - mask)
  boxes = tf.reshape(boxes, [batch_size, -1, 4])

  # Updates output_size.
  output_size += tf.reduce_sum(
      tf.cast(tf.reduce_any(box_slice > 0, [2]), tf.int32), [1])
  return boxes, iou_threshold, output_size, idx + 1


def sorted_non_max_suppression_padded(
    scores,
    boxes,
    max_output_size,
    iou_threshold):
  """A wrapper that handles non-maximum suppression.

  Assumption:
    * The boxes are sorted by scores unless the box is a dot (all coordinates
      are zero).
    * Boxes with higher scores can be used to suppress boxes with lower scores.

  The overal design of the algorithm is to handle boxes tile-by-tile:

  >>> boxes = boxes.pad_to_multiply_of(tile_size)
  >>> num_tiles = len(boxes) // tile_size
  >>> output_boxes = []
  >>> for i in range(num_tiles):
  >>>   box_tile = boxes[i * tile_size:(i + 1) * tile_size]
  >>>   for j in range(i - 1):
  >>>     suppressing_tile = boxes[j * tile_size:(j + 1) * tile_size]
  >>>     iou = bbox_overlap(box_tile, suppressing_tile)
  >>>     # if the box is suppressed in iou, clear it to a dot
  >>>     box_tile *= _update_boxes(iou)
  >>>   # Iteratively handle the diagnal tile.
  >>>   iou = _box_overlap(box_tile, box_tile)
  >>>   iou_changed = True
  >>>   while iou_changed:
  >>>     # boxes that are not suppressed by anything else
  >>>     suppressing_boxes = _get_suppressing_boxes(iou)
  >>>     # boxes that are suppressed by suppressing_boxes
  >>>     suppressed_boxes = _get_suppressed_boxes(iou, suppressing_boxes)
  >>>     # clear iou to 0 for boxes that are suppressed, as they cannot be used
  >>>     # to suppress other boxes any more
  >>>     new_iou = _clear_iou(iou, suppressed_boxes)
  >>>     iou_changed = (new_iou != iou)
  >>>     iou = new_iou
  >>>   # remaining boxes that can still suppress others, are selected boxes.
  >>>   output_boxes.append(_get_suppressing_boxes(iou))
  >>>   if len(output_boxes) >= max_output_size:
  >>>     break

  Parameters
  ----------
  scores : tf.Tensor
      A tensor with a shape of [batch_size, anchors].
  boxes : tf.Tensor
      A tensor with a shape of [batch_size, anchors, 4].
  max_output_size : tf.Tensor or int
      A scalar integer `Tensor` representing the maximum number of boxes to be
      selected by non max suppression.
  iou_threshold : float
      A float representing the threshold for deciding whether boxes overlap too
      much with respect to IoU.

  Returns
  -------
  nms_scores : tf.Tensor
      A tensor with a shape of [batch_size, anchors]. It has same dtype as input
      scores.
  nms_proposals : tf.Tensor
      A tensor with a shape of [batch_size, anchors, 4]. It has same dtype as
      input boxes.
  """
  batch_size = tf.shape(boxes)[0]
  num_boxes = tf.shape(boxes)[1]
  pad = tf.cast(
      tf.math.ceil(tf.cast(num_boxes, tf.float32) / NMS_TILE_SIZE),
      tf.int32) * NMS_TILE_SIZE - num_boxes
  boxes = tf.pad(tf.cast(boxes, tf.float32), [[0, 0], [0, pad], [0, 0]])
  scores = tf.pad(
      tf.cast(scores, tf.float32), [[0, 0], [0, pad]], constant_values=-1)
  num_boxes += pad

  def _loop_cond(unused_boxes, unused_threshold, output_size, idx):
    return tf.logical_and(
        tf.reduce_min(output_size) < max_output_size,
        idx < num_boxes // NMS_TILE_SIZE)

  selected_boxes, _, output_size, _ = tf.while_loop(
      _loop_cond, _suppression_loop_body, [
          boxes, iou_threshold,
          tf.zeros([batch_size], tf.int32),
          tf.constant(0)
      ])
  idx = num_boxes - tf.cast(
      tf.nn.top_k(
          tf.cast(tf.reduce_any(selected_boxes > 0, [2]), tf.int32) *
          tf.expand_dims(tf.range(num_boxes, 0, -1), 0), max_output_size)[0],
      tf.int32)
  idx = tf.minimum(idx, num_boxes - 1)
  idx = tf.reshape(
      idx + tf.reshape(tf.range(batch_size) * num_boxes, [-1, 1]), [-1])
  boxes = tf.reshape(
      tf.gather(tf.reshape(boxes, [-1, 4]), idx),
      [batch_size, max_output_size, 4])
  boxes = boxes * tf.cast(
      tf.reshape(tf.range(max_output_size), [1, -1, 1]) < tf.reshape(
          output_size, [-1, 1, 1]), boxes.dtype)
  scores = tf.reshape(
      tf.gather(tf.reshape(scores, [-1, 1]), idx),
      [batch_size, max_output_size])
  scores = scores * tf.cast(
      tf.reshape(tf.range(max_output_size), [1, -1]) < tf.reshape(
          output_size, [-1, 1]), scores.dtype)
  return scores, boxes
