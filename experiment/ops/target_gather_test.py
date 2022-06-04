import tensorflow as tf

from experiment.ops import target_gather


class TargetGatherTest(tf.test.TestCase):

  def test_target_gather_batched(self):
    gt_boxes = tf.constant(
        [[
            [0, 0, 5, 5],
            [0, 5, 5, 10],
            [5, 0, 10, 5],
            [5, 5, 10, 10],
        ]],
        dtype=tf.float32)
    gt_classes = tf.constant([[[2], [10], [3], [-1]]], dtype=tf.int32)

    labeler = target_gather.TargetGather()

    match_indices = tf.constant([[0, 2]], dtype=tf.int32)
    match_indicators = tf.constant([[-2, 1]])
    mask = tf.less_equal(match_indicators, 0)
    cls_mask = tf.expand_dims(mask, -1)
    matched_gt_classes = labeler(gt_classes, match_indices, cls_mask)
    box_mask = tf.tile(cls_mask, [1, 1, 4])
    matched_gt_boxes = labeler(gt_boxes, match_indices, box_mask)

    self.assertAllEqual(
        matched_gt_classes.numpy(), [[[0], [3]]])
    self.assertAllClose(
        matched_gt_boxes.numpy(), [[[0, 0, 0, 0], [5, 0, 10, 5]]])

  def test_target_gather_unbatched(self):
    gt_boxes = tf.constant(
        [
            [0, 0, 5, 5],
            [0, 5, 5, 10],
            [5, 0, 10, 5],
            [5, 5, 10, 10],
        ],
        dtype=tf.float32)
    gt_classes = tf.constant([[2], [10], [3], [-1]], dtype=tf.int32)

    labeler = target_gather.TargetGather()

    match_indices = tf.constant([0, 2], dtype=tf.int32)
    match_indicators = tf.constant([-2, 1])
    mask = tf.less_equal(match_indicators, 0)
    cls_mask = tf.expand_dims(mask, -1)
    matched_gt_classes = labeler(gt_classes, match_indices, cls_mask)
    box_mask = tf.tile(cls_mask, [1, 4])
    matched_gt_boxes = labeler(gt_boxes, match_indices, box_mask)

    self.assertAllEqual(
        matched_gt_classes.numpy(), [[0], [3]])
    self.assertAllClose(
        matched_gt_boxes.numpy(), [[0, 0, 0, 0], [5, 0, 10, 5]])


if __name__ == '__main__':
  tf.test.main()
