"""Tests for dense_prediction_heads.py."""

# Import libraries
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from experiment.model.heads import dense_prediction_heads


class RetinaNetHeadTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (False, False, False),
      (False, True, False),
      (True, False, True),
      (True, True, True),
  )
  def test_forward(self, use_separable_conv, use_sync_bn, has_att_heads):
    if has_att_heads:
      attribute_heads = [dict(name='depth', type='regression', size=1)]
    else:
      attribute_heads = None

    retinanet_head = dense_prediction_heads.RetinaNetHead(
        min_level=3,
        max_level=4,
        num_classes=3,
        num_anchors_per_location=3,
        num_convs=2,
        num_filters=256,
        attribute_heads=attribute_heads,
        use_separable_conv=use_separable_conv,
        activation='relu',
        use_sync_bn=use_sync_bn,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    features = {
        '3': np.random.rand(2, 128, 128, 16),
        '4': np.random.rand(2, 64, 64, 16),
    }
    scores, boxes, attributes = retinanet_head(features)
    self.assertAllEqual(scores['3'].numpy().shape, [2, 128, 128, 9])
    self.assertAllEqual(scores['4'].numpy().shape, [2, 64, 64, 9])
    self.assertAllEqual(boxes['3'].numpy().shape, [2, 128, 128, 12])
    self.assertAllEqual(boxes['4'].numpy().shape, [2, 64, 64, 12])
    if has_att_heads:
      for att in attributes.values():
        self.assertAllEqual(att['3'].numpy().shape, [2, 128, 128, 3])
        self.assertAllEqual(att['4'].numpy().shape, [2, 64, 64, 3])

  def test_serialize_deserialize(self):
    retinanet_head = dense_prediction_heads.RetinaNetHead(
        min_level=3,
        max_level=7,
        num_classes=3,
        num_anchors_per_location=9,
        num_convs=2,
        num_filters=16,
        attribute_heads=None,
        use_separable_conv=False,
        activation='relu',
        use_sync_bn=False,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    config = retinanet_head.get_config()
    new_retinanet_head = (
        dense_prediction_heads.RetinaNetHead.from_config(config))
    self.assertAllEqual(
        retinanet_head.get_config(), new_retinanet_head.get_config())


class RpnHeadTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (False, False),
      (False, True),
      (True, False),
      (True, True),
  )
  def test_forward(self, use_separable_conv, use_sync_bn):
    rpn_head = dense_prediction_heads.RPNHead(
        min_level=3,
        max_level=4,
        num_anchors_per_location=3,
        num_convs=2,
        num_filters=256,
        use_separable_conv=use_separable_conv,
        activation='relu',
        use_sync_bn=use_sync_bn,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    features = {
        '3': np.random.rand(2, 128, 128, 16),
        '4': np.random.rand(2, 64, 64, 16),
    }
    scores, boxes = rpn_head(features)
    self.assertAllEqual(scores['3'].numpy().shape, [2, 128, 128, 3])
    self.assertAllEqual(scores['4'].numpy().shape, [2, 64, 64, 3])
    self.assertAllEqual(boxes['3'].numpy().shape, [2, 128, 128, 12])
    self.assertAllEqual(boxes['4'].numpy().shape, [2, 64, 64, 12])

  def test_serialize_deserialize(self):
    rpn_head = dense_prediction_heads.RPNHead(
        min_level=3,
        max_level=7,
        num_anchors_per_location=9,
        num_convs=2,
        num_filters=16,
        use_separable_conv=False,
        activation='relu',
        use_sync_bn=False,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    config = rpn_head.get_config()
    new_rpn_head = dense_prediction_heads.RPNHead.from_config(config)
    self.assertAllEqual(rpn_head.get_config(), new_rpn_head.get_config())


if __name__ == '__main__':
  tf.test.main()
