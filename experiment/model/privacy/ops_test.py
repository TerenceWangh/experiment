from unittest import mock
import tensorflow as tf
from experiment.model.privacy import ops


class OpsTest(tf.test.TestCase):

  def test_clip_l2_norm(self):
    x = tf.constant([4.0, 3.0])
    y = tf.constant([[12.0]])
    tensors = [(x, x), (y, y)]
    clipped = ops.clip_l2_norm(tensors, 1.0)
    for a, b in zip(clipped, tensors):
      self.assertAllClose(a[0], b[0] / 13.0) # sqrt(4^2 + 3^2 + 12 ^3) = 13
      self.assertAllClose(a[1], b[1])

  @mock.patch.object(tf.random,
                     'normal',
                     autospec=AttributeError)
  def test_add_noise(self, mock_random):
    x = tf.constant([0.0, 0.0])
    y = tf.constant([[0.0]])
    tensors = [(x, x), (y, y)]
    mock_random.side_effect = [tf.constant([1.0, 1.0]), tf.constant([[1.0]])]
    added = ops.add_noise(tensors, 10.0)
    for a, b in zip(added, tensors):
      self.assertAllClose(a[0], b[0] + 1.0)
      self.assertAllClose(a[1], b[1])
    _, kwargs = mock_random.call_args
    self.assertEqual(kwargs['stddev'], 10.0)


if __name__ == '__main__':
  tf.test.main()
