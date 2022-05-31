from experiment.orbit.utils import common
import tensorflow as tf


class UtilsTest(tf.test.TestCase):

  def test_create_global_step(self):
    step = common.create_global_step()
    self.assertEqual(step.name, 'global_step:0')
    self.assertEqual(step.dtype, tf.int64)
    self.assertEqual(step, 0)
    step.assign_add(1)
    self.assertEqual(step, 1)


if __name__ == '__main__':
  tf.test.main()
