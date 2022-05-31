import tensorflow as tf
from experiment.model import grad_utils
from experiment.model import performance


class GradUtilsTest(tf.test.TestCase):

  def test_minimize(self):
    optimizer = tf.keras.optimizers.SGD(0.1)
    with tf.GradientTape() as tape:
      model = tf.keras.layers.Dense(2)
      outputs = model(tf.zeros((2, 2), tf.float32))
      loss = tf.reduce_mean(outputs)

    grad_utils.minimize_using_explicit_all_reduce(tape, optimizer, loss,
                                                  model.trainable_variables)

  def test_minimize_fp16(self):
    optimizer = performance.configure_optimizer(
      tf.keras.optimizers.SGD(0.1), use_float16=True)
    performance.set_mixed_precision_policy(tf.float16)
    with tf.GradientTape() as tape:
      model = tf.keras.layers.Dense(2)
      outputs = model(tf.zeros((2, 2), tf.float16))
      loss = tf.reduce_mean(outputs)

    grad_utils.minimize_using_explicit_all_reduce(tape, optimizer, loss,
                                                  model.trainable_variables)

    # Test other fp16 settings.
    def _clip_by_global_norm(grads_and_vars):
      grads, tvars = list(zip(*grads_and_vars))
      (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
      return zip(grads, tvars)

    with tf.GradientTape() as tape:
      model = tf.keras.layers.Dense(2)
      outputs = model(tf.zeros((2, 2), tf.float16))
      loss = tf.reduce_mean(outputs)
    optimizer = performance.configure_optimizer(
      tf.keras.optimizers.SGD(0.1), use_float16=True, loss_scale=128)
    grad_utils.minimize_using_explicit_all_reduce(
      tape,
      optimizer,
      loss,
      model.trainable_variables,
      pre_all_reduce_callbacks=[_clip_by_global_norm],
      post_all_reduce_callbacks=[_clip_by_global_norm])

  def test_set_mixed_precision_policy(self):
    performance.set_mixed_precision_policy(tf.float16)
    performance.set_mixed_precision_policy(tf.bfloat16)
    performance.set_mixed_precision_policy(tf.float32)

    with self.assertRaises(ValueError):
      performance.set_mixed_precision_policy(tf.int32)

if __name__ == '__main__':
  tf.test.main()
