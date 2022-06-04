import os

from absl.testing import parameterized
import numpy as np
import orbit
import tensorflow as tf

from tensorflow.python.distribute import  combinations
from tensorflow.python.distribute import  strategy_combinations
from experiment.core import actions
from experiment import optimization


class TestModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    self.value = tf.Variable(0.0)
    self.dense = tf.keras.layers.Dense(2)
    _ = self.dense(tf.zeros((2, 2), tf.float32))

  def call(self, x, training=None):
    del training
    return self.value + x


class ActionTest(tf.test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.cloud_tpu_strategy,
              strategy_combinations.one_device_strategy,
          ],
      )
  )
  def test_ema_checkpointing(self, distribution):
    with distribution.scope():
      directory = self.create_tempdir()
      model = TestModel()
      optimizer = tf.keras.Optimizers.SGD()
      optimizer = optimization.ExponentialMovingAverage(
          optimizer, trainable_wegits_only=False)

      # Creates average weights for the model variables.
      # Average weights are initialized to zero.
      optimizer.shadow_copy(model)
      checkpoint = tf.train.Checkpoint(model=model)

      # Changes model.value to 3, average value is still 0.
      model.value.assign(3)

      # Checks model.value is 3
      self.assertEqual(model(0.), 3)
      ema_action = actions.EMACheckpointing(directory, optimizer, checkpoint)

      ema_action({})
      self.assertNotEmpty(
          tf.io.gfile.glob(os.path.join(directory, 'ema_checkpoints')))

      checkpoint.read(
          tf.train.latest_checkpoint(
              os.path.join(directory, 'ema_checkpoints')))

      # Checks model.value is 0 after swapping.
      self.assertEqual(model(0.), 0)

      # Raises an error for a normal optimizer
      with self.assertRaisesRegexp(ValueError,
                                   'Optimizer has to be instance of.*'):
        _ = actions.EMACheckpointing(
            directory, tf.keras.optimizers.SGD(), checkpoint)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.default_strategy,
              strategy_combinations.cloud_tpu_strategy,
              strategy_combinations.one_device_strategy_gpu,
          ],
      )
  )
  def test_recovery_condition(self, distribution):
    with distribution.scope():
      global_step = orbit.utils.create_global_step()
      recover_condition = actions.RecoveryCondition(
          global_step, loss_upper_bound=0.5, recovery_max_trials=2)
      outputs = {'training_loss': 0.6}
      self.assertTrue(recover_condition(outputs))
      self.assertTrue(recover_condition(outputs))
      with self.assertRaises(RuntimeError):
        recover_condition(outputs)

      global_step = orbit.utils.create_global_step()
      recover_condition = actions.RecoveryCondition(
          global_step, loss_upper_bound=0.5, recovery_max_trials=2)
      outputs = {'training_loss': tf.constant([np.nan], tf.float32)}
      self.assertTrue(recover_condition(outputs))
      self.assertTrue(recover_condition(outputs))
      with self.assertRaises(RuntimeError):
        recover_condition(outputs)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.one_device_strategy_gpu,
              strategy_combinations.one_device_strategy,
          ],
      )
  )
  def test_pruning(self, distribution):
    with distribution.scope():
      directory = self.get_temp_dir()
      model = TestModel()
      optimizer = tf.keras.optimizers.SGD()
      pruning = actions.PruningAction(directory, model, optimizer)

      pruning({})


if __name__ == '__main__':
  tf.test.main()