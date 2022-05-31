"""Mock task for testing."""
import dataclasses
import numpy as np
import tensorflow as tf

from experiment.core import base_task
from experiment.core import config_definitions as cfg
from experiment.core import exp_factory
from experiment.hyperparams import base_config


class MockModel(tf.keras.Model):

  def __init__(self, network):
    super().__init__()
    self.network = network

  def call(self, inputs):
    outputs = self.network(inputs)
    self.add_loss(tf.reduce_mean(outputs))
    return outputs


@dataclasses.dataclass
class MockTaskConfig(cfg.TaskConfig):
  pass


@base_config.bind(MockTaskConfig)
class MockTask(base_task.Task):
  """Mock task object for testing."""

  def __init__(self, params=None, logging_dir=None, name=None):
    super().__init__(params=params, logging_dir=logging_dir, name=name)

  def build_model(self, *arg, **kwargs):
    inputs = tf.keras.layers.Input(shape=(2,), name="random", dtype=tf.float32)
    outputs = tf.keras.layers.Dense(
        1, bias_initializer=tf.keras.initializers.Ones(), name="dense_0")(
            inputs)
    network = tf.keras.Model(inputs=inputs, outputs=outputs)
    return MockModel(network)

  def build_metrics(self, training: bool = True):
    del training
    return [tf.keras.metrics.Accuracy(name="acc")]

  def validation_step(self, inputs, model: tf.keras.Model, metrics=None):
    logs = super().validation_step(inputs, model, metrics)
    logs["counter"] = tf.constant(1, dtype=tf.float32)
    return logs

  def build_inputs(self, params):

    def generate_data(_):
      x = tf.zeros(shape=(2,), dtype=tf.float32)
      label = tf.zeros([1], dtype=tf.int32)
      return x, label

    dataset = tf.data.Dataset.range(1)
    dataset = dataset.repeat()
    dataset = dataset.map(
        generate_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset.prefetch(buffer_size=1).batch(2, drop_remainder=True)

  def aggregate_logs(self, state, step_outputs):
    if state is None:
      state = {}
    for key, value in step_outputs.items():
      if key not in state:
        state[key] = []
      state[key].append(
          np.concatenate([np.expand_dims(v.numpy(), axis=0) for v in value]))
    return state

  def reduce_aggregated_logs(self, aggregated_logs, global_step=None):
    for k, v in aggregated_logs.items():
      aggregated_logs[k] = np.sum(np.stack(v, axis=0))
    return aggregated_logs


@exp_factory.register_config_factory("mock")
def mock_experiment() -> cfg.ExperimentConfig:
  config = cfg.ExperimentConfig(
      task=MockTaskConfig(), trainer=cfg.TrainerConfig())
  return config
