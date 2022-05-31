import tensorflow as tf


class FakeKerasModel(tf.keras.Model):
  """Fake keras model for testing."""

  def __init__(self):
    super().__init__()
    self.dense = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(4, activation=tf.nn.relu)

  def call(self, inputs):
    return self.dense2(self.dense(inputs))


class _Dense(tf.Module):
  """A dense layer."""

  def __init__(self, input_dim, output_size, name=None):
    super().__init__(name=name)
    with self.name_scope:
      self.w = tf.Variable(
          tf.random.normal([input_dim, output_size]), name='w')
      self.b = tf.Variable(tf.zeros([output_size]), name='b')

  @tf.Module.with_name_scope
  def __call__(self, x):
    y = tf.matmul(x, self.w) + self.b
    return tf.nn.relu(y)


class FakeModule(tf.Module):
  """Fake model using tf.Module for testing."""

  def __init__(self, input_size, name=None):
    super().__init__(name=name)
    with self.name_scope:
      self.dense = _Dense(input_size, 4, name='dense')
      self.dense2 = _Dense(4, 4, name='dense_1')

  @tf.Module.with_name_scope
  def __call__(self, x):
    return self.dense2(self.dense(x))
