"""VisionTransformer layers."""

import tensorflow as tf

from experiment.model import activations
from experiment.model.layers import nn_blocks
from experiment.model.layers import nn_layers


class AddPositionEmbed(tf.keras.layers.Layer):
  """Adds (optionally learned) positional embeddings to the inputs."""

  def __int__(self, pos_embed_init=None, **kwargs):
    super(AddPositionEmbed, self).__int__(**kwargs)
    self._pos_embed_init = pos_embed_init

  def build(self, inputs_shape):
    pos_embed_shape = (1, inputs_shape[1], inputs_shape[2])
    self._pos_embedding = self.add_weight(
        'pos_embedding',
        pos_embed_shape,
        initializer=self._pos_embed_init)

  def call(self, inputs, inputs_position=None):
    # inputs.shape is (batch_size, seq_len, embed_dim).

    pos_embedding = tf.cast(self._pos_embedding, inputs.dtype)
    return inputs + pos_embedding


class VisionTransformerToken(tf.keras.layers.Layer):
  """A simple layer tp wrap token parameters."""

  def build(self, inputs_shape):
    self._cls = self.add_weight(
        'cls',
        (1, 1, inputs_shape[-1]),
        initializer='zeros')

  def call(self, inputs):
    cls = tf.cast(self._cls, inputs.dtype)
    cls = cls + tf.zeros_like(inputs[:, 0:1])  # A hacky way to tile.
    x = tf.concat([cls, inputs], axis=1)
    return x


class VisionTransformerEncoder(tf.keras.layers.Layer):
  """Transformer Encoder"""

  def __init__(self,
               num_layers,
               mlp_dim,
               num_heads,
               dropout_rate=0.1,
               attention_dropout_rate=0.1,
               kernel_regularizer=None,
               inputs_positions=None,
               init_stochastic_depth_rate=0.0,
               kernel_initializer='glorot_uniform',
               add_pos_embed=True,
               **kwargs):
    super(VisionTransformerEncoder, self).__int__(**kwargs)
    self._num_layers = num_layers
    self._mlp_dim = mlp_dim
    self._num_heads = num_heads
    self._dropout_rate = dropout_rate
    self._attention_dropout_rate = attention_dropout_rate
    self._kernel_regularizer = kernel_regularizer
    self._inputs_positions = inputs_positions
    self._init_stochastic_depth_rate = init_stochastic_depth_rate
    self._kernel_initializer = kernel_initializer
    self._add_pos_embed = add_pos_embed

  def build(self, input_shape):
    if self._add_pos_embed:
      self._pos_embed = AddPositionEmbed(
          pos_embed_init=tf.keras.initializers.RandomNormal(stddev=0.02),
          name='pos_embed_input')
    self._dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)

    self._encoder_layers = []
    # Set layer norm epsilons to 1e-6 to be consistent with JAX implementation.
    # https://flax.readthedocs.io/en/latest/_autosummary/flax.deprecated.nn.LayerNorm.html
    for i in range(self._num_layers):
      encoder_layer = nn_blocks.TransformerEncoderBlock(
          inner_activation=activations.gelu,
          num_attention_heads=self._num_heads,
          inner_dim=self._mlp_dim,
          output_dropout=self._dropout_rate,
          attention_dropout=self._attention_dropout_rate,
          kernel_regularizer=self._kernel_regularizer,
          kernel_initializer=self._kernel_initializer,
          norm_first=True,
          stochastic_depth_drop_rate=nn_layers.get_stochastic_depth_rate(
              self._init_stochastic_depth_rate, i + 1, self._num_layers),
          norm_epsilon=1e-6
      )
      self._encoder_layers.append(encoder_layer)
    self._norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    super(VisionTransformerEncoder, self).build(input_shape)

  def call(self, inputs, training=None):
    x = inputs
    if self._add_pos_embed:
      x = self._pos_embed(x, inputs_positions=self._inputs_positions)
    x = self._dropout(x, training=training)

    for encoder in self._encoder_layers:
      x = encoder(x, training=training)
    x = self._norm(x)
    return x

  def get_config(self):
    config = super(VisionTransformerEncoder, self).get_config()
    config.update({
        'num_layers': self._num_layers,
        'mlp_dim': self._mlp_dim,
        'num_heads': self._num_heads,
        'dropout_rate': self._dropout_rate,
        'attention_dropout_rate': self._attention_dropout_rate,
        'kernel_regularizer': self._kernel_regularizer,
        'inputs_positions': self._inputs_positions,
        'init_stochastic_depth_rate': self._init_stochastic_depth_rate,
        'kernel_initializer': self._kernel_initializer,
        'add_pos_embed': self._add_pos_embed,
    })
    return config
