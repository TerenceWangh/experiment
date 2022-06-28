import immutabledict
import tensorflow as tf

from experiment.model.backbones import factory
from experiment.model.layers import nn_layers

layers = tf.keras.layers

VIT_SPECS = immutabledict.immutabledict({
    'vit-ti16': {
        'hidden_size': 192,
        'patch_size': 16,
        'transformer': {
            'mlp_dim': 768,
            'num_heads': 3,
            'num_layers': 12,
        },
    },
    'vit-s16': {
        'hidden_size': 384,
        'patch_size': 16,
        'transformer': {
            'mlp_dim': 1536,
            'num_heads': 6,
            'num_layers': 12,
        },
    },
    'vit-b16': {
        'hidden_size': 768,
        'patch_size': 16,
        'transformer': {
            'mlp_dim': 3072,
            'num_heads': 12,
            'num_layers': 12,
        },
    },
    'vit-b32': {
        'hidden_size': 768,
        'patch_size': 32,
        'transformer': {
            'mlp_dim': 3072,
            'num_heads': 12,
            'num_layers': 12,
        },
    },
    'vit-l16': {
        'hidden_size': 1024,
        'patch_size': 16,
        'transformer': {
            'mlp_dim': 4096,
            'num_heads': 16,
            'num_layers': 24,
        },
    },
    'vit-l32': {
        'hidden_size': 1024,
        'patch_size': 32,
        'transformer': {
            'mlp_dim': 4096,
            'num_heads': 16,
            'num_layers': 24,
        },
    },
    'vit-h14': {
        'hidden_size': 1280,
        'patch_size': 14,
        'transformer': {
            'mlp_dim': 5120,
            'num_heads': 16,
            'num_layers': 32,
        },
    },
    'vit-g14': {
        'hidden_size': 1664,
        'patch_size': 14,
        'transformer': {
            'mlp_dim': 8192,
            'num_heads': 16,
            'num_layers': 48,
        },
    },
})


@tf.keras.utils.register_keras_serializable(package='experiment')
class VisionTransformer(tf.keras.Model):
  """Class to build VisionTransformer family model."""

  def __init__(
      self,
      mlp_dim: int = 3072,
      num_heads: int = 12,
      num_layers: int = 12,
      attention_dropout_rate: float = 0.0,
      dropout_rate: float = 0.1,
      init_stochastic_depth_rate: float = 0.0,
      input_specs: layers.InputSpec = layers.InputSpec(
          shape=[None, None, None, 3]),
      patch_size: int = 16,
      hidden_size: int = 768,
      representation_size: int = 0,
      classifier: str = 'token',
      kernel_regularizer: tf.keras.regularizers.Regularizer = None,
      original_init: bool = True):
    inputs = tf.keras.Input(shape=input_specs.shape[1:])

    x = layers.Conv2D(
        filters=hidden_size,
        kernel_size=patch_size,
        strides=patch_size,
        padding='valid',
        kernel_regularizer=kernel_regularizer,
        kernel_initializer='lecun_normal' if original_init else 'he_uniform'
    )(inputs)

    if tf.keras.backend.image_data_format() == 'channels_last':
      rows_axis, cols_axis = 1, 2
    else:
      rows_axis, cols_axis = 2, 3
      # The reshape below assumes the data_format is 'channels_last,' so
      # transpose to that. Once the data is flattened by the reshape, the
      # data_format is irrelevant, so no need to update
      # tf.keras.backend.image_data_format.
      x = tf.transpose(x, perm=[0, 2, 3, 1])
    row_len = input_specs.shape[rows_axis] // patch_size
    col_len = input_specs.shape[cols_axis] // patch_size
    seq_len = row_len * col_len
    x = tf.reshape(x, [-1, seq_len, hidden_size])

    # If we want to add a class token, add it here
    if classifier == 'token':
      x = nn_layers.VisionTransformerToken(name='cls')(x)

    x = nn_layers.VisionTransformerEncoder(
        num_layers=num_layers,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer='glorot_uniform' if original_init else dict(
            class_name='TruncatedNormal', config=dict(stddev=.02)),
        init_stochastic_depth_rate=init_stochastic_depth_rate
    )(x)

    if classifier == 'token':
      x = x[:, 0]
    elif classifier == 'gap':
      x = tf.reduce_mean(x, axis=1)

    if representation_size:
      x = tf.keras.layers.Dense(
          representation_size,
          kernel_regularizer=kernel_regularizer,
          name='pre_logits',
          kernel_initializer='lecun_norm' if original_init else 'he_uniform'
      )(x)
      x = tf.nn.tanh(x)
    else:
      x = tf.identity(x, name='pre_logits')

    x = tf.reshape(x, [-1, 1, 1, representation_size or hidden_size])

    end_points = {
        'pre_logits': x
    }

    super(VisionTransformer, self).__init__(inputs=inputs, outputs=end_points)


@factory.register_backbone_builder('vit')
def build_vit(input_specs,
              backbone_config,
              norm_activation_config,
              l2_regularizer=None):
  del norm_activation_config
  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == 'vit', 'Inconsistent backbone type {}.'.format(
      backbone_type)

  backbone_cfg.override(VIT_SPECS[backbone_cfg.model_name])

  return VisionTransformer(
      mlp_dim=backbone_cfg.transformer.mlp_dim,
      num_heads=backbone_cfg.transformer.num_heads,
      num_layers=backbone_cfg.transformer.num_layers,
      attention_dropout_rate=backbone_cfg.transformer.attention_dropout_rate,
      dropout_rate=backbone_cfg.transformer.dropout_rate,
      init_stochastic_depth_rate=backbone_cfg.init_stochastic_depth_rate,
      input_specs=input_specs,
      patch_size=backbone_cfg.patch_size,
      hidden_size=backbone_cfg.hidden_size,
      representation_size=backbone_cfg.representation_size,
      classifier=backbone_cfg.classifier,
      kernel_regularizer=l2_regularizer,
      original_init=backbone_cfg.original_init)
