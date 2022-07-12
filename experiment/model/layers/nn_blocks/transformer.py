"""Keras-based TransformerEncoder block layer."""

import tensorflow as tf

from experiment.model.layers import nn_layers


@tf.keras.utils.register_keras_serializable(package="experiment")
class TransformerEncoderBlock(tf.keras.layers.Layer):
  """TransformerEncodeBlock Layer

  This layer implements the Transformer Encoder from
  "Attention Is All You Need". (https://arxiv.org/abs/1706.03762),
  which combines a `tf.keras.layers.MultiHeadAttention` layer with a
  two-layer feedforward network.

  References
  ==========
  [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  [BERT: Pre-training of Deep Bidirectional Transformers for Language
     Understanding](https://arxiv.org/abs/1810.04805)
  """

  def __init__(self,
               num_attention_heads,
               inner_dim,
               inner_activation,
               output_range=None,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               use_bias=True,
               norm_first=False,
               norm_epsilon=1e-12,
               output_dropout=0.0,
               attention_dropout=0.0,
               inner_dropout=0.0,
               attention_initializer=None,
               attention_axes=None,
               key_dim=None,
               value_dim=None,
               output_last_dim=None,
               diff_q_kv_attention_layer_norm=False,
               stochastic_depth_drop_rate=0.0,
               return_attention=False,
               **kwargs):
    """Initializes `TransformerEncoderBlock`.

    Examples
    Let's say input dims are `[batch_size, seq_dim, input_last_dim]`.
    Scenario 1: If `output_last_dim` is not `None`, then the output dims of this
    module would be `[batch_size, seq_dim, output_last_dim]`. Note `key_dim` is
    overridden by `output_last_dim`.
    Scenario 2: If `output_last_dim` is `None` and `key_dim` is not `None`, then
    the output dims of this module would be `[batch_size, seq_dim, key_dim]`.
    Scenario 3: If the `output_last_dim` and `key_dim` are both `None`, the
    output dims would be `[batch_size, seq_dim, input_last_dim]`.

    Parameters
    ----------
    num_attention_heads : int
        The number of attention heads.
    inner_dim : int
        The output dimension of the first Dense layer in a two-layer
        feedforward network.
    inner_activation : str
        The activation for the first Dense layer in a two-layer
        feedforward network.
    output_range : tuple, optional
        The sequence output range, [0, output_range) for slicing the target
        sequence. `None` means the target sequence is not sliced.
    kernel_initializer : str, default 'glorot_uniform'
        Initializer for dense layer kernels.
    bias_initializer : str, default 'zeros'
        Initializer for dense layer biases.
    kernel_regularizer : str, optional
        Regularizer for dense layer kernels.
    bias_regularizer : str, optional
        Regularizer for dense layer biases.
    activity_regularizer : str, optional
        Regularizer for dense layer activity.
    kernel_constraint : str, optional
        Constraint for dense layer kernels.
    bias_constraint : str, optional
        Constraint for dense layer bias.
    use_bias : bool, default True
        Whether to enable use_bias in attention layer. If set False,
        use_bias in attention layer is disabled.
    norm_first : bool, default False
        Whether to normalize inputs to attention and intermediate dense layers.
        If set False, output of attention and intermediate dense layers is
        normalized.
    norm_epsilon : float, default 1e-12
        Epsilon value to initialize normalization layers.
    output_dropout : float, default 0.0
        Dropout probability for the post-attention and output dropout.
    attention_dropout : float, default 0.0
        Dropout probability for within the attention layer.
    inner_dropout : float, default 0.0
        Dropout probability for the first Dense layer in a two-layer feedforward
        network.
    attention_initializer : str, optional
        Initializer for kernels of attention layers. If set `None`, attention
        layers use kernel_initializer as initializer for kernel.
    attention_axes : list, optional
        The axes over which the attention is applied. `None` means
        attention over all axes, but batch, heads, and features.
    key_dim : int, optional
        `key_dim` for the `tf.keras.layers.MultiHeadAttention`. If
        `None`, we use the first `input_shape`'s last dim.
    value_dim : int, optional
        `value_dim` for the `tf.keras.layers.MultiHeadAttention`.
    output_last_dim : int, optional
        Final dimension of the output of this module. This also dictates the
        value for the final dimension of the multi-head-attention. When it's
        `None`, we use, in order of decreasing precedence,
        `key_dim` * `num_heads` or the first `input_shape`'s last dim as the
        output's last dim.
    diff_q_kv_attention_layer_norm : bool, default False
        If `True`, create a separate attention layer norm layer for query and
        key-value if `norm_first` is `True`. Invalid to set to `True` if
        `norm_first` is `False`.
    stochastic_depth_drop_rate : float, default 0.0
        The drop rate for stochastic depth.
    return_attention : bool, default False
        Whether return attention.
    """
    super().__init__(**kwargs)

    self._num_heads = num_attention_heads
    self._inner_dim = inner_dim
    self._inner_activation = inner_activation
    self._attention_dropout = attention_dropout
    self._output_dropout = output_dropout
    self._output_range = output_range
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf.keras.initializers.get(bias_initializer)
    self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self._activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
    self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
    self._bias_constraint = tf.keras.constraints.get(bias_constraint)
    self._use_bias = use_bias
    self._norm_first = norm_first
    self._norm_epsilon = norm_epsilon
    self._inner_dropout = inner_dropout
    self._key_dim = key_dim
    self._value_dim = value_dim
    self._output_last_dim = output_last_dim
    self._diff_q_kv_attention_layer_norm = diff_q_kv_attention_layer_norm
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._return_attention = return_attention
    if attention_initializer:
      self._attention_initializer = tf.keras.initializers.get(
          attention_initializer)
    else:
      self._attention_initializer = self._kernel_initializer
    self._attention_axes = attention_axes

    if self._diff_q_kv_attention_layer_norm and not self._norm_first:
      raise ValueError(
          'Setting `diff_q_kv_attention_layer_norm` to True when `norm_first` '
          'is False is invalid.')

  def build(self, input_shape):
    if isinstance(input_shape, tf.TensorShape):
      input_tensor_shape = input_shape
    elif isinstance(input_shape, (list, tuple)):
      input_tensor_shape = tf.TensorShape(input_shape[0])
    else:
      raise ValueError('The type of input shape argument is not supported: '
                       '{}'.format(type(input_shape)))

    einsum_equation = 'abc,cd->abd'
    if len(input_tensor_shape.as_list()) > 3:
      einsum_equation = '...bc,cd->...bd'
    hidden_size = input_tensor_shape[-1]
    if hidden_size % self._num_heads != 0:
      raise ValueError(
          'The input size({}) is not a multiple of the number of attention '
          'heads ({}).'.format(hidden_size, self._num_heads))
    if self._key_dim is None:
      self._key_dim = int(hidden_size // self._num_heads)
    if self._output_last_dim is None:
      last_output_shape = hidden_size
    else:
      last_output_shape = self._output_last_dim

    common_kwargs = {
        'bias_initializer': self._bias_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activity_regularizer': self._activity_regularizer,
        'kernel_constraint': self._kernel_constraint,
        'bias_constraint': self._bias_constraint,
    }
    self._attention_layer = tf.keras.layers.MultiHeadAttention(
        num_heads=self._num_heads,
        key_dim=self._key_dim,
        value_dim=self._value_dim,
        dropout=self._attention_dropout,
        use_bias=self._use_bias,
        kernel_initializer=self._attention_initializer,
        attention_axes=self._attention_axes,
        output_shape=self._output_last_dim,
        name='self_attention',
        **common_kwargs)
    self._attention_dropout_layer = tf.keras.layers.Dropout(
        rate=self._output_dropout)
    # Use float32 in layernorm for numeric stability.
    # It is probably safe in mixed_float16, but we haven't validated this yet.
    self._attention_layer_norm = tf.keras.layers.LayerNormalization(
        name='self_attention_layer_norm',
        axis=-1,
        epsilon=self._norm_epsilon,
        dtype=tf.float32)
    self._attention_layer_norm_kv = self._attention_layer_norm
    if self._diff_q_kv_attention_layer_norm:
      self._attention_layer_norm_kv = tf.keras.layers.LayerNormalization(
          name='self_attention_layer_norm_kv',
          axis=-1,
          epsilon=self._norm_epsilon,
          dtype=tf.float32)

    self._intermediate_dense = tf.keras.layers.experimental.EinsumDense(
        einsum_equation,
        output_shape=(None, self._inner_dim),
        bias_axes='d',
        kernel_initializer=self._kernel_initializer,
        name='intermediate',
        **common_kwargs)
    policy = tf.keras.mixed_precision.global_policy()
    if policy.name == 'mixed_bfloat16':
      # bfloat16 causes BERT with the LAMB optimizer to not converge as well,
      # so we use float32.
      policy = tf.float32
    self._intermediate_activation_layer = tf.keras.layers.Activation(
        self._inner_activation, dtype=policy)
    self._inner_dropout_layer = tf.keras.layers.experimental.EinsumDense(
        einsum_equation,
        output_shape=(None, last_output_shape),
        bias_axes='d',
        name='output',
        kernel_initializer=self._kernel_initializer,
        **common_kwargs)

    self._output_dense = tf.keras.layers.experimental.EinsumDense(
        einsum_equation,
        output_shape=(None, last_output_shape),
        bias_axes='d',
        name='output',
        kernel_initializer=self._kernel_initializer,
        **common_kwargs)
    self._output_dropout_layer = tf.keras.layers.Dropout(
        rate=self._output_dropout)
    self._output_layer_norm = tf.keras.layers.LayerNormalization(
        name='output_layer_norm',
        axis=-1,
        epsilon=self._norm_epsilon,
        dtype=tf.float32)

    if self._stochastic_depth_drop_rate:
      self._stochastic_depth_layer = nn_layers.StochasticDepth(
          self._stochastic_depth_drop_rate)
    else:
      self._stochastic_depth_layer = lambda x, *args, **kwargs: tf.identity(x)

    super(TransformerEncoderBlock, self).build(input_shape)

  def get_config(self):
    config = super(TransformerEncoderBlock, self).get_config()
    config.update({
        'num_attention_heads': self._num_heads,
        'inner_dim': self._inner_dim,
        'inner_activation': self._inner_activation,
        'output_dropout': self._output_dropout,
        'attention_dropout': self._attention_dropout,
        'output_range': self._output_range,
        'kernel_initializer':
            tf.keras.initializers.serialize(self._kernel_initializer),
        'bias_initializer':
            tf.keras.initializers.serialize(self._bias_initializer),
        'kernel_regularizer':
            tf.keras.regularizers.serialize(self._kernel_regularizer),
        'bias_regularizer':
            tf.keras.regularizers.serialize(self._bias_regularizer),
        'activity_regularizer':
            tf.keras.regularizers.serialize(self._activity_regularizer),
        'kernel_constraint':
            tf.keras.constraints.serialize(self._kernel_constraint),
        'bias_constraint':
            tf.keras.constraints.serialize(self._bias_constraint),
        'use_bias': self._use_bias,
        'norm_first': self._norm_first,
        'norm_epsilon': self._norm_epsilon,
        'inner_dropout': self._inner_dropout,
        'attention_initializer':
            tf.keras.initializers.serialize(self._attention_initializer),
        'attention_axes': self._attention_axes,
        'key_dim': self._key_dim,
        'value_dim': self._value_dim,
        'output_last_dim': self._output_last_dim,
        'diff_q_kv_attention_layer_norm': self._diff_q_kv_attention_layer_norm,
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
        'reutrn_attention': self._return_attention,
    })
    return config

  def call(self, inputs, training=None):
    """Transformer self-attention encoder block call.

    Parameters
    ----------
    inputs : tf.Tensor or list of tf.Tensor
        `input tensor` as the single sequence of embeddings.
        [`input tensor`, `attention mask`] to have the additional attention
          mask.
        [`query tensor`, `key value tensor`, `attention mask`] to have separate
          input streams for the query, and key/value to the multi-head
          attention.

    Returns
    -------
    tf.Tensor
        An output tensor with the same dimensions as input/query tensor.
    """
    if isinstance(inputs, (list, tuple)):
      if len(inputs) == 2:
        key_value = None
        input_tensor, attention_mask = inputs
      elif len(inputs) == 3:
        input_tensor, key_value, attention_mask = inputs
      else:
        raise ValueError(
            'Unexpected inputs to {} with length at {}'.format(
                self.__class__, len(inputs)))
    else:
      input_tensor, key_value, attention_mask = inputs, None, None

    if self._output_range:
      if self._norm_first:
        source_tensor = input_tensor[:, 0:self._output_range, :]
        input_tensor = self._attention_layer_norm(input_tensor)
        if key_value is not None:
          key_value = self._attention_layer_norm_kv(key_value)
      target_tensor = input_tensor[:, 0:self._output_range, :]
      if attention_mask is not None:
        attention_mask = attention_mask[:, 0:self._output_range, :]
    else:
      if self._norm_first:
        source_tensor = input_tensor
        input_tensor = self._attention_layer_norm(input_tensor)
        if key_value is not None:
          key_value = self._attention_layer_norm_kv(key_value)
      target_tensor = input_tensor

    if key_value is None:
      key_value = input_tensor
    attention_output, attention_scores = self._attention_layer(
        query=target_tensor, value=key_value, attention_mask=attention_mask)
    attention_output = self._attention_dropout_layer(attention_output)

    if self._norm_first:
      attention_output = source_tensor + self._stochastic_depth_layer(
          attention_output, training=training)
    else:
      attention_output = target_tensor + self._stochastic_depth_layer(
          attention_output, training=training)
      attention_output = self._attention_layer_norm(attention_output)

    if self._norm_first:
      source_attention_output = attention_output
      attention_output = self._output_layer_norm(attention_output)
    inner_output = self._intermediate_dense(attention_output)
    inner_output = self._intermediate_activation_layer(inner_output)
    inner_output = self._inner_dropout_layer(inner_output)
    layer_output = self._output_dense(inner_output)
    layer_output = self._output_dropout_layer(layer_output)

    if self._norm_first:
      attention_output = source_attention_output + self._stochastic_depth(
          layer_output, training=training)
      if self._return_attention:
        return attention_output, attention_scores
      else:
        return attention_output

    # During mixed precision training, layer norm output is always fp32 for
    # now. Casts fp32 for the subsequent add.
    layer_output = tf.cast(layer_output, tf.float32)
    attention_output = self._output_layer_norm(layer_output + attention_output)
    if self._return_attention:
      return attention_output, attention_scores
    else:
      return attention_output

