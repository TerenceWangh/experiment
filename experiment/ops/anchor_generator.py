"""Multi scale anchor generator definition."""

import tensorflow as tf


class _SingleAnchorGenerator:
  """Utility to generate anchors for a single feature map.

  Examples
  --------
  >>> anchor_gen = _SingleAnchorGenerator(32, [.5, 1., 2.], stride=16)
  >>> anchors = anchor_gen([512, 512, 3])
  """

  def __init__(self,
               anchor_size,
               scales,
               aspect_ratios,
               stride,
               clip_boxes=False):
    """Constructs single scale anchor.

    Parameters
    ----------
    anchor_size: int
        A single int represents the base anchor size. The anchor height will
        be `anchor_size / sqrt(aspect_ratio)`, anchor width will be
        `anchor_size * sqrt(aspect_ratio)`.
    scales: array_like
        The actual anchor size to the base `anchor_size`.
    aspect_ratios: array_like
        The ratio of anchor width to anchor height.
    stride: int
        The anchor stride size between center of each anchor.
    clip_boxes: bool optional
        Whether the anchor coordinates should be clipped to the image size.
        Defaults to `True`.
    """
    self._anchor_size = anchor_size
    self._scales = scales
    self._aspect_ratios = aspect_ratios
    self._stride = stride
    self._clip_boxes = clip_boxes

  def __call__(self, image_size):
    image_h = tf.cast(image_size[0], tf.float32)
    image_w = tf.cast(image_size[1], tf.float32)

    k = len(self._scales) * len(self._aspect_ratios)
    aspect_ratios_sqrt = tf.cast(tf.sqrt(self._aspect_ratios), dtype=tf.float32)
    anchor_sz = tf.cast(self._anchor_size, tf.float32)

    anchor_hs, anchor_ws = [], []
    for scale in self._scales:
      anchor_st = anchor_sz * scale
      anchor_h = anchor_st / aspect_ratios_sqrt
      anchor_w = anchor_st * aspect_ratios_sqrt
      anchor_hs.append(anchor_h)
      anchor_ws.append(anchor_w)

    anchor_hs = tf.concat(anchor_hs, axis=0)
    anchor_ws = tf.concat(anchor_ws, axis=0)
    anchor_h_hs = tf.reshape(0.5 * anchor_hs, [1, 1, k])
    anchor_h_ws = tf.reshape(0.5 * anchor_ws, [1, 1, k])

    stride = tf.cast(self._stride, tf.float32)
    cx = tf.range(0.5 * stride, image_w, stride) # [W]
    cy = tf.range(0.5 * stride, image_h, stride) # [H]
    cx_grid, cy_grid = tf.meshgrid(cx, cy) # [H, W]
    cx_grid = tf.expand_dims(cx_grid, axis=-1) # [H, W, 1]
    cy_grid = tf.expand_dims(cy_grid, axis=-1) # [H, W, 1]
    y_min = tf.expand_dims(cy_grid - anchor_h_hs, axis=-1) # [H, W, K, 1]
    y_max = tf.expand_dims(cy_grid + anchor_h_hs, axis=-1) # [H, W, K, 1]
    x_min = tf.expand_dims(cx_grid - anchor_h_ws, axis=-1) # [H, W, K, 1]
    x_max = tf.expand_dims(cx_grid + anchor_h_ws, axis=-1) # [H, W, K, 1]

    if self._clip_boxes:
      y_min = tf.maximum(tf.minimum(y_min, image_h), 0.0)
      y_max = tf.maximum(tf.minimum(y_max, image_h), 0.0)
      x_min = tf.maximum(tf.minimum(x_min, image_w), 0.0)
      x_max = tf.maximum(tf.minimum(x_max, image_w), 0.0)

    result = tf.concat([y_min, x_min, y_max, x_max], axis=-1) # [H, W, K, 4]
    shape = result.shape.as_list()
    # [H, W, K * 4]
    return tf.reshape(result, [shape[0], shape[1], shape[2] * shape[3]])


class AnchorGenerator():
  """Utility to generate anchors for a multiple feature maps.

  Examples
  --------
  >>> anchor_gen = AnchorGenerator([32, 64], [.5, 1., 2.], strides=[16, 32])
  >>> anchors = anchor_gen([512, 512, 3])
  """

  def __init__(self,
               anchor_sizes,
               scales,
               aspect_ratios,
               strides,
               clip_boxes=False):
    """Constructs multiscale anchors.

    Parameters
    ----------
    anchor_sizes: array_like
        The anchor size for each scale. The anchor height will be
        `anchor_size / sqrt(aspect_ratio)`, anchor width  will be
        `anchor_size * sqrt(aspect_ratio)` for each scale.
    scales: array_like
        The actual anchor size to the base `anchor_size`.
    aspect_ratios: array_like
        The ratio of anchor width to anchor height.
    strides: array_like
        The anchor stride size between center of anchors at each scale.
    clip_boxes: bool
        Whether the anchor coordinates should be clipped to the image size.
        Defaults to `False`.
    """
    # aspect_ratio is a single list that is the same across all levels.
    aspect_ratios = maybe_map_structure_for_anchor(aspect_ratios, anchor_sizes)
    scales = maybe_map_structure_for_anchor(scales, anchor_sizes)
    if isinstance(anchor_sizes, dict):
      self._anchor_generators = {}
      for k in anchor_sizes.keys():
        self._anchor_generators[k] = _SingleAnchorGenerator(
            anchor_sizes[k], scales[k], aspect_ratios[k], strides[k],
            clip_boxes)
    elif isinstance(anchor_sizes, (list, tuple)):
      self._anchor_generators = []
      for anchor_size, scale_list, ar_list, stride in zip(
          anchor_sizes, scales, aspect_ratios, strides):
        self._anchor_generators.append(
            _SingleAnchorGenerator(
                anchor_size, scale_list, ar_list, stride, clip_boxes))

  def __call__(self, image_size):
    anchor_generators = tf.nest.flatten(self._anchor_generators)
    # pylint: disable=not-callable
    results = [anchor_gen(image_size) for anchor_gen in anchor_generators]
    # pylint: enable=not-callable
    return tf.nest.pack_sequence_as(self._anchor_generators, results)


def maybe_map_structure_for_anchor(params, anchor_sizes):
  """broadcast the params to match anchor_sizes."""
  if all(isinstance(param, (int, float)) for param in params):
    if isinstance(anchor_sizes, (tuple, list)):
      return [params] * len(anchor_sizes)
    elif isinstance(anchor_sizes, dict):
      return tf.nest.map_structure(lambda _: params, anchor_sizes)
    else:
      raise ValueError('The structure of `anchor_sizes` must be a tuple, '
                       'list or dict, given {}.'.format(anchor_sizes))
  else:
    return params
