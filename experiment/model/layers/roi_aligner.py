"""Contains definitions of ROI aligner."""

from typing import Mapping, Optional
import tensorflow as tf

from experiment.ops import spatial_transform_ops


@tf.keras.utils.register_keras_serializable(package='experiment')
class MultiLevelRoIAligner(tf.keras.layers.Layer):
  """Performs RoIAlign for the second stage processing."""

  def __init__(self, crop_size: int = 7, sample_offset: float = 0.5, **kwargs):
    """Initializes a RoI aligner.

    Parameters
    ----------
    crop_size : int, default 7
        The output size of the cropped features.
    sample_offset : float, default 0.5
        The subpixel sample offset.
    kwargs : dict
        Additional keyword arguments passed to Layer.
    """
    self._crop_size = crop_size
    self._sample_offset = sample_offset
    super(MultiLevelRoIAligner, self).__init__(**kwargs)

  def call(self,
           features: Mapping[str, tf.Tensor],
           boxes: tf.Tensor,
           training: Optional[bool] = None):
    """Generate RoIs.

    Parameters
    ----------
    features : dict
        A dictionary with key as pyramid level and value as features.
        The features are in shape of
        [batch_size, height_l, width_l, num_filters].
    boxes : tf.Tensor
        A 3-D `tf.Tensor` of shape [batch_size, num_boxes, 4]. Each row
        represents a box with [y1, x1, y2, x2] in un-normalized coordinates.
        from grid point.
    training : bool, optional
        Whether it is in training mode.

    Returns
    -------
    tf.Tensor
        A 5-D `tf.Tensor` representing feature crop of shape
        [batch_size, num_boxes, crop_size, crop_size, num_filters].
    """
    roi_features = spatial_transform_ops.multilevel_crop_and_resize(
        features, boxes,
        output_size=self._crop_size,
        sample_offset=self._sample_offset)
    return roi_features

  def get_config(self):
    config = super(MultiLevelRoIAligner, self).get_config()
    config.update({
        'crop_size': self._crop_size,
        'sample_offset': self._sample_offset,
    })
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
