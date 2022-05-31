"""Faster RCNN box coder.

Faster RCNN box coder follows the coding schema described below:
  ty = (y - ya) / ha
  tx = (x - xa) / wa
  th = log(h / ha)
  tw = log(w / wa)
  where x, y, w, h denote the box's center coordinates, width and height
  respectively. Similarly, xa, ya, wa, ha denote the anchor's center
  coordinates, width and height. tx, ty, tw and th denote the anchor-encoded
  center, width and height respectively.

See https://arxiv.org/abs/1506.01497 for details.
"""

import tensorflow as tf

from experiment.utils.object_detection import BoxCoder
from experiment.utils.object_detection import BoxList

EPSILON = 1e-8


class FasterRCNNBoxCoder(BoxCoder):
  """Faster RCNN box coder."""

  def __init__(self, scale_factors = None):
    """Constructor for FasterRCNNBoxCoder.

    Parameters
    ----------
    scale_factors: list of int
        List of 4 positive scalars to scale ty, tx, th and tw. If
        set to None, does not perform scaling. For Faster RCNN, the open-source
        implementation recommends using [10.0, 10.0, 5.0, 5.0].
    """
    if scale_factors:
      assert len(scale_factors) == 4
      for scale in scale_factors:
        assert scale > 0
    self._scale_factors = scale_factors

  @property
  def code_size(self):
    return 4

  def _encode(self, boxes, anchors):
    # Convert anchors to the center coordinate representation.
    ycenter_a, xcenter_a, h_a, w_a = anchors.get_center_coordinates_and_sizes()
    ycenter, xcenter, h, w = boxes.get_center_coordinates_and_sizes()

    # Avoid NaN in division and log below.
    h_a += EPSILON
    w_a += EPSILON
    h += EPSILON
    w += EPSILON

    tx = (xcenter - xcenter_a) / w_a
    ty = (ycenter - ycenter_a) / h_a
    tw = tf.math.log(w / w_a)
    th = tf.math.log(h / h_a)

    # Scales location targets as used in paper for joint training.
    if self._scale_factors:
      ty *= self._scale_factors[0]
      tx *= self._scale_factors[1]
      th *= self._scale_factors[2]
      tw *= self._scale_factors[3]
    return tf.transpose(a=tf.stack([ty, tx, th, tw]))

  def _decode(self, rel_codes, anchors):
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()

    ty, tx, th, tw = tf.unstack(tf.transpose(a=rel_codes))
    if self._scale_factors:
      ty /= self._scale_factors[0]
      tx /= self._scale_factors[1]
      th /= self._scale_factors[2]
      tw /= self._scale_factors[3]
    w = tf.exp(tw) * wa
    h = tf.exp(th) * ha
    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.
    return BoxList(tf.transpose(a=tf.stack([ymin, xmin, ymax, xmax])))
