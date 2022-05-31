"""Base box coder.

Box coders convert between coordinate frames, namely image-centric
(with (0,0) on the top left of image) and anchor-centric (with (0,0) being
defined by a specific anchor).

Users of a BoxCoder can call two methods:
  - encode: which encodes a box with respect to a given anchor(or rather, a
    tensor of boxes wrt a corresponding tensor of anchors).
  - decode: which inverts this encoding with a decode operation.

In both cases, the arguments are assumed to be in 1-1 correspondence already;
it is not the job of a BoxCoder to perform matching.
"""
from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf
from experiment.utils.object_detection import BoxList

# Box coder types.
FASTER_RCNN   = 'faster_rcnn'
KEYPOINT      = 'keypoint'
MEAN_STDDEV   = 'mean_stddev'
SQUARE        = 'square'


class BoxCoder:
  """Abstract base class for box coder."""
  __metaclass__ = ABCMeta

  @property
  @abstractmethod
  def code_size(self):
    """Return the size of each code.

    This number is a constant and should agree with the output of the `encode`
    op (e.g. if rel_codes is the output of self.encode(...), then it should have
    shape [N, code_size()]).  This abstractproperty should be overridden by
    implementations.

    Returns
    -------
    int
        The size.
    """

  def encode(self, boxes, anchors):
    """Encode a box list relative to an anchor collection.

    Parameters
    ----------
    boxes: BoxList
        BoxList holding N boxes to be encoded.
    anchors: BoxList
        BoxList of N anchors

    Returns
    -------
    tf.Tensor
        A tensor representing N relative-encoded boxes
    """
    with tf.name_scope('BoxEncode'):
      return self._encode(boxes, anchors)

  def decode(self, rel_codes, anchors):
    """Decode boxes that are encoded relative to an anchor collection.

    Parameters
    ----------
    rel_codes: tf.Tensor
        A tensor representing N relative-encoded boxes.
    anchors: BoxList
        BoxList of anchors

    Returns
    -------
    BoxList
        BoxList holding N boxes encoded in the ordinary way (i.e.,
        with corners y_min, x_min, y_max, x_max)
    """
    with tf.name_scope('BoxDecode'):
      return self._decode(rel_codes, anchors)

  @abstractmethod
  def _encode(self, boxes, anchors):
    """Method to be overridden by implementations.

    Parameters
    ----------
    boxes: BoxList
        BoxList holding N boxes to be encoded.
    anchors: BoxList
        BoxList of N anchors.

    Returns
    -------
    tf.Tensor
        A tensor representing N relative-encoded boxes.
    """

  @abstractmethod
  def _decode(self, rel_codes, anchors):
    """Method to be overridden by implementations.

    Parameters
    ----------
    rel_codes: tf.Tensor
        A tensor representing N relative-encoded boxes.
    anchors: BoxList
        BoxList of anchors.

    Returns
    -------
    BoxList
        BoxList holding N boxes encoded in the ordinary way (i.e.,
        with corners y_min, x_min, y_max, x_max)
    """


def batch_decode(encoded_boxes, box_coder, anchors):
  """Decode a batch of encoded boxes.

  This op takes a batch of encoded bounding boxes and transforms them to a batch
  of bounding boxes specified by their corners in the order of
  [y_min, x_min, y_max, x_max].

  Parameters
  ----------
  encoded_boxes: tf.Tensor
      A float32 tensor of shape [batch_size, num_anchors, code_size]
      representing the location of the objects.
  box_coder: BoxCoder
      A BoxCoder for decoding.
  anchors: BoxList
      A BoxList of anchors used to encode `encoded_boxes`.

  Returns
  -------
  tf.Tensor
      a float32 tensor of shape [batch_size, num_anchors, coder_size]
      representing the corners of the objects in the order of
      [y_min, x_min, y_max, x_max].

  Raises
  ------
  ValueError
      If batch sizes of the inputs are inconsistent, or if the number of anchors
      inferred from encoded_boxes and anchors are inconsistent.
  """
  encoded_boxes.get_shape().assert_has_rank(3)
  if encoded_boxes.get_shape()[1].value != anchors.num_boxes_static():
    raise ValueError(
        'The number of anchors inferred from encoded_boxes'
        ' and anchors are inconsistent: shape[1] of encoded_boxes'
        ' {} should be equal to the number of anchors: {}.'.format(
            encoded_boxes.get_shape()[1].value, anchors.num_boxes_static()))

  decoded_boxes = tf.stack([
    box_coder.decode(boxes, anchors).get()
    for boxes in tf.unstack(encoded_boxes)])
  return decoded_boxes
