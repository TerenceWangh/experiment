"""Bounding Box List definition.

BoxList represents a list of bounding boxes as tensorflow
tensors, where each bounding box is represented as a row of 4 numbers,
[y_min, x_min, y_max, x_max].  It is assumed that all bounding boxes
within a given list correspond to a single image.  See also
box_list_ops.py for common box related operations (such as area, iou, etc).

Optionally, users can add additional related fields (such as weights).

We assume the following things to be true about fields:
* they correspond to boxes in the box_list along the 0th dimension
* they have inferrable rank at graph construction time
* all dimensions except for possibly the 0th can be inferred
  (i.e., not None) at graph construction time.

Notes
-----
  * Following tensorflow conventions, we use height, width ordering,
  and correspondingly, y,x (or ymin, xmin, ymax, xmax) ordering
  * Tensors are always provided as (flat) [N, 4] tensors.
"""
import tensorflow as tf


class BoxList:
  """Box collection."""

  def __init__(self, boxes):
    """Constructs box collection.

    Parameters
    ----------
    boxes: tf.Tensor
        A tensor of shape [N, 4] representing box corners.

    Raises
    ------
    ValueError
        If invalid dimensions for bbox data or if bbox data is not in float32
        format.
    """
    if len(boxes.get_shape()) != 2 or boxes.get_shape()[-1] != 4:
      raise ValueError('Invalid dimensions for box data.')
    if boxes.dtype != tf.float32:
      raise ValueError('Invalid tensor type: should be tf.float32.')
    self._data = {'boxes': boxes}

  def num_boxes(self):
    """Returns number of boxes held in collection.

    Returns
    -------
    tf.Tensor
        A tensor representing the number of boxes held in the collection.
    """
    return tf.shape(input=self._data['boxes'])[0]

  def num_boxes_static(self):
    """Returns number of boxes held in collection.

    This number is inferred at graph construction time rather than run-time.

    Returns
    -------
    int
        Number of boxes held in collection (integer) or None if this is not
        inferrable at graph construction time.
    """
    return self._data['boxes'].get_shape().dims[0].value

  def get_all_fields(self):
    """Returns all fields."""
    return self._data.keys()

  def get_extra_fields(self):
    """Returns all non-box fields (i.e., everything not named `boxes`)."""
    return [k for k in self._data.keys() if k != 'boxes']

  def add_field(self, field, field_data):
    """Add field to box list.

    This method can be used to add related box data such as
    weights/labels, etc.

    Parameters
    ----------
    field: str
        A string key to access the data via `get`.
    field_data: tf.Tensor
        A tensor containing the data to store in the BoxList.
    """
    self._data[field] = field_data

  def has_field(self, field):
    """Whether contains the field with the key `field`.

    Parameters
    ----------
    field: str
        A string key.

    Returns
    -------
    bool
        Whether contains the field.
    """
    return field in self._data

  def get(self):
    """Convenience function for accessing box coordinates.

    Returns
    -------
    tf.Tensor
        A tensor with shape [N, 4] representing box coordinates.
    """
    return self.get_field('boxes')

  def set(self, boxes):
    """Convenience function for setting box coordinates.

    Parameters
    ----------
    boxes: tf.Tensor
        A tensor of shape [N, 4] representing box corners.

    Raises
    ------
    ValueError
        If invalid dimensions for bbox data.
    """
    if len(boxes.get_shape()) != 2 or boxes.get_shape()[-1] != 4:
      raise ValueError('Invalid dimensions for box data.')
    self._data['boxes'] = boxes

  def get_field(self, field):
    """Accesses a box collection and associated fields.

    This function returns specified field with object; if no field is specified,
    it returns the box coordinates.

    Parameters
    ----------
    field: str
        Specify a related field to be accessed.

    Returns
    -------
    tf.Tensor
        A tensor representing the box collection or an associated field.

    Raises
    ------
    ValueError
        If invalid field.
    """
    if not self.has_field(field):
      raise ValueError('field {} does not exist.'.format(str(field)))
    return self._data[field]

  def set_field(self, field, value):
    """Sets the value of a field.

    Updates the field of a box_list with a given value.

    Parameters
    ----------
    field: str
        key of the field to set value.
    value: tf.Tensor
        The value to assign to the field.

    Raises
    ------
    ValueError
        If the box_list does not have specified field.
    """
    if not self.has_field(field):
      raise ValueError('field {} does not exist.'.format(str(field)))
    self._data[field] = value

  def get_center_coordinates_and_sizes(self, scope=None):
    """Computes the center coordinates, height and width of the boxes.

    Parameters
    ----------
    scope: str, optional
        Name scope of the function.

    Returns
    -------
    list
        A list of 4 1-D tensors [y_center, x_center, height, width].
    """
    with tf.name_scope(scope or 'GetCenterCoordinatesAndSizes'):
      box_corners = self.get()
      ymin, xmin, ymax, xmax = tf.unstack(tf.transpose(a=box_corners))
      width = xmax - xmin
      height = ymax - ymin
      y_center = ymin + height / 2.
      x_center = xmin + width / 2.
      return [y_center, x_center, height, width]

  def transpose_coordinates(self, scope=None):
    """Transpose the coordinate representation in a box list.

    Parameters
    ----------
    scope: str, optional
        Name scope of the function.
    """
    with tf.name_scope(scope or 'TransposeCoordinates'):
      ymin, xmin, ymax, xmax = tf.split(
          value=self.get(), num_or_size_splits=4, axis=1)
      self.set(tf.concat([xmin, ymin, xmax, ymax], 1))

  def as_tensor_dict(self, fields=None):
    """Retrieves specified fields as a dictionary of tensors.

    Parameters
    ----------
    fields: list of str, optional
        List of fields to return in the dictionary. If None (default), all
        fields are returned.

    Returns
    -------
    dict of tf.Tensor
        A dictionary of tensors specified by fields.

    Raises
    ------
    ValueError
        If specified field is not contained in box list.
    """
    tensor_dict = {}
    if fields is None:
      fields = self.get_all_fields()
    for field in fields:
      if not self.has_field(field):
        raise ValueError('Box list must contains all specified fields.')
      tensor_dict[field] = self.get_field(field)
    return tensor_dict
