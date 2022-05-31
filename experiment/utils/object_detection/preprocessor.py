"""Preprocess images and bounding boxes for detection.

We perform two sets of operations in preprocessing stage:
(a) operations that are applied to both training and testing data,
(b) operations that are applied only to training data for the purpose of
    data augmentation.

A preprocessing function receives a set of inputs,
e.g. an image and bounding boxes,
performs an operation on them, and returns them.

Some examples are: randomly cropping the image, randomly mirroring the image,
                   randomly changing the brightness, contrast, hue and
                   randomly jittering the bounding boxes.

The image is a rank 4 tensor: [1, height, width, channels] with
dtype=tf.float32. The groundtruth_boxes is a rank 2 tensor: [N, 4] where
in each row there is a box with [ymin xmin ymax xmax].
Boxes are in normalized coordinates meaning
their coordinate values range in [0, 1]

Important Note: In tensor_dict, images is a rank 4 tensor, but preprocessing
functions receive a rank 3 tensor for processing the image. Thus, inside the
preprocess function we squeeze the image to become a rank 3 tensor and then
we pass it to the functions. At the end of the preprocess we expand the image
back to rank 4.
"""
import numpy as np
import tensorflow as tf
from experiment.utils.object_detection import BoxList


def _flip_boxes_left_right(boxes):
  """Left-right flip the boxes.

  Parameters
  ----------
  boxes : tf.Tensor
      Rank 2 float32 tensor containing the bounding boxes -> [N, 4]. Boxes
      are in normalized form meaning their coordinates vary between [0, 1]. Each
      row is in the form of [ymin, xmin, ymax, xmax].

  Returns
  -------
  tf.Tensor
      The flipped boxes.
  """
  ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=1)
  flipped_xmin = tf.subtract(1.0, xmax)
  flipped_xmax = tf.subtract(1.0, xmin)
  flipped_boxes = tf.concat([ymin, flipped_xmin, ymax, flipped_xmax], 1)
  return flipped_boxes


def _flip_masks_left_right(masks):
  """Left-right flip masks.

  Parameters
  ----------
  masks : tf.Tensor
      Rank 3 float32 tensor with shape [num_instances, height, width]
      representing instance masks.

  Returns
  -------
  tf.Tensor
      rank 3 float32 tensor with shape [num_instances, height, width]
      representing instance masks.
  """
  return masks[:, :, ::-1]


def keypoint_flip_horizontal(keypoints,
                             flip_point,
                             flip_permutation,
                             scope = None):
  """Flips the keypoints horizontally around the flip_point.

  This operation flips the x coordinate for each keypoint around the flip_point
  and also permutes the keypoints in a manner specified by flip_permutation.

  Parameters
  ----------
  keypoints : tf.Tensor
      A tensor of shape [num_instances, num_key_points, 2].
  flip_point : float
      The tensor representing the x coordinate to flip the keypoints around.
  flip_permutation : tf.Tensor
      Rank 1 int32 tensor containing the keypoint flip permutation. This
      specifies the mapping from original keypoint indices to the flipped
      keypoint indices. This is used primarily for keypoints that are not
      reflection invariant. E.g. Suppose there are 3 keypoints representing
      ['head', 'right_eye', 'left_eye'], then a logical choice for
      flip_permutation might be [0, 2, 1] since we want to swap the 'left_eye'
      and 'right_eye' after a horizontal flip.
  scope : str, optional
      The name scope of the function.

  Returns
  -------
  tf.Tensor
      A tensor of shape [num_instances, num_keypoints, 2]
  """
  with tf.name_scope(scope or 'FlipHorizontal'):
    keypoints = tf.transpose(a=keypoints, perm=[1, 0, 2])
    keypoints = tf.gather(keypoints, flip_permutation)
    v, u = tf.split(value=keypoints, num_or_size_splits=2, axis=2)
    u = flip_point * 2.0 - u
    new_keypoints = tf.concat([v, u], 2)
    new_keypoints = tf.transpose(a=new_keypoints, perm=[1, 0, 2])
    return new_keypoints


def keypoint_change_coordinate_frame(keypoints, window, scope = None):
  """Changes coordinate frame of the keypoints to be relative to window's frame.

  Given a window of the form [y_min, x_min, y_max, x_max], changes keypoint
  coordinates from keypoints of shape [num_instances, num_keypoints, 2]
  to be relative to this window.

  An example use case is data augmentation: where we are given groundtruth
  keypoints and would like to randomly crop the image to some window. In this
  case we need to change the coordinate frame of each groundtruth keypoint to be
  relative to this new window.

  Parameters
  ----------
  keypoints : tf.Tensor
      A tensor of shape [num_instance, num_keypoints, 2].
  window : tf.Tensor
      A tensor of shape [4] representing the [y_min, x_min, y_max, x_max] window
      we should change the coordinate frame to.
  scope : str, optional
      The name scope of the function.

  Returns
  -------
  tf.Tensor
      A tensor of shape [num_instance, num_keypoints, 2].
  """
  with tf.name_scope(scope or 'ChangeCoordinateFrame'):
    win_h = window[2] - window[0]
    win_w = window[3] - window[1]
    new_keypoints = box_list_ops.scale(
        keypoints - [window[0], window[1]], 1.0 / win_h, 1.0 / win_w)
    return new_keypoints


def keypoint_prune_outside_window(keypoints, window, scope = None):
  """Prunes keypoints that fall outside a given window.

  This function replaces keypoints that fall outside the given window with nan.
  See also clip_to_window which clips any keypoints that fall outside the given
  window.

  Parameters
  ----------
  keypoints : tf.Tensor
      A tensor of shape [num_instance, num_keypoints, 2].
  window : tf.Tensor
      A tensor of shape [4] representing the [y_min, x_min, y_max, x_max]
      window outside of which the op should prune the keypoints.
  scope : str, optional
      The name scope of the function.

  Returns
  -------
  tf.Tensor
      A tensor of shape [num_instance, num_keypoints, 2].
  """
  with tf.name_scope(scope or 'PruneOutsideWindow'):
    y, x = tf.split(value=keypoints, num_or_size_splits=2, axis=2)
    win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)
    valid_indices = tf.logical_and(
        tf.logical_and(y >= win_y_min, y <= win_y_max),
        tf.logical_and(x >= win_x_min, x <= win_x_max))
    new_y = tf.where(valid_indices, y, np.nan * tf.ones_like(y))
    new_x = tf.where(valid_indices, x, np.nan * tf.ones_like(x))
    new_keypoints = tf.concat([new_y, new_x], 2)
    return new_keypoints


def random_horizontal_flip(image,
                           boxes = None,
                           masks = None,
                           keypoints = None,
                           keypoint_flip_permutation = None,
                           seed = None):
  """Randomly flips the image and detections horizontally.

  The probability of flipping the image is 50%.

  Parameters
  ----------
  image : tf.Tensor
      Rank 3 float32 tensor with shape [height, width, channels].
  boxes : tf.Tensor, optional
      Rank 2 float32 tensor with shape [N, 4] containing the bounding boxes.
      Boxes are in normalized form meaning their coordinates vary between
      [0, 1]. Each row is in the form of [ymin, xmin, ymax, xmax].
  masks : tf.Tensor, optional
      Rank 3 float32 tensor with shape [num_instances, height, width] containing
      instance masks. The masks are of the same height, width as the input
      `image`.
  keypoints : tf.Tensor, optional
      Rank 3 float32 tensor with shape [num_instances, num_keypoints, 2]. The
      keypoints are in y-x normalized coordinates.
  keypoint_flip_permutation : tf.Tensor, optional
      Rank 1 int32 tensor containing the keypoint flip permutation.
  seed : int
      Random seed.

  Returns
  -------
  image : tf.Tensor
      The image which is the same shape as input image.
  boxes : tf.Tensor
      Rank 2 float32 tensor containing the bounding boxes -> [N, 4].
      Boxes are in normalized form meaning their coordinates vary
      between [0, 1].
  masks : tf.Tensor
      Rank 3 float32 tensor with shape [num_instances, height, width]
      containing instance masks.
  keypoints : tf.Tensor
      Rank 3 float32 tensor with shape [num_instances, num_keypoints, 2]

  Raises
  ------
  ValueError
      If keypoints are provided but keypoint_flip_permutation is not.
  """

  def _flip_image(image):
    image_flipped = tf.image.flip_left_right(image)
    return image_flipped

  if keypoints is not None and keypoint_flip_permutation is None:
    raise ValueError('keypoints are provided but keypoints_flip_permutation is '
                     'not provided.')

  with tf.name_scope('RandomHorizontalFlip'):
    result = []
    # random variable defining whether to do flip or not
    do_a_flip_random = tf.greater(tf.random.uniform([], seed=seed), 0.5)
    # flip image
    image = tf.cond(pred=do_a_flip_random,
                    true_fn=lambda: _flip_image(image),
                    false_fn=lambda: image)
    result.append(image)

    if masks is not None:
      masks = tf.cond(pred=do_a_flip_random,
                      true_fn=lambda: _flip_masks_left_right(masks),
                      false_fn=lambda: masks)
      result.append(masks)

    if keypoints is not None and keypoint_flip_permutation is not None:
      permutation = keypoint_flip_permutation
      keypoints = tf.cond(
          pred=do_a_flip_random,
          true_fn=lambda: keypoint_flip_horizontal(keypoints, 0.5, permutation),
          false_fn=lambda: keypoints)
      result.append(keypoints)
    return tuple(result)


def _compute_new_static_size(image, min_dimension, max_dimension):
  """Compute new static shape for resize_to_range method."""
  image_shape = image.get_shape().as_list()
  orig_height = image_shape[0]
  orig_width = image_shape[1]
  num_channels = image_shape[2]
  orig_min_dim = min(orig_height, orig_width)
  # Calculates the larger of the possible sizes
  large_scale_factor = min_dimension / float(orig_min_dim)
  # Scaling orig_(height|width) by large_scale_factor will make the smaller
  # dimension equal to min_dimension, save for floating point rounding errors.
  # For reasonably-sized images, taking the nearest integer will reliably
  # eliminate this error.
  large_height = int(round(orig_height * large_scale_factor))
  large_width = int(round(orig_width * large_scale_factor))
  large_size = [large_height, large_width]
  if max_dimension:
    # Calculates the smaller of the possible sizes, use that if the larger
    # is too big.
    orig_max_dim = max(orig_height, orig_width)
    small_scale_factor = max_dimension / float(orig_max_dim)
    # Scaling orig_(height|width) by small_scale_factor will make the larger
    # dimension equal to max_dimension, save for floating point rounding
    # errors. For reasonably-sized images, taking the nearest integer will
    # reliably eliminate this error.
    small_height = int(round(orig_height * small_scale_factor))
    small_width = int(round(orig_width * small_scale_factor))
    small_size = [small_height, small_width]
    new_size = large_size
    if max(large_size) > max_dimension:
      new_size = small_size
  else:
    new_size = large_size
  return tf.constant(new_size + [num_channels])


def _compute_new_dynamic_size(image, min_dimension, max_dimension):
  """Compute new dynamic shape for resize_to_range method."""
  image_shape = tf.shape(input=image)
  orig_height = tf.cast(image_shape[0], dtype=tf.float32)
  orig_width = tf.cast(image_shape[1], dtype=tf.float32)
  num_channels = image_shape[2]
  orig_min_dim = tf.minimum(orig_height, orig_width)
  # Calculates the larger of the possible sizes
  min_dimension = tf.constant(min_dimension, dtype=tf.float32)
  large_scale_factor = min_dimension / orig_min_dim
  # Scaling orig_(height|width) by large_scale_factor will make the smaller
  # dimension equal to min_dimension, save for floating point rounding errors.
  # For reasonably-sized images, taking the nearest integer will reliably
  # eliminate this error.
  large_height = tf.cast(
      tf.round(orig_height * large_scale_factor), dtype=tf.int32)
  large_width = tf.cast(
      tf.round(orig_width * large_scale_factor), dtype=tf.int32)
  large_size = tf.stack([large_height, large_width])
  if max_dimension:
    # Calculates the smaller of the possible sizes, use that if the larger
    # is too big.
    orig_max_dim = tf.maximum(orig_height, orig_width)
    max_dimension = tf.constant(max_dimension, dtype=tf.float32)
    small_scale_factor = max_dimension / orig_max_dim
    # Scaling orig_(height|width) by small_scale_factor will make the larger
    # dimension equal to max_dimension, save for floating point rounding
    # errors. For reasonably-sized images, taking the nearest integer will
    # reliably eliminate this error.
    small_height = tf.cast(
        tf.round(orig_height * small_scale_factor), dtype=tf.int32)
    small_width = tf.cast(
        tf.round(orig_width * small_scale_factor), dtype=tf.int32)
    small_size = tf.stack([small_height, small_width])
    new_size = tf.cond(
        pred=tf.cast(tf.reduce_max(input_tensor=large_size), dtype=tf.float32) >
        max_dimension,
        true_fn=lambda: small_size,
        false_fn=lambda: large_size)
  else:
    new_size = large_size
  return tf.stack(tf.unstack(new_size) + [num_channels])


def resize_to_range(image,
                    masks = None,
                    min_dimension = None,
                    max_dimension = None,
                    method = tf.image.ResizeMethod.BILINEAR,
                    align_corners = False,
                    pad_to_max_dimension = False):
  """Resizes an image so its dimensions are within the provided value.

  The output size can be described by two cases:
  1. If the image can be rescaled so its minimum dimension is equal to the
     provided value without the other dimension exceeding max_dimension,
     then do so.
  2. Otherwise, resize so the largest dimension is equal to max_dimension.

  Parameters
  ----------
  image : tf.Tensor
      A 3D tensor of shape [height, width, channels]
  masks : tf.Tensor, optional
      Rank 3 float32 tensor with shape [num_instances, height, width] containing
      instance masks.
  min_dimension : int, optional
      The desired size of the smaller image dimension.
  max_dimension : int, optional
      The maximum allowed size of the larger image dimension.
  method : tf.image.ResizeMethod, default tf.image.ResizeMethod.BILINEAR.
      The interpolation method used in resizing.
  align_corners : bool, default False
      If true, exactly align all 4 corners of the input and output.
  pad_to_max_dimension : bool, default False
      Whether to resize the image and pad it with zeros so the resulting image
      is of the spatial size [max_dimension, max_dimension]. If masks are
      included they are padded similarly.

  Returns
  -------
  resized_image : tf.Tensor
      A 3D tensor of shape [new_height, new_width, channels], where the image
      has been resized (with bilinear interpolation) so that
      `min(new_height, new_width) == min_dimension` or
      `max(new_height, new_width) == max_dimension`.
  resized_masks : tf.Tensor
      If masks is not None, also outputs masks. A 3D tensor of shape
      [num_instances, new_height, new_width].
  resized_image_shape : tf.Tensor
      A 1D tensor of shape [3] containing shape of the resized image.

  Raises
  ------
  ValueError
      If the image is not a 3D tensor.
  """
  if len(image.get_shape()) != 3:
    raise ValueError('Image should be 3D tensor')

  with tf.name_scope('ResizeToRange'):
    if image.get_shape().is_fully_defined():
      new_size = _compute_new_static_size(image, min_dimension, max_dimension)
    else:
      new_size = _compute_new_dynamic_size(image, min_dimension, max_dimension)
    new_image = tf.image.resize(image, new_size[:-1], method=method)

    if pad_to_max_dimension:
      new_image = tf.image.pad_to_bounding_box(new_image, 0, 0, max_dimension,
                                               max_dimension)

    result = [new_image]
    if masks is not None:
      new_masks = tf.expand_dims(masks, 3)
      new_masks = tf.image.resize(
          new_masks,
          new_size[:-1],
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      new_masks = tf.squeeze(new_masks, 3)
      if pad_to_max_dimension:
        new_masks = tf.image.pad_to_bounding_box(new_masks, 0, 0, max_dimension,
                                                 max_dimension)
      result.append(new_masks)

    result.append(new_size)
    return result


def _copy_extra_fields(box_list_to_copy_to, box_list_to_copy_from):
  """Copies the extra fields of boxlist_to_copy_from to boxlist_to_copy_to.

  Parameters
  ----------
  box_list_to_copy_to : BoxList
      BoxList to which extra fields are copied.
  box_list_to_copy_from : BoxList
      BoxList from which fields are copied.

  Returns
  -------
  BoxList
      The `box_list_to_copy_to` with extra fields.
  """
  for field in box_list_to_copy_from.get_extra_fields():
    box_list_to_copy_to.add_field(field, box_list_to_copy_from.get_field(field))
  return box_list_to_copy_to


def box_list_scale(box_list, y_scale, x_scale, scope = None):
  """Scale box coordinates in x and y dimensions.

  Parameters
  ----------
  box_list : BoxList
      BoxList holding N boxes.
  y_scale : float
      The scale of y axis.
  x_scale : float
      The scale of x axis.
  scope : str, optional
      The name scope of the function.

  Returns
  -------
  BoxList
      BoxList holding N boxes.
  """
  with tf.name_scope(scope or 'BoxListScale'):
    y_scale = tf.cast(y_scale, tf.float32)
    x_scale = tf.cast(x_scale, tf.float32)
    y_min, x_min, y_max, x_max = tf.split(
        value=box_list.get(), num_or_size_splits=4, axis=1)
    y_min = y_scale * y_min
    y_max = y_scale * y_max
    x_min = x_scale * x_min
    x_max = x_scale * x_max
    scaled_box_list = BoxList(
        tf.concat([y_min, x_min, y_max, x_max], 1))
    return _copy_extra_fields(scaled_box_list, box_list)


def keypoint_scale(keypoints, y_scale, x_scale, scope=None):
  """Scales keypoint coordinates in x and y dimensions.

  Parameters
  ----------
  keypoints : tf.Tensor
      A tensor of shape [num_instances, num_keypoints, 2].
  y_scale : float
      The scale of y axis.
  x_scale : float
      The scale of x axis.
  scope : str, optional
      The name scope of the function.

  Returns
  -------
  tf.Tensor
      A tensor of shape [num_instances, num_keypoints, 2].
  """
  with tf.name_scope(scope or 'Scale'):
    y_scale = tf.cast(y_scale, tf.float32)
    x_scale = tf.cast(x_scale, tf.float32)
    new_keypoints = keypoints * [[[y_scale, x_scale]]]
    return new_keypoints


def scale_boxes_to_pixel_coordinates(image, boxes, keypoints=None):
  """Scales boxes from normalized to pixel coordinates.

  Parameters
  ----------
  image : tf.Tensor
      A 3D float32 tensor of shape [height, width, channels].
  boxes : tf.Tensor
      A 2D float32 tensor of shape [num_boxes, 4] containing the bounding
      boxes in normalized coordinates. Each row is of the form
      [ymin, xmin, ymax, xmax].
  keypoints : tf.Tensor, optional
      Rank 3 float32 tensor with shape [num_instances, um_keypoints, 2].
      The keypoints are in y-x normalized coordinates.

  Returns
  -------
  image : tf.Tensor
      unchanged input image.
  scaled_boxes : tf.Tensor
      A 2D float32 tensor of shape [num_boxes, 4] containing the bounding boxes
      in pixel coordinates.
  scaled_keypoints : tf.Tensor
      A 3D float32 tensor with shape [num_instances, num_keypoints, 2]
      containing the keypoints in pixel coordinates.
  """
  box_list = BoxList(boxes)
  image_height = tf.shape(input=image)[0]
  image_width = tf.shape(input=image)[1]
  scaled_boxes = box_list_scale(box_list, image_height, image_width).get()
  result = [image, scaled_boxes]
  if keypoints is not None:
    scaled_keypoints = keypoint_scale(keypoints, image_height, image_width)
    result.append(scaled_keypoints)
  return tuple(result)
