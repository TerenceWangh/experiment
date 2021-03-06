"""Utility functions to create tf.Example and tf.SequnceExample for test.

Example: video classification end-to-end test
i.e. from reading input file to train and eval.

>>> class FooTrainTest(tf.test.TestCase):
>>>  def setUp(self):
>>>    super(TrainTest, self).setUp()
>>>    # Write the fake tf.train.SequenceExample to file for test.
>>>    data_dir = os.path.join(self.get_temp_dir(), 'data')
>>>    tf.io.gfile.makedirs(data_dir)
>>>    self._data_path = os.path.join(data_dir, 'data.tfrecord')
>>>    examples = [
>>>        tfexample_utils.make_video_test_example(
>>>            image_shape=(36, 36, 3),
>>>            audio_shape=(20, 128),
>>>            label=random.randint(0, 100)) for _ in range(2)
>>>    ]
>>>    tf_example_utils.dump_to_tfrecord(self._data_path, tf_examples=examples)
>>>  def test_foo(self):
>>>    dataset = tf.data.TFRecordDataset(self._data_path)
>>>    ...
"""
import io
from typing import Sequence, Union

import numpy as np
from PIL import Image
import tensorflow as tf

IMAGE_KEY = 'image/encoded'
CLASSIFICATION_LABEL_KEY = 'image/class/label'
DISTILATION_LABEL_KEY = 'image/class/soft_labels'
LABEL_KEY = 'clip/label/index'
AUDIO_KEY = 'features/audio'
DUMP_SOURCE_ID = b'123'


def encode_image(image_array: np.array, fmt: str) -> bytes:
  """Encodes the image using PIL.

  Parameters
  ----------
  image_array : np.array
      The input image to be encoded.
  fmt : str
      The format to use is determined from the filename extension.

  Returns
  -------
  bytes
      The encoded image.
  """
  image = Image.fromarray(image_array)
  with io.BytesIO() as output:
    image.save(output, format=fmt)
    return output.getvalue()


def make_image_bytes(shape: Sequence[int], fmt: str = 'JPEG') -> bytes:
  """Generates image and return bytes in specified format.

  Parameters
  ----------
  shape : list of int
      The shape of image to be generated.
  fmt : str, default 'JPEG'
      The format to use is determined from the filename extension.

  Returns
  -------
  bytes
      The generated image encoded with `fmt`.
  """
  random_image = np.random.randint(0, 256, size=shape, dtype=np.uint8)
  return encode_image(random_image, fmt=fmt)


def put_int64_to_context(seq_example: tf.train.SequenceExample,
                         label: int = 0,
                         key: str = LABEL_KEY):
  """Put int64 or SequenceExample context with key.

  Parameters
  ----------
  seq_example : tf.train.SequenceExample
      The example to put the feature.
  label : int, default 0
      The value to be put into the example.
  key : str, default LABEL_KEY
      The key of the feature to put into seq_example.
  """
  seq_example.context.feature[key].int64_list.value[:] = [label]


def put_bytes_list_to_feature(seq_example: tf.train.SequenceExample,
                              raw_image_bytes: bytes,
                              key: str = IMAGE_KEY,
                              repeat_num: int = 2):
  """Puts bytes list to SequenceExample context with key.

  Parameters
  ----------
  seq_example : tf.train.SequenceExample
      The example to put the feature.
  raw_image_bytes : bytes
      The bytes of image to be put into the example.
  key : str, default LABEL_KEY
      The key of the feature to put into seq_example.
  repeat_num : int, default 2
      The number of repeat time.
  """
  for _ in range(repeat_num):
    seq_example.feature_lists.feature_list.get_or_create(
        key).feature.add().byte_list.value[:] = [raw_image_bytes]


def put_float_list_to_feature(seq_example: tf.train.SequenceExample,
                              value: Sequence[Sequence[float]],
                              key: str):
  """Puts float list to SequenceExample context with key.

  Parameters
  ----------
  seq_example : tf.train.SequenceExample
      The example to put the feature.
  value : list of float
      The value to be put into the example.
  key : str, default LABEL_KEY
      The key of the feature to put into seq_example.
  """
  for s in value:
    seq_example.feature_lists.feature_list.get_or_create(
        key).feature.add().float_list.value[:] = s


def make_video_test_example(image_shape: Sequence[int] = (263, 320, 3),
                            audio_shape: Sequence[int] = (10, 256),
                            label: int = 42):
  """Generates data for testing video models (inc. RGB, audio, & label).

  Parameters
  ----------
  image_shape : list of int, default (263, 320, 3)
      The shape of the image in video.
  audio_shape : list of int, default (10, 256)
      The shape of the audio in video.
  label : int, default 42
      The label of the test example.

  Returns
  -------
  tf.train.SequenceExample
      The example contains image, audio and label information.
  """
  raw_image_bytes = make_image_bytes(shape=image_shape)
  random_audio = np.random.normal(size=audio_shape).tolist()
  seq_example = tf.train.SequenceExample()
  put_int64_to_context(seq_example, label=label, key=LABEL_KEY)
  put_bytes_list_to_feature(seq_example, raw_image_bytes, key=IMAGE_KEY,
                            repeat_num=4)
  put_float_list_to_feature(seq_example, value=random_audio, key=AUDIO_KEY)
  return seq_example


def dump_to_tfrecord(record_file: str,
                     tf_examples: Sequence[Union[tf.train.Example,
                                                 tf.train.SequenceExample]]):
  """Writes serialized Example to TFRecord file with path.

  Parameters
  ----------
  record_file : str
      The path of TFRecord to be written.
  tf_examples : list of tf.train.Example or list of tf.train.SequenceExample
      The example.
  """
  with tf.io.TFRecordWriter(record_file) as writer:
    for tf_example in tf_examples:
      writer.write(tf_example.SerializeToString())


def _encode_image(image_array: np.ndarray, fmt: str) -> bytes:
  """Util function to encode an image."""
  image = Image.fromarray(image_array)
  with io.BytesIO() as output:
    image.save(output, format=fmt)
    return output.getvalue()


def create_classification_example(
    image_height: int,
    image_width: int,
    image_format: str = 'JPEG',
    is_multilabel: bool = False) -> tf.train.Example:
  """Creates image and labels for image classification input pipeline."""
  image = _encode_image(
      np.uint8(np.random.rand(image_height, image_width, 3) * 255),
      fmt=image_format)
  labels = [0, 1] if is_multilabel else [0]
  # pylint: disable=bad-whitespace
  serialized_example = tf.train.Example(
      features=tf.train.Features(
          feature={
              IMAGE_KEY               : (tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=[image]))),
              CLASSIFICATION_LABEL_KEY: (tf.train.Feature(
                  int64_list=tf.train.Int64List(value=labels))),
          })).SerializeToString()
  # pylint: enable=bad-whitespace
  return serialized_example


def create_distillation_example(
    image_height: int,
    image_width: int,
    num_labels: int,
    image_format: str = 'JPEG') -> tf.train.Example:
  """Creates image and labels for image classification with distillation."""
  image = _encode_image(
      np.uint8(np.random.rand(image_height, image_width, 3) * 255),
      fmt=image_format)
  soft_labels = [0.6] * num_labels
  # pylint: disable=bad-whitespace
  serialized_example = tf.train.Example(
      features=tf.train.Features(
          feature={
              IMAGE_KEY            : (tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=[image]))),
              DISTILATION_LABEL_KEY: (tf.train.Feature(
                  float_list=tf.train.FloatList(value=soft_labels))),
          })).SerializeToString()
  # pylint: enable=bad-whitespace
  return serialized_example


def create_3d_image_test_example(image_height: int, image_width: int,
                                 image_volume: int,
                                 image_channel: int) -> tf.train.Example:
  """Creates 3D image and label."""
  images = np.random.rand(image_height, image_width, image_volume,
                          image_channel)
  images = images.astype(np.float32)

  labels = np.random.randint(
      low=2, size=(image_height, image_width, image_volume, image_channel))
  labels = labels.astype(np.float32)

  # pylint: disable=bad-whitespace
  feature = {
      IMAGE_KEY               : (tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[images.tobytes()]))),
      CLASSIFICATION_LABEL_KEY: (tf.train.Feature(
          bytes_list=tf.train.BytesList(value=[labels.tobytes()])))
  }
  # pylint: enable=bad-whitespace
  return tf.train.Example(features=tf.train.Features(feature=feature))


def create_detection_test_example(image_height: int,
                                  image_width: int,
                                  image_channel: int,
                                  num_instances: int) -> tf.train.Example:
  """Creates and returns a test example containing box and mask annotations.

  Parameters
  ----------
  image_height : int
      The height of test image.
  image_width : int
      The width of test image.
  image_channel : int
      The channel of test image.
  num_instances : int
      The number of object instances per image.

  Returns
  -------
  tf.train.Example
      A tf.train.Example for testing.
  """
  image = make_image_bytes([image_height, image_width, image_channel])
  if num_instances == 0:
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    labels = []
    areas = []
    is_crowds = []
    masks = []
    labels_text = []
  else:
    xmins = list(np.random.rand(num_instances))
    xmaxs = list(np.random.rand(num_instances))
    ymins = list(np.random.rand(num_instances))
    ymaxs = list(np.random.rand(num_instances))
    labels_text = [b'class_1'] * num_instances
    labels = list(np.random.randint(100, size=num_instances))
    areas = [(xmax - xmin) * (ymax - ymin) * image_height * image_width
             for xmin, xmax, ymin, ymax in zip(xmins, xmaxs, ymins, ymaxs)]
    is_crowds = [0] * num_instances
    masks = []
    for _ in range(num_instances):
      mask = make_image_bytes([image_height, image_width], fmt='PNG')
      masks.append(mask)
  # pylint: disable=bad-whitespace
  return tf.train.Example(
      features=tf.train.Features(
          feature={
              'image/encoded'           : (tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=[image]))),
              'image/source_id'         : (tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=[DUMP_SOURCE_ID]))),
              'image/height'            : (tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[image_height]))),
              'image/width'             : (tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[image_width]))),
              'image/object/bbox/xmin'  : (tf.train.Feature(
                  float_list=tf.train.FloatList(value=xmins))),
              'image/object/bbox/xmax'  : (tf.train.Feature(
                  float_list=tf.train.FloatList(value=xmaxs))),
              'image/object/bbox/ymin'  : (tf.train.Feature(
                  float_list=tf.train.FloatList(value=ymins))),
              'image/object/bbox/ymax'  : (tf.train.Feature(
                  float_list=tf.train.FloatList(value=ymaxs))),
              'image/object/class/label': (tf.train.Feature(
                  int64_list=tf.train.Int64List(value=labels))),
              'image/object/class/text' : (tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=labels_text))),
              'image/object/is_crowd'   : (tf.train.Feature(
                  int64_list=tf.train.Int64List(value=is_crowds))),
              'image/object/area'       : (tf.train.Feature(
                  float_list=tf.train.FloatList(value=areas))),
              'image/object/mask'       : (tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=masks))),
          }))
  # pylint: enable=bad-whitespace


def create_segmentation_test_example(image_height: int,
                                     image_width: int,
                                     image_channel: int) -> tf.train.Example:
  """Creates and returns a test example containing mask annotations.

  Parameters
  ----------
  image_height : int
      The height of test image.
  image_width : int
      The width of test image.
  image_channel : int
      The channel of test image.

  Returns
  -------
  tf.train.Example
      A tf.train.Example for testing.
  """
  image = make_image_bytes([image_height, image_width, image_channel])
  mask = make_image_bytes([image_height, image_width], fmt='PNG')
  # pylint: disable=bad-whitespace
  return tf.train.Example(
      features=tf.train.Features(
          feature={
              'image/encoded'                   : (tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=[image]))),
              'image/segmentation/class/encoded': (tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=[mask]))),
              'image/height'                    : (tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[image_height]))),
              'image/width'                     : (tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[image_width])))
          }))
  # pylint: enable=bad-whitespace
