"""TFDS detection decoders."""

import tensorflow as tf
from experiment.dataloaders import Decoder


class COCODecoder(Decoder):
  """A tf.Example decoder for tfds coco datasets."""

  def decode(self, serialized_example):
    decoded_tensors = {
      'source_id'           : tf.strings.as_string(
          serialized_example['image/id']),
      'image'                : serialized_example['image'],
      'height'               : tf.cast(tf.shape(serialized_example['image'])[0],
                                      tf.int64),
      'width'                : tf.cast(tf.shape(serialized_example['image'])[1],
                                      tf.int64),
      'ground_truth_classes' : serialized_example['objects']['label'],
      'ground_truth_is_crowd': serialized_example['objects']['is_crowd'],
      'ground_truth_area'    : tf.cast(
          serialized_example['objects']['area'], tf.float32),
      'ground_truth_boxes'   : serialized_example['objects']['bbox'],
    }
    return decoded_tensors


TFDS_ID_TO_DECODER_MAP = {
  'coco/2017': COCODecoder,
  'coco/2014': COCODecoder,
  'coco'     : COCODecoder
}
