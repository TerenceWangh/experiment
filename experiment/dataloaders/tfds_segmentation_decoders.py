"""TFDS Semantic Segmentation decoders."""
import tensorflow as tf
from experiment.dataloaders import Decoder


class CityScapesDecorder(Decoder):
  def __init__(self):
    # Original labels to trainable labels map, 255 is the ignore class.
    # pylint: disable=bad-whitespace
    self._label_map = {
        -1: 255,
        0 : 255,
        1 : 255,
        2 : 255,
        3 : 255,
        4 : 255,
        5 : 255,
        6 : 255,
        7 : 0,
        8 : 1,
        9 : 255,
        10: 255,
        11: 2,
        12: 3,
        13: 4,
        14: 255,
        15: 255,
        16: 255,
        17: 5,
        18: 255,
        19: 6,
        20: 7,
        21: 8,
        22: 9,
        23: 10,
        24: 11,
        25: 12,
        26: 13,
        27: 14,
        28: 15,
        29: 255,
        30: 255,
        31: 16,
        32: 17,
        33: 18,
    }
    # pylint: enable=bad-whitespace

  def decode(self, serialized_example):
    # Convert labels according to the self._label_map
    label = serialized_example['segmentation_label']
    for original_label in self._label_map:
      label = tf.where(label == original_label,
                       self._label_map[original_label] * tf.ones_like(label),
                       label)

    # pylint: disable=bad-whitespace
    sample_dict = {
        'image/encoded'                   :
            tf.io.encode_jpeg(serialized_example['image_left'], quality=100),
        'image/height'                    :
            serialized_example['image_left'].shape[0],
        'image/width'                     :
            serialized_example['image_left'].shape[1],
        'image/segmentation/class/encoded': tf.io.encode_png(label),
    }
    # pylint: enable=bad-whitespace
    return sample_dict


# pylint: disable=bad-whitespace
TFDS_ID_TO_DECODER_MAP = {
    'cityscapes'                            : CityScapesDecorder,
    'cityscapes/semantic_segmentation'      : CityScapesDecorder,
    'cityscapes/semantic_segmentation_extra': CityScapesDecorder,
}
# pylint: enable=bad-whitespace
