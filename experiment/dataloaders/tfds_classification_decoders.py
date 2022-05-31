"""TFDS Classification decoders."""
import tensorflow as tf
from experiment.dataloaders import Decoder


class TfdsClassificationDecoder(Decoder):
  """A tf.Example decoder for tfds classification datasets."""

  def decode(self, serialized_example):
    sample_dict = {
      'image/encoded': tf.io.encode_jpeg(serialized_example['image'],
                                         quality=100),
      'image/class/label': serialized_example['label'],
    }
    return sample_dict


TFDS_ID_TO_DECODER_MAP = {
  'mnist'        : TfdsClassificationDecoder,
  'cifar10'      : TfdsClassificationDecoder,
  'cifar100'     : TfdsClassificationDecoder,
  'imagenet2012' : TfdsClassificationDecoder,
}
