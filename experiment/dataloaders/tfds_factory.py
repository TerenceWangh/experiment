"""TFDS factory functions."""

from experiment.dataloaders import Decoder
from experiment.dataloaders import tfds_classification_decoders
from experiment.dataloaders import tfds_detection_decoders
from experiment.dataloaders import tfds_segmentation_decoders


def get_classification_decoder(tfds_name: str) -> Decoder:
  """Gets classification decoder.

  Parameters
  ----------
  tfds_name : str
      The name of the tfds classification decoder.

  Returns
  -------
  Decoder
      `Decoder` instance

  Raises
  ------
  ValueError
      If the tfds_name doesn't exist in available decoders.
  """
  if tfds_name in tfds_classification_decoders.TFDS_ID_TO_DECODER_MAP:
    decoder = tfds_classification_decoders.TFDS_ID_TO_DECODER_MAP[tfds_name]()
  else:
    raise ValueError('TFDS Classification {} is not support.'.format(tfds_name))
  return decoder


def get_detection_decoder(tfds_name: str) -> Decoder:
  """Gets detection decoder.

  Parameters
  ----------
  tfds_name : str
      The name of the tfds detection decoder.

  Returns
  -------
  Decoder
      `Decoder` instance

  Raises
  ------
  ValueError
      If the tfds_name doesn't exist in available decoders.
  """
  if tfds_name in tfds_detection_decoders.TFDS_ID_TO_DECODER_MAP:
    decoder = tfds_detection_decoders.TFDS_ID_TO_DECODER_MAP[tfds_name]()
  else:
    raise ValueError('TFDS detection {} is not support.'.format(tfds_name))
  return decoder


def get_segmentation_decoder(tfds_name: str) -> Decoder:
  """Gets segmentation decoder.

  Parameters
  ----------
  tfds_name : str
      The name of the tfds segmentation decoder.

  Returns
  -------
  Decoder
      `Decoder` instance

  Raises
  ------
  ValueError
      If the tfds_name doesn't exist in available decoders.
  """
  if tfds_name in tfds_segmentation_decoders.TFDS_ID_TO_DECODER_MAP:
    decoder = tfds_segmentation_decoders.TFDS_ID_TO_DECODER_MAP[tfds_name]()
  else:
    raise ValueError('TFDS segmentation {} is not support.'.format(tfds_name))
  return decoder
