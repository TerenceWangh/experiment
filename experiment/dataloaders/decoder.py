"""The generic decoder interface."""

from abc import ABCMeta
from abc import abstractmethod


class Decoder:
  """Decodes the raw data into tensors."""

  __metaclass__ = ABCMeta

  @abstractmethod
  def decode(self, serialized_example):
    """Decodes the serialized example into tensors.

    Parameters
    ----------
    serialized_example : str
        A serialized string tensor that encodes the data.

    Returns
    -------
    dict
        A dict of Tensors.
    """
