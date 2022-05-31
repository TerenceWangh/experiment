"""The generic parser interface."""
from abc import ABCMeta
from abc import abstractmethod


class Parser:
  """Parses data and produces tensors to be consumed by models."""

  __metaclass__ = ABCMeta

  @abstractmethod
  def _parse_train_data(self, decoded_tensors):
    """Generates images and labels that are usable for model training.

    Parameters
    ----------
    decoded_tensors : dict
        A dict of Tensors produced by the decoder.

    Returns
    -------
    images : tf.Tensor
        The image tensor.
    labels : dict
        The dict of tensors that contains labels.
    """

  @abstractmethod
  def _parse_eval_data(self, decoded_tensors):
    """Generates images and labels that are usable for model evaluation.

    Parameters
    ----------
    decoded_tensors : dict
        A dict of Tensors produced by the decoder.

    Returns
    -------
    images : tf.Tensor
        The image tensor.
    labels : dict
        The dict of tensors that contains labels.
    """

  def parse_fn(self, is_training):
    """Returns a parse fn that reads and parses raw tensors from the decoder.

    Parameters
    ----------
    is_training : bool
        Whether it is in training mode.

    Returns
    -------
    callable
        `callable` that takes the serialized example and generate the images,
        labels tuple where labels is a dict of Tensors that contains labels.
    """

    def parse(decoded_tensors):
      """parses the serialized example data."""
      if is_training:
        return self._parse_train_data(decoded_tensors)
      return self._parse_eval_data(decoded_tensors)

    return parse

  @classmethod
  def inference_fn(cls, input):
    """Parses inputs for predictions.

    Parameters
    ----------
    input : dict or tf.Tensor
        The input data.

    Returns
    -------
    tf.Tensor
        The input tensor to the model.
    """
