"""Factory for getting TF-Vision input readers."""

from experiment.common import dataset_fn as dataset_fn_utils
from experiment.core import DataConfig
from experiment.core import InputReader
from experiment.dataloaders import CombinationDatasetInputReader


def input_reader_generator(params: DataConfig, **kwargs) -> InputReader:
  """Instantiates an input reader class according to the params.

  Parameters
  ----------
  params : DataConfig
      The parameters of input reader.
  kwargs : dict
      Additional arguments passed to input reader initialization.

  Returns
  -------
  InputReader
      An InputReader object.
  """
  if params.is_training and params.get('pseudo_label_data', False):
    return CombinationDatasetInputReader(
        params,
        pseudo_label_dataset_fn=dataset_fn_utils.pick_dataset_fn(
            params.pseudo_label_data.file_type),
        **kwargs)
  return InputReader(params, **kwargs)
