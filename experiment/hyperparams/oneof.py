"""Config class that supports oneof functionality."""

from typing import Optional

import dataclasses
from experiment.hyperparams import base_config


@dataclasses.dataclass
class OneOfConfig(base_config.Config):
  """Configuration for configuration with one of feature.

  :attribute type: str for name of the field to select.
  """
  type: Optional[str] = None

  def as_dict(self):
    """Returns a dict representation of OneOfConfig.

    For the nested base_config.Config, a nested dict will be returned.

    :return: a dict representation of OneOfConfig.
    """
    if self.type is None:
      return {'type': None}
    elif self.__dict__['type'] not in self.__dict__:
      raise ValueError('type: {!r} is not a valid key!'.format(
          self.__dict__['type']))
    else:
      chosen_type = self.type
      chosen_value = self.__dict__[chosen_type]
      return {'type': self.type, chosen_type: self._export_config(chosen_value)}

  def get(self):
    """Returns selected config based on the value of type.

    :return: the selected config based on the value of type and None if type is
        not set.
    """
    chosen_type = self.type
    if chosen_type is None:
      return None
    if chosen_type not in self.__dict__:
      raise ValueError('type: {!r} is not a valid key!'.format(self.type))
    return self.__dict__[chosen_type]
