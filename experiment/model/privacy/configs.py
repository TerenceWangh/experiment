"""Configs for differential privacy."""
import dataclasses
from experiment.hyperparams import base_config


@dataclasses.dataclass
class DifferentialPrivacyConfig(base_config.Config):
  # Applied to the gradients
  # Setting to a large number so nothing is clipped.
  clipping_norm: float = 1000000000.0
  noise_multiplier: float = 0.0
