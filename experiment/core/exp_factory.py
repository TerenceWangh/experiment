"""Experiment factory methods."""
from experiment.core import config_definitions as cfg
from experiment.core import registry


_REGISTERED_CONFIGS = {}


def register_config_factory(name):
  """Register ExperimentConfig factory method."""
  return registry.register(_REGISTERED_CONFIGS, name)


def get_exp_config(exp_name: str) -> cfg.ExperimentConfig:
  """Looks up the `ExperimentConfig` according to the `exp_name`."""
  exp_creator = registry.lookup(_REGISTERED_CONFIGS, exp_name)
  return exp_creator()
