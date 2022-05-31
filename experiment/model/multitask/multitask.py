"""Experimental MultiTask base class for multi-task training/evaluation."""

from experiment.core import config_definitions
from experiment import optimization
from experiment.model.privacy import configs as dp_configs

OptimizationConfig = optimization.OptimizationConfig
RuntimeConfig = config_definitions.RuntimeConfig
DifferentialPrivacyConfig = dp_configs.DifferentialPrivacyConfig
