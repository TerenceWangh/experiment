"""Contains definitions of EfficientNet Networks.

References
----------
https://arxiv.org/pdf/1905.11946.pdf
"""
import math
from typing import Any, List, Tuple
import tensorflow as tf

from experiment.model import tf_utils
from experiment.model.backbones import factory
from experiment.model.layers import nn_blocks
from experiment.model.layers import nn_layers
