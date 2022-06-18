"""Tests for maskrcnn_model.py."""

import os
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations

from experiment.model.segmentations import mrcnn
from experiment.model.backbones import resnet

