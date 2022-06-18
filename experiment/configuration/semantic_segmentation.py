"""Semantic segmentation configuration definition."""
import dataclasses
from typing import List, Optional, Union
import os
import numpy as np

from experiment.core import config_definitions as cfg
from experiment.configuration.common import DataDecoder


@dataclasses.dataclass
class DataConfig(cfg.DataConfig):
  output_size: List[int] = dataclasses.field(default_factory=list)
  # If crop_size is specified, image will be resized first to
  # output_size, then crop of size crop_size will be cropped.
  crop_size: List[int] = dataclasses.field(default_factory=list)
  input_path: str = ''
  global_batch_size: int = 0
  is_training: bool = True
  dtype: str = 'float32'
  shuffle_buffer_size: int = 1000
  cycle_length: int = 10
  # If resize_eval_groundtruth is set to False, original image sizes are used
  # for eval. In that case, groundtruth_padded_size has to be specified too to
  # allow for batching the variable input sizes of images.
  resize_eval_groundtruth: bool = True
  groundtruth_padded_size: List[int] = dataclasses.field(default_factory=list)
  aug_scale_min: float = 1.0
  aug_scale_max: float = 1.0
  aug_rand_hflip: bool = True
  preserve_aspect_ratio: bool = True
  aug_policy: Optional[str] = None
  drop_remainder: bool = True
  file_type: str = 'tfrecord'
  decoder: Optional[DataDecoder] = DataDecoder()