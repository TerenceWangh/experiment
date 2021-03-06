"""Common Configurations."""

import dataclasses
from typing import List, Optional

from experiment.hyperparams import Config
from experiment.hyperparams import OneOfConfig
from experiment.core import config_definitions as cfg


@dataclasses.dataclass
class TFExampleDecoder(Config):
  regenerate_source_id: bool = False
  mask_binarize_threshold: Optional[float] = None


@dataclasses.dataclass
class TFExampleDecoderLabelMap(Config):
  regenerate_source_id: bool = False
  mask_binarize_threshold: Optional[float] = None
  label_map: str = ''


@dataclasses.dataclass
class DataDecoder(OneOfConfig):
  type: Optional[str] = 'simple_decoder'
  simple_decoder: TFExampleDecoder = TFExampleDecoder()
  label_map_decoder: TFExampleDecoderLabelMap = TFExampleDecoderLabelMap()


@dataclasses.dataclass
class RandAugment(Config):
  """Configuration for RandAugment."""
  num_layers: int = 2
  magnitude: float = 10
  cutout_const: float = 40
  translate_const: float = 10
  magnitude_std: float = 0.0
  prob_to_apply: Optional[float] = None
  exclude_ops: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class AutoAugment(Config):
  """Configuration for AutoAugment."""
  augmentation_name: str = 'v0'
  cutout_const: float = 100
  translate_const: float = 250


@dataclasses.dataclass
class RandomErasing(Config):
  """Configuration for RandomErasing."""
  probability: float = 0.25
  min_area: float = 0.02
  max_area: float = 1 / 3
  min_aspect: float = 0.3
  max_aspect = None
  min_count = 1
  max_count = 1
  trials = 10


@dataclasses.dataclass
class MixupAndCutmix(Config):
  """Configuration for mixup and cutmix."""
  mixup_alpha: float = .8
  cutmix_alpha: float = 1.
  prob: float = 1.0
  switch_prob: float = 0.5
  label_smoothing: float = 0.1


@dataclasses.dataclass
class Augmentation(OneOfConfig):
  """Configuration for input data augmentation.
  """
  type: Optional[str] = None
  randaug: RandAugment = RandAugment()
  autoaug: AutoAugment = AutoAugment()


@dataclasses.dataclass
class NormActivation(Config):
  activation: str = 'relu'
  use_sync_bn: bool = True
  norm_momentum: float = 0.99
  norm_epsilon: float = 0.001


@dataclasses.dataclass
class PseudoLabelDataConfig(cfg.DataConfig):
  """Psuedo Label input config for training."""
  input_path: str = ''
  data_ratio: float = 1.0  # Per-batch ratio of pseudo-labeled to labeled data.
  is_training: bool = True
  dtype: str = 'float32'
  shuffle_buffer_size: int = 10000
  cycle_length: int = 10
  aug_rand_hflip: bool = True
  aug_type: Optional[
      Augmentation] = None  # Choose from AutoAugment and RandAugment.
  file_type: str = 'tfrecord'

  # Keep for backward compatibility.
  aug_policy: Optional[str] = None  # None, 'autoaug', or 'randaug'.
  randaug_magnitude: Optional[int] = 10


@dataclasses.dataclass
class TFLitePostProcessingConfig(Config):
  max_detections: int = 200
  max_classes_per_detection: int = 5
  # Regular NMS run in a multi-class fashion and is slow. Setting it to False
  # uses class-agnostic NMS, which is faster.
  use_regular_nms: bool = False
  nms_score_threshold: float = 0.1
  nms_iou_threshold: float = 0.5
