"""Backbones configurations."""

import dataclasses
from typing import Optional, List

from experiment.hyperparams import Config
from experiment.hyperparams import OneOfConfig


@dataclasses.dataclass
class ResNet(Config):
  """ResNet config."""
  model_id: int = 50
  depth_multiplier: float = 1.0
  stem_type: str = 'v0'
  se_ratio: float = 0.0
  stochastic_depth_drop_rate: float = 0.0
  scale_stem: bool = True
  resnetd_shortcut: bool = False
  replace_stem_max_pool: bool = False
  bn_trainable: bool = True


@dataclasses.dataclass
class DilatedResNet(Config):
  """DilatedResNet config."""
  model_id: int = 50
  output_stride: int = 16
  multigrid: Optional[List[int]] = None
  stem_type: str = 'v0'
  last_stage_repeats: int = 1
  se_ratio: float = 0.0
  stochastic_depth_drop_rate: float = 0.0


@dataclasses.dataclass
class EfficientNet(Config):
  """EfficientNet config."""
  model_id: str = 'b0'
  se_ratio: float = 0.0
  stochastic_depth_drop_rate: float = 0.0


@dataclasses.dataclass
class MobileNet(Config):
  """Mobilenet config."""
  model_id: str = 'MobileNetV2'
  filter_size_scale: float = 1.0
  stochastic_depth_drop_rate: float = 0.0
  output_stride: Optional[int] = None
  output_intermediate_endpoints: bool = False


@dataclasses.dataclass
class SpineNet(Config):
  """SpineNet config."""
  model_id: str = '49'
  stochastic_depth_drop_rate: float = 0.0
  min_level: int = 3
  max_level: int = 7


@dataclasses.dataclass
class SpineNetMobile(Config):
  """SpineNet config."""
  model_id: str = '49'
  stochastic_depth_drop_rate: float = 0.0
  se_ratio: float = 0.2
  expand_ratio: int = 6
  min_level: int = 3
  max_level: int = 7
  # If use_keras_upsampling_2d is True, model uses UpSampling2D keras layer
  # instead of optimized custom TF op. It makes model be more keras style. We
  # set this flag to True when we apply QAT from model optimization toolkit
  # that requires the model should use keras layers.
  use_keras_upsampling_2d: bool = False


@dataclasses.dataclass
class RevNet(Config):
  """RevNet config."""
  # Specifies the depth of RevNet.
  model_id: int = 56


@dataclasses.dataclass
class MobileDet(Config):
  """Mobiledet config."""
  model_id: str = 'MobileDetCPU'
  filter_size_scale: float = 1.0


@dataclasses.dataclass
class Transformer(Config):
  """Transformer config"""
  mlp_dim: int = 1
  num_heads: int = 1
  num_layers: int = 1
  attention_dropout_rate: float = 0.0
  dropout_rate: float = 0.1


@dataclasses.dataclass
class VisionTransformer(Config):
  """VisionTransformer config."""
  model_name: str = 'vit-b16'
  # 'token' or 'gap'. If set to 'token', an extra classification token is added
  # to sequence.
  classifier: str = 'token'
  representation_size: int = 0
  hidden_size: int = 1
  patch_size: int = 16
  transformer: Transformer = Transformer()
  init_stochastic_depth_rate: float = 0.0
  original_init: bool = True


@dataclasses.dataclass
class Backbone(OneOfConfig):
  """Configuration for backbones.
  """
  type: Optional[str] = None
  resnet: ResNet = ResNet()
  dilated_resnet: DilatedResNet = DilatedResNet()
  revnet: RevNet = RevNet()
  efficientnet: EfficientNet = EfficientNet()
  spinenet: SpineNet = SpineNet()
  spinenet_mobile: SpineNetMobile = SpineNetMobile()
  mobilenet: MobileNet = MobileNet()
  mobiledet: MobileDet = MobileDet()
  vit: VisionTransformer = VisionTransformer()
