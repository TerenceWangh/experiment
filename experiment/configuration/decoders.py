"""Decoders configurations."""
import dataclasses
from typing import List, Optional

from experiment.hyperparams import Config, OneOfConfig


@dataclasses.dataclass
class Identity(Config):
  """Identity config."""
  pass


@dataclasses.dataclass
class FPN(Config):
  """FPN config."""
  num_filters: int = 256
  fusion_type: str = 'sum'
  use_separable_conv: bool = False
  use_keras_layer: bool = False


@dataclasses.dataclass
class NASFPN(Config):
  """NASFPN config."""
  num_filters: int = 256
  num_repeats: int = 5
  use_seprable_conv: bool = False


@dataclasses.dataclass
class ASPP(Config):
  """ASPP config."""
  level: int = 4
  dilation_rate: List[int] = dataclasses.field(default_factory=list)
  dropout_rate: float = 0.0
  num_filters: int = 256
  use_depthwise_convolution: bool = False
  pool_kernel_size: Optional[List[int]] = None
  spp_layer_version: str = 'v1'
  output_tensor: bool = False


@dataclasses.dataclass
class Decoder(OneOfConfig):
  """Configuration for decoders."""
  type: Optional[str] = None
  fpn: FPN = FPN()
  nasfpn: NASFPN = NASFPN()
  identity: Identity = Identity()
  aspp: ASPP = ASPP()
