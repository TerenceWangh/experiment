"""Backbone registers and factory method.

One can regitered a new backbone model by the following two steps:

1 Import the factory and register the build in the backbone file.
2 Import the backbone class and add a build in __init__.py.

```
# my_backbone.py

from model.backbones import factory

class MyBackbone():
  ...

@factory.register_backbone_builder('my_backbone')
def build_my_backbone():
  return MyBackbone()

# backbones/__init__.py adds import
from model.backbones.my_backbone import MyBackbone

```

If one wants the MyBackbone class to be used only by those binary
then don't imported the backbone module in backbones/__init__.py, but import it
in place that uses it.
"""

from typing import Sequence, Union

import tensorflow as tf

from experiment.core import registry
from experiment.hyperparams import Config

_REGISTERED_BACKBONE_CLS = {}


def register_backbone_builder(key: str):
  """Decorates a builder of backbone class.

  The builder should be a Callable (a class or a function).
  This decorator supports registration of backbone builder as follows:

  ```
  class MyBackbone(tf.keras.Model):
    pass

  @register_backbone_builder('mybackbone')
  def builder(input_specs, config, l2_reg):
    return MyBackbone(...)

  # Builds a MyBackbone object.
  my_backbone = build_backbone_3d(input_specs, config, l2_reg)
  ```

  Arguments
  =========
  key : str
      The key to look up the builder.

  Returns
  =======
  callable
      A callable for using as class decorator that registers the decorated class
      for creation from an instance of task_config_cls.
  """
  return registry.register(_REGISTERED_BACKBONE_CLS, key)


def build_backbone(input_specs: Union[tf.keras.layers.InputSpec,
                                      Sequence[tf.keras.layers.InputSpec]],
                   backbone_config: Config,
                   norm_activation_config: Config,
                   l2_regularizer: tf.keras.regularizers.Regularizer = None,
                   **kwargs) -> tf.keras.Model: # pytype: disable=annotation-type-mismatch  # typed-keras
  """Builds backbone from a config.

  Arguments
  =========
  input_specs : tf.keras.layers.InputSpec
      A (sequence of) of input.
  backbone_config : Config
      A `OneOfConfig` of backbone config.
  norm_activation_config : Config
      A config for normalization/activation layer.
  l2_regularizer : tf.keras.regularizers.Regularizer, optional
      The regularizer of kernel.
  **kwargs : dict
      Additional keyword args to be passed to backbone builder.
  """
  backbone_builder = registry.lookup(_REGISTERED_BACKBONE_CLS,
                                     backbone_config.type)

  return backbone_builder(
      input_specs=input_specs,
      backbone_config=backbone_config,
      norm_activation_config=norm_activation_config,
      l2_regularizer=l2_regularizer,
      **kwargs)
