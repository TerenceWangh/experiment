"""Decoder registers and factory method.

One can register a new decoder model by the following two steps:

1. Import the factory and register the build in the decoder file.
2. Import the decoder class and add a build in __init__.py.

```
# my_decoder.py

from modeling.decoders import factory

class MyDecoder():
  ...

@factory.register_decoder_builder('my_decoder')
def build_my_decoder():
  return MyDecoder()

# decoders/__init__.py adds import
from modeling.decoders.my_decoder import MyDecoder
```

If one wants the MyDecoder class to be used only by those binary then don't
imported the decoder module in decoders/__init__.py, but import it in place
that uses it.
"""
from typing import Any, Callable, Mapping, Optional, Union
import tensorflow as tf

from experiment.core import registry
from experiment.hyperparams import Config

Regularizer = tf.keras.regularizers.Regularizer
Decoder = Union[None, tf.keras.Model, tf.keras.layers.Layer]

_REGISTERED_DECODER_CLS = {}


def register_decoder_builder(key: str) -> Callable[..., Any]:
  """Decorates a builder of decoder class.

  The builder should be a Callable (a class or a function).
  This decorator supports registration of decoder builder as follows:

  ```
  class MyDecoder(tf.keras.Model):
    pass

  @register_decoder_builder('mydecoder')
  def builder(input_specs, config, l2_reg):
    return MyDecoder(...)

  # Builds a MyDecoder object.
  my_decoder = build_decoder_3d(input_specs, config, l2_reg)
  ```

  Parameters
  ==========
  key : str
      A `str` of key to look up the builder.

  Returns
  =======
  callable
      A callable for using as class decorator that registers the decorated class
      for creation from an instance of task_config_cls.
  """
  return registry.register(_REGISTERED_DECODER_CLS, key)


@register_decoder_builder('identity')
def build_identity(input_specs: Optional[Mapping[str, tf.TensorShape]] = None,
                   model_config: Optional[Config] = None,
                   l2_regularizer: Optional[Regularizer] = None):
  """Builds identity decoder from a config.

  All the input arguments are not used by identity decoder but kept here to
  ensure the interface is consistent.

  Parameters
  ----------
  input_specs : dict, optional
      A dictionary consists of {level: TensorShape} from a backbone.
  model_config : Config, optional
      The configuration of model.
  l2_regularizer : Regularizer, optional
      The lr regularizer.
  """
  del input_specs, model_config, l2_regularizer


def build_decoder(input_specs: Mapping[str, tf.TensorShape],
                  model_config: Config,
                  l2_regularizer: Regularizer = None,
                  **kwargs) -> Decoder:
  """Builds decoder from a config.

  A decoder can be a keras.Model, a keras.layers.Layer, or None. If it is not
  None, the decoder will take features from the backbone as input and generate
  decoded feature maps. If it is None, such as an identity decoder, the decoder
  is skipped and features from the backbone are regarded as model output.

  Parameters
  ----------
  input_specs : dict
      A dictionary consists of {level: TensorShape} from a backbone.
  model_config : Config
      The model config.
  l2_regularizer : Regularizer, optional
      The lr regularizer.
  kwargs : dict
      Additional keyword args to be passed to decoder builder.
  """
  decoder_builder = registry.lookup(_REGISTERED_DECODER_CLS,
                                    model_config.decoder.type)

  return decoder_builder(input_specs=input_specs,
                         model_config=model_config,
                         l2_regularizer=l2_regularizer,
                         **kwargs)
