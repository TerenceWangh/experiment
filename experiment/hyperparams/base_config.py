"""Base configurations to standardize experiments."""
import copy
import dataclasses
import functools
import inspect
from typing import Any, List, Mapping, Optional, Type

from absl import logging
import tensorflow as tf
import yaml

from experiment.hyperparams import params_dict

_BOUND = set()


def bind(config_cls):
  """Bind a class to config cls."""
  if not inspect.isclass(config_cls):
    raise ValueError('The bind decorator is supposed to apply on the class '
                     'attribute. Received {}, not a class.'.format(config_cls))

  def decorator(builder):
    if config_cls in _BOUND:
      raise ValueError('Inside a program, we should not bind the config with a '
                       'class twice.')
    if inspect.isclass(builder):
      config_cls._BUILDER = builder # pylint: disable=protected-access
    elif inspect.isfunction(builder):

      def _wrapper(self, *arg, **kwargs): # pylintL disable=unused-argument
        return builder(*arg, **kwargs)

      config_cls._BUILDER = _wrapper
    else:
      raise ValueError('The `BUILDER` type is not supported: {}'.format(
          builder))
    _BOUND.add(config_cls)
    return builder

  return decorator


@dataclasses.dataclass
class Config(params_dict.ParamsDict):
  """The base configuration class that supports YAML/JSON based overrides.

  Because of YAML/JSON serialization limitations, some semantics of dataclass
  are not supported:
  * It recursively enforces a allow list of basic types and container types, so
      it avoids surprises with copy and reuse caused by unanticipated types.
  * Warning: it converts Dict to `Config` even within sequences,
    e.g. for config = Config({'key': [([{'a': 42}],)]),
         type(config.key[0][0][0]) is Config rather than dict.
    If you define/annotate some field as Dict, the field will convert to a
    `Config` instance and lose the dictionary type.
  """
  # The class of method to bind with the params class.
  _BUILDER = None
  # It's safe to add bytes and other immutable types here.
  IMMUTABLE_TYPES = (str, int, float, bool, type(None))
  # It's safe to add set, frozenset and other collections here.
  SEQUENCE_TYPES = (list, tuple)

  default_params: dataclasses.InitVar[Optional[Mapping[str, Any]]] = None
  restrictions: dataclasses.InitVar[Optional[List[str]]] = None

  def __post_init__(self,
                    default_params: Optional[Mapping[str, Any]],
                    restrictions: Optional[List[str]]):
    super().__init__(
        default_params=default_params,
        restrictions=restrictions)

  @property
  def BUILDER(self):
    return self._BUILDER

  @classmethod
  def _is_valid_sequence(cls, v):
    """Check if the input values are valid sequence.

    :param v: Input sequence.
    :return: True if the sequence is valid. Valid sequence includes the sequence
        type in cls.SEQUENCE_TYPES and element type is in cls.IMMUTABLE_TYPES or
        is dict or ParamsDict.
    """
    if not isinstance(v, cls.SEQUENCE_TYPES):
      return False
    return (all(isinstance(e, cls.IMMUTABLE_TYPES) for e in v) or
            all(isinstance(e, dict) for e in v) or
            all(isinstance(e, params_dict.ParamsDict) for e in v))

  @classmethod
  def _import_config(cls, v, subconfig_type):
    """Returns v with dicts converted to Configs, recursively."""
    if not issubclass(subconfig_type, params_dict.ParamsDict):
      raise TypeError(
          'subconfig_type should be subclass of ParamsDict, found {!r}'.format(
              subconfig_type))
    if isinstance(v, cls.IMMUTABLE_TYPES):
      return v
    elif isinstance(v, cls.SEQUENCE_TYPES):
      # Only support one layer of sequence.
      if not cls._is_valid_sequence(v):
        raise TypeError(
            'Invalid sequence: only supports single level {!r} of {!r} or dict '
            'or ParamDict found: {!r}'.format(
                cls.SEQUENCE_TYPES, cls.IMMUTABLE_TYPES, v))
      import_fn = functools.partial(cls._import_config,
                                    subconfig_type=subconfig_type)
      return type(v)(map(import_fn, v))
    elif isinstance(v, params_dict.ParamsDict):
      return copy.deepcopy(v)
    elif isinstance(v, dict):
      return subconfig_type(v)
    else:
      raise TypeError('Unknown type: {!r}'.format(type(v)))

  @classmethod
  def _export_config(cls, v):
    """Returns v with Configs converted to dicts, recursively"""
    if isinstance(v, cls.IMMUTABLE_TYPES):
      return v
    elif isinstance(v, cls.SEQUENCE_TYPES):
      return type(v)(map(cls._export_config, v))
    elif isinstance(v, params_dict.ParamsDict):
      return v.as_dict()
    elif isinstance(v, dict):
      raise TypeError('dict value not supported in converting.')
    else:
      raise TypeError('Unknown type: {!r}'.format(type(v)))

  @classmethod
  def _get_subconfig_type(cls, k) -> Type[params_dict.ParamsDict]:
    """Get element type by the field name.

    :param k: the key/name of the field.
    :return: Config as default. If a type annotation is found for `k`,
      1) returns the type of the annotation if it is subtype of ParamsDict;
      2) returns the element type if the annotation of `k` is List[SubType]
         or Tuple[SubType].
    """
    subconfig_type = Config
    if k in cls.__annotations__:
      # Directly Config subtype.
      type_annotation = cls.__annotations__[k]  # pytype: disable=invalid-annotation
      if (isinstance(type_annotation, type) and
          issubclass(type_annotation, Config)):
        subconfig_type = cls.__annotations__[k]  # pytype: disable=invalid-annotation
      else:
        # Check if the field is a sequence of subtypes.
        field_type = getattr(type_annotation, '__origin__', type(None))
        if (isinstance(field_type, type) and
            issubclass(field_type, cls.SEQUENCE_TYPES)):
          element_type = getattr(type_annotation, '__args__', [type(None)])[0]
          subconfig_type = (
              element_type if issubclass(element_type, params_dict.ParamsDict)
              else subconfig_type)
    return subconfig_type

  def _set(self, k, v):
    """Overrides same method in ParamsDict.

    Also called by ParamsDict methods.

    :param k: key string to set.
    :param v: value.
    :raise RuntimeError
    """
    subconfig_type = self._get_subconfig_type(k)

    def is_null(k):
      if k not in self.__dict__ or not self.__dict__[k]:
        return True
      return False

    if isinstance(v, dict):
      if is_null(k):
        # If the key not exist or the value is None, a new Config-family object
        # should be created for the key.
        self.__dict__[k] = subconfig_type(v)
      else:
        self.__dict__[k].override(v)
    elif not is_null(k) and isinstance(v, self.SEQUENCE_TYPES) and all(
        [not isinstance(e, self.IMMUTABLE_TYPES) for e in v]):
      if len(self.__dict__[k]) == len(v):
        for i in range(len(v)):
          self.__dict__[k][i].override(v[i])
      elif not all([isinstance(e, self.IMMUTABLE_TYPES) for e in v]):
        logging.warning(
            'The list/tuple don\'t match the value dictionaries provided, '
            'Thus, the list/tuple is determined by the type annotation and '
            'values provided. This is error-prone.')
        self.__dict__[k] = self._import_config(v, subconfig_type)
      else:
        self.__dict__[k] = self._import_config(v, subconfig_type)
    else:
      self.__dict__[k] = self._import_config(v, subconfig_type)

  def __setattr__(self, k, v):
    if k == 'BUILDER' or k == '_BUILDER':
      raise AttributeError('`BUILDER` is a property and `_BUILDER` is the '
                           'reserved class attribute. We should only assign '
                           '`_BUILDER` at the class level.')

    if k not in self.RESERVED_ATTR:
      if getattr(self, '_locked', False):
        raise ValueError('The config has been locked. No change is allowed.')
    self._set(k, v)

  def _override(self, override_dict, is_strict=True):
    """Overrides same method is ParamsDict.

    Also called by ParamsDict methods.

    :param override_dict: dictionary to write to.
    :param is_strict: If True, not allows to add new keys.
    :raise KeyError: overriding reserved keys or keys not exists if `is_strict`
      is True.
    """
    for k, v in sorted(override_dict.items()):
      if k in self.RESERVED_ATTR:
        raise KeyError('The key {!r} is internally reserved. Can not be '
                       'overridden.'.format(k))
      if k not in self.__dict__:
        if is_strict:
          raise KeyError('The key {!r} does not exists in {!r}. To extend the '
                         'existing keys, use `override` with '
                         '`is_strict` = False'.format(k, type(self)))
        else:
          self._set(k, v)
      else:
        if isinstance(v, dict) and self.__dict__[k]:
          self.__dict__[k]._override(v, is_strict)
        elif isinstance(v, params_dict.ParamsDict) and self.__dict__[k]:
          self.__dict__[k]._override(v.as_dict(), is_strict)
        else:
          self._set(k, v)

  def as_dict(self):
    """Returns a dict representation of params_dict.ParamsDict.

    :return: a dict representation of params_dict.ParamsDict.
    """
    return {
        k: self._export_config(v)
        for k, v in self.__dict__.items()
        if k not in self.RESERVED_ATTR
    }

  def replace(self, **kwargs):
    """Overrides/returns a unlocked copy with the current config unchanged."""
    # pylint: disable=protected-access
    params = copy.deepcopy(self)
    params._locked = False
    params._override(kwargs, is_strict=True)
    # pylint: enable=protected-access
    return params

  @classmethod
  def from_yaml(cls, file_path: str):
    # Note: This only works if the Config has all default values.
    with tf.io.gfile.GFile(file_path, 'r') as f:
      loaded = yaml.load(f, Loader=yaml.FullLoader)
      config = cls()
      config.override(loaded)
      return config

  @classmethod
  def from_json(cls, file_path: str):
    return cls.from_yaml(file_path)

  @classmethod
  def from_args(cls, *args, **kwargs):
    """Builds a config from the given list of arguments."""
    attributes = list(cls.__annotations__.keys())
    default_params = {a: p for a, p in zip(attributes, args)}
    default_params.update(kwargs)
    return cls(default_params=default_params)
