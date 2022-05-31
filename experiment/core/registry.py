"""Registry utility."""
from absl import logging


def register(registered_collection, reg_key):
  """Register decorated function or class to collection.

  Register decorated function or class into registered_collection, in a
  hierarchical order. For example, when reg_key="my_model/my_exp/my_config_0"
  the decorated function or class is stored under
  registered_collection["my_model"]["my_exp"]["my_config_0"].
  This decorator is supposed to be used together with the lookup() function in
  this file.

  :param registered_collection: a dictionary. The decorated function or class
      will be put into this collection.
  :param reg_key: The key for retrieving the registered function or class. If
      reg_key is a string, it can be hierarchical like
      `my_model/my_exp/my_config_0`.
  :return: A decorator function.
  :raise KeyError: when function or class register already exists.
  """
  def decorator(fn_or_cls):
    """Put fn_or_cls in the dictionary."""
    if isinstance(reg_key, str):
      hierarchy = reg_key.split('/')
      collection = registered_collection
      for h_idx, entry_name in enumerate(hierarchy[:-1]):
        if entry_name not in collection:
          collection[entry_name] = {}
        collection = collection[entry_name]
        if not isinstance(collection, dict):
          raise KeyError(
            'Collection path {} at position {} already registered as a '
            'function or class.'.format(entry_name, h_idx))
      leaf_reg_key = hierarchy[-1]
    else:
      collection = registered_collection
      leaf_reg_key = reg_key

    if leaf_reg_key in collection:
      if 'beta' in fn_or_cls.__module__:
        logging.warning(
          'Duplicate register of beta module name {} new {} old {}'.format(
            reg_key, collection[leaf_reg_key], fn_or_cls.__module__))
        return fn_or_cls
      else:
        raise KeyError('Function or class {} registered multiple times.'.format(
          leaf_reg_key))
    collection[leaf_reg_key] = fn_or_cls
    return fn_or_cls

  return decorator


def lookup(registered_collection, reg_key):
  """Lookup and return decorated function or class in the collection.

  Lookup decorated function or class in registered_collection, in a
  hierarchical order. For example, when
  reg_key="my_model/my_exp/my_config_0",
  this function will return
  registered_collection["my_model"]["my_exp"]["my_config_0"].

  :param registered_collection: a dictionary. The decorated function or class
      will be retrieved from this collection.
  :param reg_key: The key for retrieving the registered function or class. If
      reg_key is a string, it can be hierarchical like
      `my_model/my_exp/my_config_0`.
  :return: The registered function or class.
  :raise LookupError: when reg_key cannot be found.
  """
  if isinstance(reg_key, str):
    hierarchy = reg_key.split('/')
    collection = registered_collection
    for h_idx, entry_name in enumerate(hierarchy):
      if entry_name not in collection:
        raise LookupError(
          'collection path {} at position {} is never registered. Please make '
          'sure the {} and its library is imported and linked to the trainer '
          'binary.'.format(entry_name, h_idx, entry_name))
      collection = collection[entry_name]
    return collection
  else:
    if reg_key not in registered_collection:
      raise LookupError(
        'registration key {} is never registered. Please ,ake sure the {} and '
        'its library is imported and linked to the trainer binary.'.format(
          reg_key, reg_key))
    return registered_collection[reg_key]
