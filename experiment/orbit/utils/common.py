"""Some layered modules/functions to help users writing custom training loop."""

import inspect
import tensorflow as tf


def create_global_step() -> tf.Variable:
  """Creates a `tf.Variable` suitable for use as a global step counter.

  Creating and managing a global step variable may be necessary for
  `AbstractTrainer` subclass that perform multiple parameter updates per
  `Controller` "step", or use different optimizers on different steps.

  In these cases, an `optimizer.iterations` property generally can't be used
  directly, since it would correspond to parameter updates instead of iterations
  in the `Controller`'s training loop. Such use cases should simply call
  `step.assign_add(1)` at the end of each step.

  :return: A non-trainable scalar `tf.Variable` of dtype `tf.int64`, with only
      the first replica's value retained when synchronizing across replicas in
      a distributed setting.
  """
  return tf.Variable(0, dtype=tf.int64, name='global_step', trainable=False,
                     aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)


def make_distributed_dataset(strategy, dataset_or_fn, *args, **kwargs):
  """A utility function to help create a `tf.distribute.DistributedDataset`.

  :param strategy: An instance of `tf.distribute.Strategy`.
  :param dataset_or_fn: A instance of `tf.data.Dataset`, or a 'dataset function'
      returning a `tf.data.Dataset`. If it is a function, it may optionally have
      an argument named `input_context` which will be passed a
      `tf.distribute.InputContext` instance.
  :param args: Any positional arguments to pass through to `dataset_or_fn`.
  :param kwargs: Any keyword arguments to pass through to `dataset_or_fn`,
      except that the `input_options` keyword is used to specify a
      `tf.distribute.InputOptions` for making the distributed dataset.
  :return: A distributed Dataset.
  """
  if strategy is None:
    strategy = tf.distribute.get_strategy()

  input_options = kwargs.pop('input_options', None)
  if isinstance(dataset_or_fn, tf.data.Dataset):
    return strategy.experimental_distribute_dataset(dataset_or_fn,
                                                    input_options)

  if not callable(dataset_or_fn):
    raise ValueError('`dataset_fn` should be either callable or an instance '
                     'of `tf.data.Dataset`.')

  def dataset_fn(input_context):
    """Wrap `dataset_or_fn` for strategy.distribute_datasets_from_function."""
    # If `dataset_or_fn` is a function and has an argument named
    # `input_context`, pass through the given `input_context`. Otherwise
    # `input_context` will be ignored.
    argspec = inspect.getfullargspec(dataset_or_fn)
    arg_names = argspec.args

    if 'input_context' in arg_names:
      kwargs['input_context'] = input_context
    return dataset_or_fn(*args, **kwargs)

  return strategy.distribute_datasets_from_function(dataset_fn, input_options)


def get_value(x):
  """Returns input value, converting any TensorFlow values to NumPy values.

  :param x: The input. May be a `tf.Tensor` or `tf.Variable`.
  :return: If the input is a TensorFlow `Tensor`, returns the `Tensor`'s
      equivalent NumPy value. Otherwise, just returns the input.
  """
  if not tf.is_tensor(x):
    return x
  return x.numpy()
