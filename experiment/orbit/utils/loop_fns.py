"""Utilities for creating loop functions."""

from absl import logging
from experiment.orbit.utils import tpu_summaries
import tensorflow as tf


def create_loop_fn(step_fn):
  """Creates a loop function driven by a Python `while` loop.

  :param step_fn: A function taking a nested structure of `tf.data.Iterator` or
      `DistributedIterator`. There are no constraints on the return value of the
      function (except that it must be compatible with any `reduce_fn` provided
      to the returned `loop_fn`).
  :return: A loop function taking required `iterator` and `num_steps`
      parameters, as well as optional `state` and `reduce_fn` parameters for
      accumulating state over multiple iterations of the loop. See the `loop_fn`
      definition below for additional details.
  """

  def loop_fn(iterator, num_steps, state=None, reduce_fn=None):
    """Makes `num_steps` calls to `step_fn(iterator)`.

    Additionally, state may be accumulated across iterations of the loop.
    Conceptually, state accumulation is handled roughly as follows:

        for _ in range(num_steps):
          step_outputs  = step_fn(iterator)
          state = reduce_fn(state, step_outputs)
        return state

    However, the implementation is slightly more complicated in order to support
    looping until the iterator is exhausted (when `num_steps == -1`) and to
    properly catch exceptions when running under async remote eager (as is the
    case in TPU training setups involving separate coordinator/worker machines).

    :param iterator: A nested structure of `tf.data.Iterator` or
        `DistributedIterator`.
    :param num_steps: The number of steps in the loop. If `num_steps == -1`,
        will iterate until exhausting the iterator.
    :param state: An optional initial state before running the loop.
    :param reduce_fn: A callable taking two inputs, `state` and `value`, where
        `state` is the previous output from `reduce_fn`, and `value` is the
        output from `step_fn`.
    :return: The final state returned by `reduce_fn`, or `None` if `state` and
        `reduce_fn` are not provided.
    """
    step = 0
    try:
      # To make sure the OutOfRangeError exception can be handled well under
      # async remote eager, we need to wrap the loop body in `async_scope`.
      with tf.experimental.async_scope():
        while num_steps == -1 or step < num_steps:
          outputs = step_fn(iterator)
          if reduce_fn is not None:
            state = reduce_fn(state, outputs)
          step += 1
        return state
    except (StopIteration, tf.errors.OutOfRangeError):
      logging.info('The dataset iterator is exhausted after %d steps.', step)
      tf.experimental.async_clear_error()
      return  state

  return loop_fn


def create_tf_while_loop_fn(step_fn):
  """Creates a loop function compatible with TF's AutoGraph loop conversion.

  :param step_fn: A function taking a nested structure of `tf.data.Iterator` or
      `DistributedIterator`. Currently, any return values are ignored.
  :return: A loop function taking required `iterator` and `num_steps`
      parameters. If called inside a `tf.function`, the loop will be converted
      by AutoGraph into a `tf.while_loop` construct. See the `loop_fn`
      definition below for additional details.
  """

  def loop_fn(iterator, num_steps):
    """Makes `num_steps` calls to `step_fn(iterator)`.

    :param iterator: A nested structure of `tf.data.Iterator` or
        `DistributedIterator`.
    :param num_steps: The number of steps in the loop. Should be passed as a
        `tf.Tensor`. Iterating until iterator exhaustion is not supported.
    """
    if not isinstance(num_steps, tf.Tensor):
      raise ValueError('`num_steps` should be a `tf.Tensor`. Passing a Python '
                       'value can cause unnecessary retracing when wrapped by '
                       '`tf.function`.')

    for _ in tf.range(num_steps):
      # Clear out the outer name scope so the ops created inside `tf.while_loop`
      # don't get 'while/' as name prefix.
      with tf.name_scope(''):
        step_fn(iterator)

  return loop_fn


def create_tf_while_loop_fn_with_state(step_fn):
  """Creates a TF while loop function with state.

  This function is similar to `create_tf_while_loop_fn`, but allowing a `state`
  to be accumulated over multiple iterations of the loop. Note that the
  structure of the `state` cannot be changed across iterations.

  :param step_fn: A function taking a nested structure of `tf.data.Iterator` or
      `DistributedIterator`. Currently, any return values are ignored.
  :return: A loop function taking required `iterator`, `num_steps`, `state` and
      `reduce_fn` parameters. If called inside a `tf.function`, the loop will be
      converted by AutoGraph into a `tf.while_loop` construct. See the `loop_fn`
      definition below for additional details.
  """

  def loop_fn_with_state(iterator, num_steps, state, reduce_fn):
    """Makes `num_steps` calls to `step_fn(iterator)`.

    :param iterator: A nested structure of `tf.data.Iterator` or
        `DistributedIterator`.
    :param num_steps: The number of steps in the loop. Should be passed as a
        `tf.Tensor`. Iterating until iterator exhaustion is not supported.
    :param state: An initial state before running the loop.
    :param reduce_fn: A callable taking two inputs, `state` and `value`, where
        `state` is the previous output from `reduce_fn`, and `value` is the
        output from `step_fn`.
    :return: The final state returned by `reduce_fn`.
    """
    if not isinstance(num_steps, tf.Tensor):
      raise ValueError('`num_steps` should be a `tf.Tensor`. Passing a Python '
                       'value can cause unnecessary retracing when wrapped by '
                       '`tf.function`.')

    def _get_relaxed_tensor_shape(t):
      if not tf.is_tensor(t):
        return None
      shape = t.shape
      if shape.rank is not None and shape.rank > 0:
        return tf.TensorShape([None] * shape.rank)
      return shape

    def _get_relaxed_shape_structure(s):
      return tf.nest.pack_sequence_as(
          state, [_get_relaxed_tensor_shape(t) for t in tf.nest.flatten(s)])

    for _ in tf.range(num_steps):
      # Clear out the outer name scope so the ops created inside `tf.while_loop`
      # don't get "while/" as name prefix.
      with tf.name_scope(""):
        # Relax the shapes within the loop, so the shape of `state` can change
        # across iterations. This is useful to aggregate outputs from each step
        # and concat to `state`.
        tf.autograph.experimental.set_loop_options(
            shape_invariants=[(state, _get_relaxed_shape_structure(state))])
        outputs = step_fn(iterator)
        state = reduce_fn(state, outputs)
    return state

  return loop_fn_with_state


class LoopFnWithSummaries(tpu_summaries.OptionalSummariesFunction):
  """Implements a two-program approach for optimizing summaries on TPU.

  This version works with the result of `create_tf_while_loop_fn`.
  """

  def __call__(self, iterator, num_steps):
    if tf.summary.should_record_summaries():
      output = self.with_summaries(iterator, tf.constant(1))
      num_steps -= 1
    if num_steps >= 1:
      output = self.without_summaries(iterator, num_steps)
    return output
