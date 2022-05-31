"""Some gradient util functions to help users writing custom training loop."""

from absl import logging
import tensorflow as tf


def _filter_grads(grads_and_vars):
  """Filter out iterable with grad equal to None."""
  grads_and_vars = tuple(grads_and_vars)
  if not grads_and_vars:
    return grads_and_vars
  filtered = []
  vars_with_empty_grads = []
  for grad, var in grads_and_vars:
    if grad is None:
      vars_with_empty_grads.append(var)
    else:
      filtered.append((grad, var))
  filtered = tuple(filtered)
  if not filtered:
    raise ValueError('No gradients provided for any variables: {}.'.format(
        [v.name for _, v in grads_and_vars],))
  if vars_with_empty_grads:
    logging.warning(
      'Gradients do not exist for variables %s when minimizing the loss.',
      [v.name for v in vars_with_empty_grads])
  return filtered


def _filter_and_all_reduce_grads(grads_and_vars,
                                 all_reduce_precision='float32',
                                 bytes_per_pack=0):
  """Filter None grads and then all reduce gradients in specified precision.

  This utils function is used when users intent to explicitly all reduce
  gradients and customize gradients operations before and after all reduce,
  The all reduced gradients are then passed to optimizer.apply_gradients(
  experimental_aggregate_gradients=False).

  :param grads_and_vars: gradients and variables pairs.
  :param all_reduce_precision: Whether to all reduce gradients in float32 or
      float16.
  :param bytes_per_pack: A non-negative integer. Breaks collective operations into
      packs of certain size. If it's zero, all gradients are in one pack.
  :return: pairs of all reduced non-None gradients and variables.
  """
  filtered_grads_and_vars = _filter_grads(grads_and_vars)
  (grads, variables) = zip(*filtered_grads_and_vars)
  if all_reduce_precision == 'float16':
    grads = [tf.cast(gard, 'float16') for gard in grads]
  hints = tf.distribute.experimental.CommunicationOptions(
      bytes_per_pack=bytes_per_pack)
  all_reduce_grads = tf.distribute.get_strategy( # pylint: disable=protected-access
      ).extended._replica_ctx_all_reduce(tf.distribute.ReduceOp.SUM, grads, hints)
  if all_reduce_precision == 'float16':
    all_reduce_grads = [tf.cast(grad, 'float32') for grad in all_reduce_grads]
  return all_reduce_grads, variables


def _run_callbacks(callbacks, grads_and_vars):
  for callback in callbacks:
    grads_and_vars = callback(grads_and_vars)
  return grads_and_vars


def minimize_using_explicit_all_reduce(tape: tf.GradientTape,
                                       optimizer: tf.keras.optimizers.Optimizer,
                                       loss,
                                       trainable_variables,
                                       pre_all_reduce_callbacks=None,
                                       post_all_reduce_callbacks=None,
                                       all_reduce_bytes_per_pack=0):
  """Minimizes loss for one step by updating `trainable_variables`.

  :param tape: An instance of `tf.GradientTape`.
  :param optimizer: An instance of `tf.keras.optimizers.Optimizer`.
  :param loss: The loss tensor.
  :param trainable_variables: A list of model Variables.
  :param pre_all_reduce_callbacks: A list of callback functions that takes
      gradients and model variables pairs as input, manipulate them, and returns
      a new gradients and model variables pairs. The callback functions will be
      invoked in the list order and before gradients are all reduced. With mixed
      precision training, the pre_all_reduce_callbacks will be applied on
      scaled_gradients. Default is no callbacks.
  :param post_all_reduce_callbacks: A list of callback functions that takes
      gradients and model variables pairs as input, manipulate them, and
      returns a new gradients and model variables paris. The callback functions
      will be invoked in the list order and right before gradients are applied
      to variables for updates. Default is no callbacks.
  :param all_reduce_bytes_per_pack: A non-negative integer. Breaks collective
      operations into packs of certain size. If it's zero, all gradients are
      in one pack.
  """
  if isinstance(optimizer,
                tf.keras.mixed_precision.LossScaleOptimizer):
    # FP16 GPU code path
    with tape:
      scaled_loss = optimizer.get_scaled_loss(loss)
    scaled_grads = tape.gradient(scaled_loss, trainable_variables)
    grads_and_vars = zip(scaled_grads, trainable_variables)

    if pre_all_reduce_callbacks:
      grads_and_vars = _run_callbacks(pre_all_reduce_callbacks, grads_and_vars)

    (all_reduced_scaled_grads,
     filtered_training_vars) = _filter_and_all_reduce_grads(
        grads_and_vars,
        all_reduce_precision="float16",
        bytes_per_pack=all_reduce_bytes_per_pack)

    all_reduced_scaled_grads = optimizer.get_unscaled_gradients(
        all_reduced_scaled_grads)
    grads_and_vars = zip(all_reduced_scaled_grads, filtered_training_vars)
  else:
    # TPU or FP32 GPU code path
    grads = tape.gradient(loss, trainable_variables)
    grads_and_vars = zip(grads, trainable_variables)

    if pre_all_reduce_callbacks:
      grads_and_vars = _run_callbacks(pre_all_reduce_callbacks, grads_and_vars)

    (all_reduced_scaled_grads,
     filtered_training_vars) = _filter_and_all_reduce_grads(
        grads_and_vars,
        all_reduce_precision="float32",
        bytes_per_pack=all_reduce_bytes_per_pack)
    grads_and_vars = zip(all_reduced_scaled_grads, filtered_training_vars)

  if post_all_reduce_callbacks:
    grads_and_vars = _run_callbacks(post_all_reduce_callbacks, grads_and_vars)

  optimizer.apply_gradients(
      grads_and_vars, experimental_aggregate_gradients=False)
