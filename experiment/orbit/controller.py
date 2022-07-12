"""Provides a `Controller` class for managing the outer training loop."""

import pprint
import time

from typing import Callable, List, Optional, Union
from absl import logging

from experiment.orbit import runner
from experiment.orbit import utils

import tensorflow as tf


def _log(message: str):
  """Log `message` to the `info` log."""
  logging.info(message)


logging.ABSLLogger.register_frame_to_skip(__file__, _log.__name__)


def get_kv_from_lines(lines: List[str]):
  kvs = []
  max_length = 0
  for line in lines:
    kv = line.split(':', maxsplit=2)
    if len(kv) == 1:
      max_length = max(max_length, len(kv))
      kvs.append([kv, ''])
    else:
      max_length = max(max_length, len(kv[0]))
      kvs.append(kv)

  out_lines = []
  max_length += 2
  for k, v in kvs:
    out_lines.append('{}{}: {}'.format(
        k, ' ' * (max_length - len(k)), v))
  return out_lines


def _format_output(output, indent=2):
  """Format `output`, either on one line, or indented across multiple line."""
  formatted = pprint.pformat(output)

  # Remove the '{' and '}'
  if formatted.startswith('{'):
    formatted = formatted[1:]
  if formatted.endswith('}'):
    formatted = formatted[:-1]
  lines = get_kv_from_lines(formatted.splitlines())
  if len(lines) == 1:
    return formatted
  lines = [' ' * indent + line for line in lines]
  return '\n' + '\n'.join(lines) + '\n'


Action = Callable[[runner.Output], None]


class Controller:
  """Class that controls the outer loop of model training and evaluation.

  Orbit divides training and evaluation into "inner" and "outer" loops. Inner
  loops are implemented by users in the form of `AbstractTrainer` and
  `AbstractEvaluator` subclasses, and define how to run a given number of
  training or evaluation steps. The outer loop is provided by this `Controller`,
  and interleaves calls to the user-provided inner loops with additional actions
  such as saving checkpoints, running evaluations, writing summaries, as well as
  (optionally) user provided `Action`s (see below).

  There are four top-level "outer loops" provided:
    - `train`, which trains until a specified number of global steps is reached;
    - `evaluate`, for one-off model evaluation;
    - `train_and_evaluate`, for interleaved training and evaluation;
    - `evaluate_continuously`, for monitoring a given directory and running
      evaluations on new model checkpoints.

  While this class attempts to provide out-of-the-box solutions for common
  training and evaluation use cases, the internal details and method
  implementations are also intended to be simple enough to make subclassing or
  other custom outer loop implementations easy to achieve.

  Some additional customization can be achieved by supplying `train_actions` or
  `eval_actions` when constructing the `Controller`. These are just lists of
  arbitrary callables that are applied by the `Controller` to the output of
  train steps (after each inner loop of `steps_per_loop` steps) or an
  evaluation. This provides a hook mechanism, enabling things like reporting
  metrics to Vizier, model exporting, additional logging, etc. See the
  `orbit.actions` package for a small handful of predefined actions and some
  utility classes that may be useful in defining your own.
  """

  def __init__(self,
               *, # Makes all args keyword only
               global_step: tf.Variable,
               trainer: Optional[runner.AbstractTrainer] = None,
               evaluator: Optional[runner.AbstractEvaluator] = None,
               strategy: Optional[tf.distribute.Strategy] = None,
               # Actions
               train_actions: Optional[List[Action]] = None,
               eval_actions: Optional[List[Action]] = None,
               # Train related
               steps_per_loop: Optional[int] = None,
               checkpoint_manager: Optional[tf.train.CheckpointManager] = None,
               # Summary related
               summary_interval: Optional[int] = None,
               summary_dir: Optional[str] = None,
               # Evaluation related
               eval_summary_dir: Optional[str] = None):
    """Initializes a `Controller` instance.

    Note that if `checkpoint_manager` is provided and there are checkpoints in
    the associated model directory, the model will be restored from the most
    recent checkpoint during this `__init__` method.

    :param global_step: An integer `tf.Variable` storing the global training
        step number. Usually this can be obtained from the `iterations` property
        of the model's optimizer (e.g. `trainer.optimizer.iterations`). In cases
        where multiple optimizers are used, or if one model "step" corresponds
        to more than one update to model parameters, users can create and
        increment their own global step variable as well. In this case it is
        recommended to create the `tf.Variable` inside the distribution strategy
        scope, with `aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA` (see
        also `orbit.utils.create_global_step()`).
    :param trainer: An instance of `orbit.AbstractTrainer`, which implements the
        inner training loop.
    :param evaluator: An instance of `orbit.AbstractEvaluator`, which implements
        evaluation.
    :param strategy: An instance of `tf.distribute.Strategy`. If not provided,
        the strategy will be initialized from the current in-scope strategy
        using `tf.distribute.get_strategy()`.
    :param train_actions: An optional list of `orbit.Action`s to call after each
        block of `steps_per_loop` training steps are run. These will be called
        with the output of `trainer.train`.
    :param eval_actions: An optional list of `orbit.Action`s to call after each
        evaluation. These will be called with the output of
        `evaluator.evaluate`.
    :param steps_per_loop: The number of steps to run in each inner loop of
        training (passed as the `num_steps` parameter of `trainer.train`).
    :param checkpoint_manager: An instance of `tf.train.CheckpointManager`. If
        provided and there are checkpoints in the associated model directory,
        the model will be restored from the most recent checkpoint inside this
        `__init__` method. If not provided, the `Controller` will not
        automatically save to or restore from checkpoints.
    :param summary_interval: Step interval for training summaries. Note that
        this argument only applies to `tf.summary` calls inside the
        `trainer.train` function. Summaries written by the `Controller`
        (specifically "steps_per_second" and output from the `trainer.train`
        method) will always be enabled unless the `summary_dir` parameter is
        `None`. If set, the value must be divisible by `steps_per_loop`.
    :param summary_dir: The directory to write summaries to. To use the same
        directory as for checkpointing, pass `checkpoint_manager.directory`. If
        `None`, no training summaries will be written.
    :param eval_summary_dir: The directory to write eval summaries to.
        If `None`, it will be set to `summary_dir`. If both `summary_dir` and
        `eval_summary_dir` are `None`, no eval summaries will be written.
    :raise ValueError: If both `trainer` and `evaluator` are `None`, or if
        `steps_per_loop` is not a positive integer, or if `summary_interval` is
        not a positive integer or is not divisible by `steps_per_loop`.
    """
    if trainer is None and evaluator is None:
      raise ValueError('`trainer` and `evaluator` should not both be `None`.')

    if trainer is not None:
      if steps_per_loop is None:
        raise ValueError(
            '`steps_per_loop` is required when `trainer` is provided.')
      elif not isinstance(steps_per_loop, int) or steps_per_loop < 1:
        raise ValueError(
            f'`steps_per_loop` ({steps_per_loop}) must be a positive integer.')

      if summary_interval is not None:
        if summary_interval <= 0:
          raise ValueError(
              f'`summary_interval` ({summary_interval}) must be larger than 0.')
        elif summary_interval % steps_per_loop != 0:
          raise ValueError(
              f'`summary interval` ({summary_interval}) must be a multiple '
              f'of `steps_per_loop` ({steps_per_loop}).')

    if not isinstance(global_step, tf.Variable):
      raise ValueError('`global_step` must be a `tf.Variable`.')

    self._trainer = trainer
    self._evaluator = evaluator

    self._strategy = strategy or tf.distribute.get_strategy()

    self._train_actions = train_actions or []
    self._eval_actions = eval_actions or []

    self._global_step = global_step
    self._checkpoint_manager = checkpoint_manager

    if self._trainer is not None:
      self._step_timer = None
      self._steps_per_loop = steps_per_loop
      self._summary_dir = summary_dir
      self._summary_interval = summary_interval
      self._summary_manager = utils.SummaryManager(
          summary_dir, tf.summary.scalar, global_step=self._global_step)

    if self._evaluator is not None:
      eval_summary_dir = eval_summary_dir or summary_dir
      if eval_summary_dir == summary_dir and self._trainer is not None:
        # Reuse the summary writer if train and evaluation summary directory
        # are the same.
        self._eval_summary_manager = self._summary_manager
      else:
        self._eval_summary_manager = utils.SummaryManager(
            eval_summary_dir, tf.summary.scalar, global_step=self._global_step)
    tf.summary.experimental.set_step(self._global_step)

    # Restores the model if needed.
    if self._checkpoint_manager is not None:
      restored_path = self.restore_checkpoint()
      if restored_path:
        _log(f'Restored from checkpoint: {restored_path}')

  def train(self, steps: int, checkpoint_at_completion: bool = True):
    """Runs training until the specified global step count has been reached.

    This method makes calls to `self.trainer.train()` until the global step
    count is equal to `steps`. It will additionally save checkpoints (if a
    `CheckpointManager` was passed to `Controller.__init__`) and summarize
    training output (if `summary_dir` is set).

    :param steps: The global step count to train up to.
    :param checkpoint_at_completion: Whether to save a checkpoint when this
        method returns (regardless of the checkpointing interval). Defaults to
        `True`.
    """
    self._require('_trainer', for_method='train')

    current_step = self._global_step.numpy() # Cache since this is expensive.
    _log('train | step: {:7d} | training until step {}...'.format(
        current_step, steps))

    if current_step == 0:
      tf.summary.trace_on(graph=True, profile=True)
      with self._summary_manager.summary_writer().as_default():
        tf.summary.trace_export(name='model_trace', step=0,
                                profiler_outdir=self._summary_dir)
      tf.summary.trace_off()

    while current_step < steps:
      # Calculates steps to run for the next train loop.
      num_steps = min(steps - current_step, self._steps_per_loop)
      self._train_n_steps(num_steps)
      self._maybe_save_checkpoint()
      current_step = self._global_step.numpy()

    if checkpoint_at_completion:
      self._maybe_save_checkpoint(check_interval=False)

  def evaluate(self, steps: int = -1) -> Optional[runner.Output]:
    """Runs evaluation for the given number of steps.

    This method calls `self.evaluator.evaluate(steps)`, then writes the returned
    summaries (if any).

    :param steps: The number of evaluation steps to run. The value `-1` is
        reserved as a special sentinel to indicate a "complete" evaluation that
        runs until the underlying dataset is exhausted. Support for this is
        dependent on the specific `evaluator` being used.
    :return: The evaluation results as a dictionary mapping names to NumPy
        values.
    :raise ValueError: If `evaluator` was not provided to `Controller.__init__`,
        or if no checkpoint is present in `checkpoint_manager.directory`, or if
        `steps` is not a positive value or -1.
    """
    self._require('_evaluator', for_method='evaluate')

    if steps > 0:
      steps_msg = 'running {} steps of evaluation...'.format(steps)
    elif steps == -1:
      steps_msg = 'running complete evaluation...'
    else:
      raise ValueError('`steps` ({}) should be > 0 or == -1'.format(steps))

    current_step = self._global_step.numpy()
    _log(' eval | step: {:7d} | {}'.format(current_step, steps_msg))

    start = time.time()
    with self._eval_summary_manager.summary_writer().as_default():
      steps_tensor = tf.convert_to_tensor(steps, dtype=tf.int32)
      eval_output = self._evaluator.evaluate(steps_tensor)
    elapsed = time.time() - start

    eval_output = eval_output or {}
    for action in self._eval_actions:
      action(eval_output)
    eval_output = tf.nest.map_structure(utils.get_value, eval_output)

    _log(' eval | step: {:7d} | eval time: {:6.1f} sec | output: {}'.format(
        current_step, elapsed, _format_output(eval_output)))

    self._eval_summary_manager.write_summaries(eval_output)
    self._eval_summary_manager.flush()

    return eval_output

  def train_and_evaluate(self,
                         train_steps: int,
                         eval_steps: int,
                         eval_interval: Optional[int] = None) -> None:
    """Runs interleaved training and evaluation.

    This method interleaves calls to `self.train()` and `self.evaluate()`,
    training the model until the global step count equals `train_steps`, and
    running an evaluation for `eval_steps` every `eval_interval` training steps.
    In addition, this method will run a final evaluation at the end of the
    training sequence.

    :param train_steps: The global step count to train up to.
    :param eval_steps: The number of steps to run during an evaluation. If -1,
        this method will evaluate over the entire evaluation dataset.
    :param eval_interval: The number of training steps to run between
        evaluations. If set, training will always stop every `eval_interval`
        steps, even if this results in a shorter inner loop than specified by
        `steps_per_loop` setting. If None, evaluation will only be performed
        after training is complete.
    :raise ValueError: If eval_interval is not a multiple of
        self.steps_per_loop.
    """
    self._require('_trainer', for_method='train_and_evaluate')
    self._require('_evaluator', for_method='train_and_evaluate')

    current_step = self._global_step.numpy()  # Cache, since this is expensive.
    eval_interval = eval_interval or (train_steps - current_step)
    while current_step < train_steps:
      interval = min(train_steps - current_step, eval_interval)
      num_steps = current_step + interval
      self.train(steps=num_steps, checkpoint_at_completion=False)
      self.evaluate(steps=eval_steps)
      current_step = self._global_step.numpy()
    self._maybe_save_checkpoint(check_interval=False)

  def evaluate_continuously(self,
                            steps: int = -1,
                            timeout: Optional[Union[int, float]] = None,
                            timeout_fn: Optional[Callable[[], bool]] = None):
    """Continuously monitors a directory and evaluates new checkpoints in it.

    This method continuously monitors a directory as specified by this
    Controller's CheckpointManager init arg and runs evaluation on the
    checkpoints found there.

    :param steps: The number of steps to run when evaluating. If -1, this method
        will evaluate over the entire evaluation dataset.
    :param timeout: The maximum number of seconds to wait between checkpoints.
        See tf.train.checkpoints_iterator documentation.
    :param timeout_fn: Optional callable to call after a timeout. If the
        function returns True, then it means that no new checkpoints will be
        generated and the iterator will exit.
    :raise ValueError: If no checkpoint found in
        `self.checkpoint_manager.directory`, or if `evaluator` was not provided
        as a controller init arg.
    """
    self._require('_evaluator', for_method='evaluate_continuously')
    self._require('_checkpoint_manager', for_method='evaluate_continuously')

    for checkpoint_path in tf.train.checkpoints_iterator(
        self._checkpoint_manager.directory,
        timeout=timeout, timeout_fn=timeout_fn):
      self.restore_checkpoint(checkpoint_path)
      self.evaluate(steps)

  def restore_checkpoint(self, checkpoint_path: Optional[str] = None):
    """Restores the model from a checkpoint.

    :param checkpoint_path: An optional string specifying the checkpoint path to
        restore from. If `None`, will restore from the most recent checkpoint
        (or initialize the model using a custom `init_fn` if no checkpoints can
        be found) using `self.checkpoint_manager.restore_or_initialize()`.
    :return: The path to the restored checkpoint if a restore happened, or
        `None` if no restore occurred.
    """
    self._require('_checkpoint_manager', for_method='restore_checkpoint')

    with self._strategy.scope():
      # Checkpoint restoring should be inside scope
      if checkpoint_path is not None:
        _log('Restoring model from {}...'.format(checkpoint_path))
        self._checkpoint_manager.checkpoint.restore(checkpoint_path)
      else:
        _log('Restoring or initializing model...')
        checkpoint_path = self._checkpoint_manager.restore_or_initialize()

    if checkpoint_path is not None:
      _log('Restored model from {}.'.format(checkpoint_path))
    else:
      _log('Initialized model.')

    return checkpoint_path

  def save_checkpoint(self):
    """Saves the model to a checkpoint.

    This method will save a checkpoint containing the current state of the
    model.

    :raise ValueError: If no `checkpoint_manager` was provided to
        `Controller.__init__`.
    """
    self._require('_checkpoint_manager', for_method='save_checkpoint')
    self._maybe_save_checkpoint(check_interval=False)

  def _train_n_steps(self, num_steps: int):
    """Runs training for `num_steps` steps.

    Also prints/logs updates about training progress, and summarizes training
    output (if output is returned from `self.trainer.train()`, and if
    `self.summary_dir` is set).

    :param num_steps: An integer specifying how many steps of training to run.
    :raise RuntimeError: If `global_step` is not properly incremented by
        `num_steps` after calling `self.trainer.train(num_steps)`.
    """
    if not self._step_timer:
      self._step_timer = StepTimer(self._global_step)
    current_step = self._global_step.numpy()

    with self._summary_manager.summary_writer().as_default():
      should_record = False # Allows static optimization in no-summary cases.
      if self._summary_interval:
        # Create a predicate to determine when summaries should be written.
        should_record = lambda: (
            self._global_step % self._summary_interval == 0)
      with tf.summary.record_if(should_record):
        num_steps_tensor = tf.convert_to_tensor(num_steps, dtype=tf.int32)
        train_output = self._trainer.train(num_steps_tensor)

    # Verify that global_step was updated properly, then update current_step.
    expected_step = current_step + num_steps
    if self._global_step.numpy() != expected_step:
      message = '`trainer.train({n})` did not update `global_step` by {n}. '   \
                'Old value was {c}, expected updated value to be {e}, but it ' \
                'was {g}.'.format(n=num_steps, c=current_step, e=expected_step,
                                  g=self._global_step.numpy())
      logging.warning(message)

    train_output = train_output or {}
    for action in self._train_actions:
      action(train_output)
    train_output = tf.nest.map_structure(utils.get_value, train_output)

    current_step = self._global_step.numpy()
    steps_per_second = self._step_timer.steps_per_second()
    _log('train | step: {:7d} | steps/sec: {:6.1f} | output: {}'.format(
        current_step, steps_per_second, _format_output(train_output)))

    train_output['steps_per_second'] = steps_per_second
    self._summary_manager.write_summaries(train_output)
    self._summary_manager.flush()

  def _maybe_save_checkpoint(self, check_interval: bool = True):
    """Conditionally saves a checkpoint.

    A checkpoint is saved if a `CheckpointManager` is available, and if the
    required number of steps has elapsed since the last checkpoint was saved
    (although this condition can be disabled by setting `check_interval=False`).

    :param check_interval: Whether to check if the checkpoint interval has fully
        elapsed. If `False`, a checkpoint is saved regardless of the elapsed
        steps since the most recent checkpoint, unless no `checkpoint_manager`
        was provided to `Controller.__init__`.
    :return: A boolean indicating whether a checkpoint was saved.
    """
    if self._checkpoint_manager and \
       self._checkpoint_manager.checkpoint_interval:
      ckpt_path = self._checkpoint_manager.save(
          checkpoint_number=self._global_step.numpy(),
          check_interval=check_interval)
      if ckpt_path is not None:
        _log('Save checkpoint to {}'.format(ckpt_path))
        return True
    return False

  def _require(self, attribute, for_method):
    if getattr(self, attribute, None) is None:
      raise ValueError('`{a}` is not set. Pass `{a}` to `Controller.__init__` '
                       'before calling `{f}()`'.format(a=attribute,
                                                       f=for_method))


class StepTimer:
  """Utility class for measuring steps/second."""

  def __init__(self, step):
    self._step = step
    self._last_iteration = None
    self._last_time = None
    self.start()

  def start(self):
    self._last_iteration = self._step.numpy()
    self._last_time = time.time()

  def steps_per_second(self, restart=True):
    value = ((self._step.numpy() - self._last_iteration) /
             (time.time() - self._last_time))
    if restart:
      self.start()
    return value
