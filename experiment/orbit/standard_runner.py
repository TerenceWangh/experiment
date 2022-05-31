"""AbstractTrainer/Evaluator subclasses with added functionality.

The classes in this module provide some additional structure to the bare
`AbstractTrainer`/`AbstractEvaluator` APIs.

Both `StandardTrainer` and `StandardEvaluator` split the train/eval loops into
"begin", "step", and "end" methods, and provide an implementation of the loop
itself that makes calls to the relevant step method.

`StandardTrainer` supports running the loop using the TF while loop construct
for added performance (particularly on TPUs). It additionally provides some
functionality to make writing summaries from inside a model more performant when
running on TPUs.

These classes are intended to work well in common settings, however there may
be use cases these classes don't support (for instance, `StandardEvaluator` in
particular doesn't support running full evaluations over multiple different eval
datasets). Users are encouraged to simply fall back to custom `AbstractTrainer`
and `AbstractEvaluator` subclasses in these cases.
"""

import abc
from typing import Any, Optional
import dataclasses
import tensorflow as tf

from experiment.orbit import runner
from experiment.orbit.utils import loop_fns


@dataclasses.dataclass(frozen=True)
class StandardTrainerOptions:
  """Advanced options for `orbit.StandardTrainer`.

  :var use_tf_function: A boolean indicating whether to apply
      `tf.function` to the training loop. This will only affect the body of the
      loop (involving `train_step`); `train_loop_begin` and `train_loop_end`
      will always be run in eager mode.
  :var use_tf_while_loop: A boolean indicating whether to run the training
      loop using a `tf.while_loop`. If `True`, `use_tf_function` must also be
      `True`.
  :var use_tpu_summary_optimization: A boolean indicating whether to enable a
      performance optimization for summaries in TPUs. Writing summaries
      conditionally with outside compilation on TPUs can be extremely slow. If
      `True`, this optimization creates two `tf.function`s with two XLA programs
      (one with summary calls, and one without). The program with summaries runs
      only for one step when summaries should be recorded.
  """
  use_tf_function: bool = True
  use_tf_while_loop: bool = True
  use_tpu_summary_optimization: bool = False


class StandardTrainer(runner.AbstractTrainer, metaclass=abc.ABCMeta):
  """Implements standard functionality on top of the AbstractTrainer API.

  This class structures the training "inner loop" roughly as follows:
      train_loop_begin()
      for _ in range(num_steps):
        train_step(train_iterator)
      return train_loop_end()

  Calls to `train_loop_begin` and `train_loop_end` are always done in eager
  mode, while the loop/`train_step` may be implemented using `tf.while` and/or
  `tf.function`, as determined by the `options` passed to `__init__`.
  """

  def __init__(self, train_dataset,
               options: Optional[StandardTrainerOptions]=None):
    """Initializes the `StandardTrainer` instance.

    :param train_dataset: A `tf.nest`-compatible structure of `tf.data.Dataset`
        or `DistributedDataset`.
    :param options: An `orbit.StandardTrainerOptions` instance.
    """
    options = options or StandardTrainerOptions()
    if options.use_tf_while_loop and not options.use_tf_function:
      raise ValueError('`use_tf_while_loop=True` and `use_tf_function=False` '
                       'is not supported.')
    if options.use_tpu_summary_optimization and not options.use_tf_while_loop:
      raise ValueError('`use_tpu_summary_optimization=True` and '
                       '`use_tf_while_loop=False` is not supported.')

    self._train_options = options
    self._train_dataset = train_dataset
    self._train_iterator = None
    self._train_loop_fn = None

  def create_train_loop_fn(self):
    """Creates a training loop from the current step function and options.

    :return: The train loop function, i.e. wrapper of multiple train steps.
    """
    train_step_fn = self.train_step
    if self._train_options.use_tf_while_loop:
      loop_fn = loop_fns.create_tf_while_loop_fn(train_step_fn)
      if self._train_options.use_tpu_summary_optimization:
        loop_fn = loop_fns.LoopFnWithSummaries(loop_fn)
      else:
        loop_fn = tf.function(loop_fn)
    else:
      if self._train_options.use_tf_function:
        train_step_fn = tf.function(train_step_fn)
      loop_fn = loop_fns.create_loop_fn(train_step_fn)
    return loop_fn

  def train(self, num_steps: tf.Tensor) -> Optional[runner.Output]:
    """Implements `num_steps` steps of training.

    :param num_steps: The number of training steps to run. This corresponds
        directly to the number of calls made to `train_step`.
    :return: The output of `train_loop_end`.
    """
    self.train_loop_begin()

    if self._train_loop_fn is None:
      self._train_loop_fn = self.create_train_loop_fn()

    if self._train_iterator is None:
      self._train_iterator = tf.nest.map_structure(iter, self._train_dataset)

    self._train_loop_fn(self._train_iterator, num_steps)
    return self.train_loop_end()

  def train_loop_begin(self):
    """Called once at the beginning of the training loop.

    This method is always called in eager mode, and is a good place to reset
    metrics that accumulate values over multiple steps of training.

    Note that this method is called before dataset iterator creation.
    """
    pass

  @abc.abstractmethod
  def train_step(self, iterator):
    """Implements one step of training.

    What a "step" consists of is up to the implementer. When using distribution
    strategies, the call to this method takes place in the "cross-replica
    context" for generality, to allow e.g. multiple iterator dequeues and calls
    to `strategy.run`.

    Note that if `use_tf_function=True`, all the code inside `train_step` should
    be compatible with `tf.function` tracing (and in particular, any state
    modifications involving `self` should be avoided). In some cases, non-
    `tf.function` compatible code can be moved to `train_loop_begin` or
    `train_loop_end`, which always execute eagerly.

    :param iterator: A `tf.nest`-compatible structure of `tf.data.Iterator` or
        `DistributedIterator`. The structure of this input matches the structure
        of `train_dataset` as passed to `__init__`.
    """
    pass

  def train_loop_end(self) -> Optional[runner.Output]:
    """Called once at the end of the training loop.

    This method is always called in eager mode, and is a good place to get
    metric results. The value returned from this function will be returned as-is
    from the `train` method implementation provided by `StandardTrainer`.

    :return: The function may return a dictionary of `Tensors`, which will be
        written to logs and as TensorBoard summaries. It can also be a
        nested dictionary, yielding a hierarchy of summary directories.
    """
    pass

  @property
  def train_dataset(self):
    """The current training dataset."""
    return self._train_dataset

  @train_dataset.setter
  def train_dataset(self, train_dataset):
    """Sets a new training dataset, replacing the current one.

    Any unprocessed examples in the current dataset are discarded.

    :param train_dataset: A `tf.nest`-compatible structure of `tf.data.Dataset`
        or `DistributedDataset`.
    """
    self._train_dataset = train_dataset
    self._train_iterator = None



@dataclasses.dataclass(frozen=True)
class StandardEvaluatorOptions:
  """Advanced options for the `orbit.StandardEvaluator`.

  :var use_tf_function: A boolean indicating whether to apply `tf.function` to
      the evaluation loop. This will only affect the body of the loop (involving
      `eval_step`); `eval_loop_begin` and `eval_loop_end` will always be run
      in eager mode.
  :var use_tf_while_loop: A boolean indicating whether to run the evaluation
      loop using a `tf.while_loop`. If `True`, `use_tf_function` must also be
      `True`.
  :var recreate_iterator_for_each_eval: A boolean indicating whether to recreate
      a new iterator for the evaluation dataset before each round of evaluation,
      which implies each round of evaluation starts from the beginning of
      the evaluation dataset. For example, the evaluation dataset is
      `[1, 2, 3, 4]`, batch size is 1 and evaluation steps is 2. If `True`, the
      data to be evaluated is [1, 2] every time. If `False`, the iterator
      state is maintained between calls to `StandardEvaluator.evaluate()`.
  """
  use_tf_function: bool = True
  use_tf_while_loop: bool = False
  recreate_iterator_for_each_eval: bool = True


class StandardEvaluator(runner.AbstractEvaluator, metaclass=abc.ABCMeta):
  """Implements the standard functionality of AbstractEvaluator APIs.

  This class structures evaluation roughly as follows:

    state = eval_begin()
    for _ in range(num_steps):
      step_outputs = eval_step(eval_iterator)
      state = eval_reduce(state, step_outputs)
    return eval_end(state)

  Calls to `eval_begin` and `eval_end` are always done in eager
  mode, while `eval_step` may be compiled with `tf.function` as determined by
  the `options` passed to `__init__`. `eval_reduce` is in eager mode if
  `use_tf_while_loop=False` in `StandardEvaluatorOptions`, but in graph mode if
  `use_tf_while_loop=True`.

  This class does not support completely evaluating multiple different datasets
  (i.e., where every example of each dataset should be processed, as opposed to
  running for a fixed number of evaluation steps). A custom `AbstractEvaluator`
  is recommended in this case.
  """

  def __init__(self, eval_dataset,
               options: Optional[StandardEvaluatorOptions]=None):
    """Initializes the `StandardEvaluator` instance.

    :param eval_dataset: A `tf.nest`-compatible structure of `tf.data.Dataset`
        or `DistributedDataset`. On TPUs, if users want to exhaust the dataset
        without specifying number of eval steps, it is recommended to set
        `drop_remainder=False` when batching the dataset, so the infrastructure
        can handle the last partial batch properly.
    :param options: An `orbit.StandardEvaluatorOptions` instance.
    """
    options = options or StandardEvaluatorOptions()
    if options.use_tf_while_loop and not options.use_tf_function:
      raise ValueError('`use_tf_while_loop=True` and `use_tf_function=False` '
                       'is not supported.')

    self._eval_options = options
    self._eval_dataset = eval_dataset
    self._eval_iterator = None
    self._eval_loop_fn = None

  def create_eval_loop_fn(self, has_state: bool):
    """Creates an eval loop from the current step function and options.

    :param has_state: If the step function has state, state will be kept in the
        loop.
    :return: The eval loop function, i.e. wrapper of multiple eval steps.
    """
    eval_step_fn = self.eval_step
    if self._eval_options.use_tf_while_loop:
      # tf.while_loop doesn't support `None` as a loop input even when it is not
      # used inside the loop. To workaround this limitation, we have to build
      # two tf.functions for it.
      if has_state:
        loop_fn = loop_fns.create_tf_while_loop_fn_with_state(eval_step_fn)
      else:
        loop_fn = loop_fns.create_tf_while_loop_fn(eval_step_fn)
      loop_fn = tf.function(loop_fn)
    else:
      if self._eval_options.use_tf_function:
        eval_step_fn = tf.function(eval_step_fn)
      loop_fn = loop_fns.create_loop_fn(eval_step_fn)
    return loop_fn

  def evaluate(self, num_steps: tf.Tensor) -> Optional[runner.Output]:
    """Implements `num_steps` steps of evaluation

    :param num_steps: The number of evaluation steps to run. When this is -1,
        evaluation proceeds until a call to `eval_step` raises a `StopIteration`
        or `tf.errors.OutOfRangeError`.
    :return: The output of `self.eval_end()`.
    :raise ValueError: If `options.use_tf_while_loop` is `True` and `num_steps`
        is unspecified.
    """
    if self._eval_options.use_tf_while_loop and num_steps == -1:
      raise ValueError('Looping until exhausted is not supported if '
                       '`options.use_tf_while_loop` us `True`.')

    outputs = self.eval_begin() # pylint: disable=assignment-from-no-return

    has_state = outputs is not None
    if self._eval_loop_fn is None:
      self._eval_loop_fn = self.create_eval_loop_fn(has_state)

    # If `recreate_iterator_for_each_eval` is `True`, `self._eval_iter` is
    # always None.
    if self._eval_iterator is None:
      eval_iterator = tf.nest.map_structure(iter, self._eval_dataset)
      if not self._eval_options.recreate_iterator_for_each_eval:
        self._eval_iterator = eval_iterator
    else:
      eval_iterator = self._eval_iterator

    if self._eval_options.use_tf_while_loop and not has_state:
      self._eval_loop_fn(eval_iterator, num_steps)
    else:
      outputs = self._eval_loop_fn(
        eval_iterator, num_steps, state=outputs, reduce_fn=self.eval_reduce)

    if outputs is None:
      return self.eval_end()
    else:
      return self.eval_end(outputs)

  def eval_begin(self) -> Any:
    """Called once at the beginning of the evaluation.

    This method is always called in eager mode, and is a good place to reset
    metrics that accumulate values over the course of evaluation.

    Note that this method is called before dataset iterator creation.

    :return: An value to pass as the `state` argument to `eval_reduce`.
    """
    pass

  @abc.abstractmethod
  def eval_step(self, iterator) -> Any:
    """Implements one step of evaluation.

    What a "step" consists of is up to the implementer. When using distribution
    strategies, the call to this method takes place in the "cross-replica
    context" for generality, to allow e.g. multiple iterator dequeues and calls
    to `strategy.run`.

    Note that if `use_tf_function=True`, all the code inside `eval_step` should
    be compatible with `tf.function` tracing (and in particular, any state
    modifications involving `self` should be avoided). In some cases, non-
    `tf.function` compatible code can be moved to `eval_loop_begin`,
    `eval_reduce`, or `eval_loop_end`, which always execute eagerly.

    :param iterator: A `tf.nest`-compatible structure of `tf.data.Iterator` or
        `DistributedIterator`.
    :return: An output which is passed as `step_outputs` argument into
        `eval_reduce` function.
    """
    pass

  def eval_end(self, *args) -> Optional[runner.Output]:
    """Called at the end of the evaluation.

    Called once at the end of evaluation.

    This method is always called in eager mode, and is a good place to get
    metric results. The value returned from this function will be returned as-is
    from the `evaluate` method implementation provided by `StandardEvaluator`.

    :param args: The outputs from `eval_reduce` for the last eval step, if they
        are non-`None` (if they are `None`, nothing is passed).
    :return: The function may return a dictionary of `Tensors`, which will be
        written to logs and as TensorBoard summaries. It can also be a
        nested dictionary, yielding a hierarchy of summary directories.
    """
    pass

  def eval_reduce(self,
                  state: Optional[Any]=None,
                  step_outputs: Optional[runner.Output]=None) -> Any:
    """A function to perform per-step reduction on the evaluation outputs.

    This is useful for passing state throughout evaluation, especially in cases
    where maintaining or accumulating state is hard to accomplish using
    `tf.metrics.Metric` or other `tf.Variable`-based approaches. For instance,
    it can be used to easily accumulate all per-example losses from the full
    evaluation for subsequent processing in `eval_end()`.

    :param state: A state being maintained throughout the evaluation.
    :param step_outputs: Outputs from the current evaluation step.
    :return: An output which is passed as the `state` argument to this function
        for the next step. After evaluation is finished, the output from last
        step will be passed to `eval_end`.
    """
    pass

  @property
  def eval_dataset(self):
    """The current evaluation dataset."""
    return self._eval_dataset

  @eval_dataset.setter
  def eval_dataset(self, eval_dataset):
    """Sets a new eval dataset, replacing the current one.

    Any unprocessed examples in the current dataset are discarded.

    :param eval_dataset: A `tf.nest`-compatible structure of `tf.data.Dataset`
        or `DistributedDataset`.
    """
    self._eval_dataset = eval_dataset
    self._eval_iterator = None
