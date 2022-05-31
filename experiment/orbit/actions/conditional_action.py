"""Provides a `ConditionalAction` abstraction."""
from typing import Any, Callable, Sequence, Union
import tensorflow as tf

from experiment.orbit import controller
from experiment.orbit import runner

Condition = Callable[[runner.Output], Union[bool, tf.Tensor]]


def _as_sequence(maybe_sequence: Union[Any, Sequence[Any]]) -> Sequence[Any]:
  if isinstance(maybe_sequence, Sequence):
    return maybe_sequence
  return [maybe_sequence]


class ConditionalAction:
  """Represents an action that is only taken when a given condition is met.

  This class is itself an `Action` (a callable that can be applied to train or
  eval outputs), but is intended to make it easier to write modular and reusable
  conditions by decoupling "when" something happens (the condition) from "what"
  happens (the action).
  """
  def __init__(self, condition: Condition,
               action: Union[controller.Action, Sequence[controller.Action]]):
    """Initializes the instance.

    :param condition: A callable accepting train or eval outputs and returning
        a bool.
    :param action: The action (or optionally sequence of actions) to perform
        when `condition` is met.
    """
    self._condition = condition
    self._action = action

  def __call__(self, output: runner.Output) -> None:
    if self._condition(output):
      for action in _as_sequence(self._action):
        action(output)
