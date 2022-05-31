"""A global factory to register and access all registered tasks."""

from experiment.core import registry

_REGISTERED_TASK_CLS = {}


def register_task_cls(task_config_cls):
  """Decorates a factory of Tasks for lookup by a subclass of TaskConfig.

  This decorator supports registration of tasks as follows:

  ```
  @dataclasses.dataclass
  class MyTaskConfig(TaskConfig):
    # Add fields here.
    pass
  @register_task_cls(MyTaskConfig)
  class MyTask(Task):
    # Inherits def __init__(self, task_config).
    pass
  my_task_config = MyTaskConfig()
  my_task = get_task(my_task_config)  # Returns MyTask(my_task_config).
  ```

  Besides a class itself, other callables that create a Task from a TaskConfig
  can be decorated by the result of this function, as long as there is at most
  one registration for each config class.

  :param task_config_cls: a subclass of TaskConfig (*not* an instance of
      TaskConfig). Each task_config_cls can only be used for a single
      registration.
  :return: A callable for use as class decorator that registers the decorated
    class for creation from an instance of task_config_cls.
  """
  return registry.register(_REGISTERED_TASK_CLS, task_config_cls)


def get_task(task_config, **kwargs):
  """Creates a Task (of suitable subclass type) from task_config."""
  if task_config.BUILDER is not None:
    return task_config.BUILDER(task_config, **kwargs)
  return get_task_cls(task_config.__class__)(task_config, **kwargs)

def get_task_cls(task_config_cls):
  task_cls = registry.lookup(_REGISTERED_TASK_CLS, task_config_cls)
  return task_cls
