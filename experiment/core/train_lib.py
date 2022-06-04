import os
from typing import Any, Mapping, Optional, Tuple

from absl import logging
import tensorflow as tf

from experiment import orbit
from experiment.core import actions
from experiment.core import base_task
from experiment.core import base_trainer
from experiment.core import config_definitions
from experiment.core import train_utils

maybe_create_best_ckpt_exporter = train_utils.maybe_create_best_ckpt_exporter


def run_experiment(
    distribution_strategy: tf.distribute.Strategy,
    task: base_task.Task,
    mode: str,
    params: config_definitions.ExperimentConfig,
    model_dir: str,
    run_post_eval: bool = False,
    save_summary: bool = True,
    trainer: Optional[base_trainer.Trainer] = None,
    controller_cls=orbit.Controller
) -> Tuple[tf.keras.Model, Mapping[str, Any]]:
  """Runs train/eval configured by the experiment params.

  Parameters
  ==========
  distribution_strategy : tf.distribute.Strategy
      A distribution distribution_strategy.
  task : base_task.Task
      A task instance.
  mode : str
      Specifying the mode. Can be 'train', 'eval', 'train_and_eval' or
      'continuous_eval'.
  params : config_definitions.ExperimentConfig
      The  ExperimentConfig instance.
  model_dir : str
      A path to store model checkpoints and summaries.
  run_post_eval : bool, default False
      Whether to run post eval once after training, metrics logs are returned.
  save_summary : bool, default True
      Whether to save train and validation summary.
  trainer : base_trainer.Trainer, optional
      The Trainer instance. It should be created within the strategy.scope().
  controller_cls : orbit.Controller
      The controller class to manage the train and eval process. Must be a
      orbit.Controller subclass.

  Returns
  =======
  tuple
      model : tf.keras.Model
          The model instance.
      eval_logs : dict
          The eval metrics logs when run_post_eval is set to true, otherwise
          return {}.
  """
  with distribution_strategy.scope():
    if not trainer:
      trainer = train_utils.create_trainer(
          params,
          task,
          train='train' in mode,
          evaluate=('eval' in mode) or run_post_eval,
          checkpoint_exporter=maybe_create_best_ckpt_exporter(
              params, model_dir))

    if trainer.checkpoint:
      if model_dir is None:
        raise ValueError('model_dir must be specified, but got None')
      checkpoint_manager = tf.train.CheckpointManager(
          trainer.checkpoint,
          directory=model_dir,
          max_to_keep=params.trainer.max_to_keep,
          step_counter=trainer.global_step,
          checkpoint_interval=params.trainer.checkpoint_interval,
          init_fn=trainer.initialize)
    else:
      checkpoint_manager = None

    eval_summary_dir = os.path.join(
        model_dir, params.trainer.validation_summary_subdir)
    summary_interval = params.trainer.summary_interval

    controller = controller_cls(
        strategy=distribution_strategy,
        trainer=trainer if 'train' in mode else None,
        evaluator=trainer,
        global_step=trainer.global_step,
        steps_per_loop=params.trainer.steps_per_loop,
        checkpoint_manager=checkpoint_manager,
        summary_dir=os.path.join(model_dir, 'train') if save_summary else None,
        eval_summary_dir=eval_summary_dir if save_summary else None,
        summary_interval=summary_interval if save_summary else None,
        train_actions=actions.get_train_actions(
            params, trainer, model_dir, checkpoint_manager=checkpoint_manager),
        eval_actions=actions.get_eval_actions(params, trainer, model_dir))

    logging.info('Starts to execute mode: %s', mode)
    with distribution_strategy.scope():
      if mode == 'train':
        controller.train(steps=params.trainer.train_steps)
      elif mode == 'train_and_eval':
        controller.train_and_evaluate(
            train_steps=params.trainer.train_steps,
            eval_steps=params.trainer.validation_steps,
            eval_interval=params.trainer.validation_interval)
      elif mode == 'eval':
        controller.evaluate(steps=params.trainer.validation_steps)
      elif mode == 'continuous_eval':
        def timeout_fn():
          if trainer.global_step.numpy() >= params.trainer.train_steps:
            return True
          return False

        controller.evaluate_continuously(
            steps=params.trainer.validation_steps,
            timeout=params.trainer.continuous_eval_timeout,
            timeout_fn=timeout_fn)
      else:
        raise NotImplementedError('The mode is not implemented: %s' % mode)

  num_params = train_utils.try_count_params(trainer.model)
  if num_params is not None:
    logging.info(
        'Number of trainable params in model: %f Millions.',
        num_params / 10.**6)

  flops = train_utils.try_count_flops(trainer.model)
  if flops is not None:
    logging.info(
        'FLOPs (multi-adds) in model: %f Billions.', flops / 10.**9 / 2)

  if run_post_eval:
    with distribution_strategy.scope():
      return trainer.model, trainer.evaluate(
          tf.convert_to_tensor(params.trainer, params.trainer.validation_steps))
    return trainer.model, {}
