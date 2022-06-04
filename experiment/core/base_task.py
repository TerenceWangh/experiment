"""Defines the base task abstraction."""

import abc
import functools
from typing import Optional

from absl import logging
import tensorflow as tf

from experiment.core import config_definitions
from experiment import optimization
from experiment.model import performance
from experiment.model.privacy import configs
from experiment.model.privacy import ops

OptimizationConfig = optimization.OptimizationConfig
RuntimeConfig = config_definitions.RuntimeConfig
DifferentialPrivacyConfig = configs.DifferentialPrivacyConfig


class Task(tf.Module, metaclass=abc.ABCMeta):
  """A single-replica view of training procedure.

  Tasks provide artifacts for training/validation procedures, including
  loading/iterating over Datasets, training/validation steps, calculating the
  loss and customized metrics with reduction.
  """
  # Special keys in train/validate step returned logs.
  loss = 'loss'

  def __init__(self,
               params,
               logging_dir: Optional[str] = None,
               name: Optional[str] = None):
    """Task initialization.

    Parameters
    ==========
    params : dataclass or ConfigDict or namedtuple
        the task configuration instance, which can be any of dataclass,
        ConfigDict, namedtuple, etc.
    logging_dir : str, optional
        Where the model, summaries etc. will be saved. You can also write
        additional stuff in this directory.
    name : str, optional
        The task name.
    """
    super(Task, self).__init__(name=name)
    self._task_config = params
    self._logging_dir = logging_dir

  @property
  def task_config(self):
    return self._task_config

  @property
  def logging_dir(self):
    return self._logging_dir

  @classmethod
  def create_optimizer(cls,
                       optimizer_config: OptimizationConfig,
                       runtime_config: Optional[RuntimeConfig] = None,
                       dp_config: Optional[DifferentialPrivacyConfig] = None):
    """Creates an TF optimizer from configuration.

    Parameters
    ==========
    optimizer_config : OptimizationConfig
        The parameters of the Optimization settings.
    runtime_config : RuntimeConfig, optional
        The parameters of the runtime.
    dp_config : DifferentialPrivacyConfig, optional
        The parameter of  differential privacy.

    Returns
    =======
    tf.optimizers.Optimizer
        The optimizer object.
    """
    gradient_transformers = None
    if dp_config is not None:
      logging.info('Adding differential privacy transform with config '
                   '{}.'.format(dp_config.as_dict()))
      noise_stddev = dp_config.clipping_norm * dp_config.noise_multiplier
      gradient_transformers = [
          functools.partial(
              ops.clip_l2_norm, l2_norm_clip=dp_config.clipping_norm),
          functools.partial(
              ops.add_noise, noise_stddev=noise_stddev),
      ]

    opt_factory = optimization.OptimizerFactory(optimizer_config)
    optimizer = opt_factory.build_optimizer(
        opt_factory.build_learning_rate(),
        gradient_transformers=gradient_transformers)
    # Configuring optimizer when loss_scale is set in runtime config. This helps
    # avoiding overflow/underflow for float16 computations.
    if runtime_config:
      optimizer = performance.configure_optimizer(
          optimizer,
          use_float16=runtime_config.mixed_precision_dtype == "float16",
          loss_scale=runtime_config.loss_scale)
    return optimizer


  def initialize(self, model: tf.keras.Model):
    """[Optional] A callback function used as CheckpointManager's init_fn.

    This function will be called when no checkpoint is found for the model.
    If there is a checkpoint, the checkpoint will be loaded and this function
    will not be called. You can use this callback function to load a pretrained
    checkpoint, saved under a directory other than the model_dir.

    Parameters
    model : tf.keras.Model
        The keras.Model built or used by this task.
    """
    ckpt_dir_or_file = self.task_config.init_checkpoint
    logging.info('Trying to load pretrained checkpoint from %s',
                 ckpt_dir_or_file)
    if ckpt_dir_or_file and tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)
    if not ckpt_dir_or_file:
      logging.info('No checkpoint file found from %s. Will not load.',
                   ckpt_dir_or_file)
      return

    if hasattr(model, 'checkpoint_items'):
      checkpoint_items = model.checkpoint_items
    else:
      checkpoint_items = dict(model=model)

    ckpt = tf.train.Checkpoint(**checkpoint_items)
    status = ckpt.read(ckpt_dir_or_file)
    status.expect_partial().assert_existing_objects_matched()
    logging.info('Finished loading pretrained checkpoint from %s.',
                 ckpt_dir_or_file)

  def build_model(self) -> tf.keras.Model:
    """[Optional] Creates model architecture.

    Returns
    =======
    tf.keras.Model
        A model instance.
    """ # pytype: disable=bad-return-type  # typed-keras

  @abc.abstractmethod
  def build_inputs(self,
                   params,
                   input_context: Optional[tf.distribute.InputContext] = None):
    """ Returns a dataset or a nested structure of dataset functions.

    Dataset functions define per-host datasets with the per-replica batch size.
    With distributed training, this method runs on remote hosts.

    Parameters
    ==========
    params
        The hyperparams to create input pipelines, which can be any of
        dataclass, ConfigDict, namedtuple, etc.
    input_context : tf.distribute.InputContext, optional
        The distribution input pipeline context.

    Returns
    =======
    function
        A nested structure of per-replica input functions.
    """


  def build_losses(self,
                   labels,
                   model_outputs,
                   aux_losses=None) -> tf.Tensor:
    """Standard interface to compute loss

    Parameters
    ==========
    labels : tf.Tensor
        The optional label Tensor.
    model_outputs : tf.Tensor
        A nested structure of output tensors.
    aux_losses : tf.Tensor, optional
        The auxiliary loss tensor, i.e. `losses` in keras.Model.

    Returns
    =======
    tf.Tensor
        The total loss tensor.
    """
    del model_outputs, labels

    if aux_losses is None:
      losses = [tf.constant(0.0, dtype=tf.float32)]
    else:
      losses = aux_losses

    total_loss = tf.add_n(losses)
    return total_loss

  def build_metrics(self,
                    training: bool = True):
    """Gets streaming metrics for training/validation."""
    del training
    return []

  def process_metrics(self, metrics, labels, model_outputs, **kwargs):
    """Process and update metrics.

    Called when using custom training loop API.

    Parameters
    ==========
    metrics :
        A nested structure of metrics objects. The return of function
        self.build_metrics.
    labels : tf.Tensor
        A tensor or a nested structure of tensors.
    model_outputs : tf.Tensor
        A tensor or a nested structure of tensors. For example, output of the
        keras model built by self.build_model.
    kwargs : dict
        Other arguments.
    """
    for metric in metrics:
      metric.update_state(labels, model_outputs)

  def process_compiled_metrics(self, compiled_metrics, labels, model_outputs):
    """Process and update compiled_metrics.

    call when using compile/fit API.

    Parameters
    ==========
    compiled_metrics :
        The compiled metrics (model.compiled_metrics).
    labels : tf.Tensor
        A tensor or a nested structure of tensors.
    model_outputs : tf.Tensor
        A tensor or a nested structure of tensors. For example, output of the
        keras model built by self.build_model.
    """
    compiled_metrics.update_state(labels, model_outputs)

  def train_step(self,
                 inputs,
                 model: tf.keras.Model,
                 optimizer: tf.keras.optimizers.Optimizer,
                 metrics=None):
    """Does forward and backward.

    With distribution strategies, this method runs on devices.

    Parameters
    ==========
    inputs : dict of tf.Tensor
        A dictionary of input tensors.
    model : tf.keras.Model
        The model, forward pass definition.
    optimizer : tf.keras.optimizers.Optimizer
        The optimizer for this training step.
    metrics : tf.Tensor, optional
        A nested structure of metrics objects.

    Returns
    =======
    dict
        A dictionary of logs.
    """
    if isinstance(inputs, tuple) and len(inputs) == 2:
      features, labels = inputs
    else:
      features, labels = inputs, inputs
    with tf.GradientTape() as tape:
      outputs = model(features, training=True)
      # Computes per-replica loss.
      if model.compiled_loss:
        loss = model.compiled_loss(
            labels, outputs, regularization_losses=model.losses)
        loss += self.build_losses(
            labels=labels, model_outputs=outputs, aux_losses=None)
      else:
        loss = self.build_losses(
            labels=labels, model_outputs=outputs, aux_losses=model.losses)
      # Scales loss as the default gradients allreduce performs sum inside the
      # optimizer.
      scaled_loss = loss / tf.distribute.get_strategy().num_replicas_in_sync

      # For mixed precision, when a LossScaleOptimizer is used, the loss is
      # scaled to avoid numeric underflow.
      if isinstance(optimizer,
                    tf.keras.mixed_precision.LossScaleOptimizer):
        scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    tvars = model.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)

    if isinstance(optimizer,
                  tf.keras.mixed_precision.LossScaleOptimizer):
      grads = optimizer.get_unscaled_gradients(grads)
    optimizer.apply_gradients(list(zip(grads, tvars)))
    logs = {self.loss: loss}
    if metrics:
      self.process_metrics(metrics, labels, outputs)
    if model.compiled_metrics:
      self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
      logs.update({m.name: m.result() for m in metrics or []})
      logs.update({m.name: m.result() for m in model.metrics})
    return logs

  def validation_step(self, inputs, model: tf.keras.Model, metrics=None):
    """Validation step.

    With distribution strategies, this method runs on devices.

    Parameters
    ==========
    inputs : dict of tf.Tensor
        A dictionary of input tensors.
    model : tf.keras.Model
        The keras.Model.
    metrics : tf.Tensor
        A nested structure of metrics objects.

    Returns
    =======
    dict
        A dictionary of logs.
    """
    if isinstance(inputs, tuple) and len(inputs) == 2:
      features, labels = inputs
    else:
      features, labels = inputs, inputs
    outputs = self.inference_step(features, model)
    loss = self.build_losses(
        labels=labels, model_outputs=outputs, aux_losses=model.losses)
    logs = {self.loss: loss}
    if metrics:
      self.process_metrics(metrics, labels, outputs)
    if model.compiled_metrics:
      self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
      logs.update({m.name: m.result() for m in metrics or []})
      logs.update({m.name: m.result() for m in model.metrics})
    return logs

  def inference_step(self, inputs, model: tf.keras.Model):
    """Performs the forward step.

    With distribution strategies, this method runs on devices.

    Parameters
    ==========
    inputs : dict of tf.Tensor
        A dictionary of input tensors.
    model : tf.keras.Model
        The keras.Model.

    Returns
    =======
    tf.Tensor
        model outputs.
    """
    return model(inputs, training=False)

  def aggregate_logs(self, state, step_logs):
    """Optional aggregation over logs returned from a validation step.

    Given step_logs from a validation step, this function aggregates the logs
    after each eval_step() (see eval_reduce() function in
    official/core/base_trainer.py). It runs on CPU and can be used to aggregate
    metrics during validation, when there are too many metrics that cannot fit
    into TPU memory. Note that this may increase latency due to data transfer
    between TPU and CPU. Also, the step output from a validation step may be a
    tuple with elements from replicas, and a concatenation of the elements is
    needed in such case.

    Parameters
    ==========
    state : list
        The current state of training, for example, it can be a sequence of
        metrics.
    step_logs : dict
        Logs from a validation step. Can be a dictionary.
    """
    pass

  def reduce_aggregated_logs(self,
                             aggregated_logs,
                             global_step: Optional[tf.Tensor] = None):
    """Optional reduce of aggregated logs over validation steps.

    This function reduces aggregated logs at the end of validation, and can be
    used to compute the final metrics. It runs on CPU and in each eval_end() in
    base trainer (see eval_end() function in official/core/base_trainer.py).

    Parameters
    ==========
    aggregated_logs : dict
        Aggregated logs over multiple validation steps.
    global_step : tf.Tensor, optional
        An optional variable of global step.

    Returns
    =======
    dict
        A dictionary of reduced results.
    """
    return {}
