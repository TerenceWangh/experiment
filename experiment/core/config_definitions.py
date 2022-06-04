"""Common configuration settings."""

import dataclasses
from typing import Optional, Sequence, Union

from experiment.hyperparams import base_config
from experiment.optimization import OptimizationConfig
from experiment.model.privacy import configs as dp_configs


@dataclasses.dataclass
class DataConfig(base_config.Config):
  """The base configuration for building datasets.
  """
  input_path: Union[Sequence[str], str, base_config.Config] = ""
  tfds_name: str = ""
  tfds_split: str = ""
  global_batch_size: int = 0
  is_training: bool = None
  drop_remainder: bool = True
  shuffle_buffer_size: int = 100
  cache: bool = False
  cycle_length: Optional[int] = None
  block_length: int = 1
  deterministic: Optional[bool] = None
  sharding: bool = True
  enable_tf_data_service: bool = False
  tf_data_service_address: Optional[str] = None
  tf_data_service_job_name: Optional[str] = None
  tfds_data_dir: str = ""
  tfds_as_supervised: bool = False
  tfds_skip_decoding_feature: str = ""
  seed: Optional[int] = None
  prefetch_buffer_size: Optional[int] = None


@dataclasses.dataclass
class RuntimeConfig(base_config.Config):
  """High-level configurations for Runtime.

  These include parameters that are not directly related to the experiment,
  e.g. directories, accelerator type, etc.
  """
  distribution_strategy: str = "mirrored"
  enable_xla: bool = False
  gpu_thread_mode: Optional[str] = None
  dataset_num_private_threads: Optional[int] = None
  per_gpu_thread_count: int = 0
  tpu: Optional[str] = None
  num_gpus: int = 0
  worker_hosts: Optional[str] = None
  task_index: int = -1
  all_reduce_alg: Optional[str] = None
  num_packs: int = 1
  mixed_precision_dtype: Optional[str] = None
  loss_scale: Optional[Union[str, float]] = None
  run_eagerly: bool = False
  batchnorm_spatial_persistent: bool = False

  # XLA runtime params.
  # XLA params are only applied to the train_step.
  # These augments can improve training speed. They can also improve eval, but
  # may reduce usability and users would need to make changes to code.

  # Whether to enable XLA dynamic padder
  # infrastructure to handle dynamic shapes inputs inside XLA. True by
  # default. Disabling this may cause correctness issues with dynamic shapes
  # inputs, as XLA will just assume the inputs are with padded shapes. However
  # users can optionally set it to False to improve device time if masking is
  # already handled in the user side.
  # If None, will respect XLA default.
  tpu_enable_xla_dynamic_padder: Optional[bool] = None

  # Global model parallelism configurations.
  num_cores_per_replica: int = 1
  default_shard_dim: int = -1

  def model_parallelism(self):
    return dict(
        num_cores_per_replica=self.num_cores_per_replica,
        default_shard_dim=self.default_shard_dim)


@dataclasses.dataclass
class TrainerConfig(base_config.Config):
  """Configuration for trainer.
  """
  optimizer_config: OptimizationConfig = OptimizationConfig()
  # Orbit settings.
  train_tf_while_loop: bool = True
  train_tf_function: bool = True
  eval_tf_function: bool = True
  eval_tf_while_loop: bool = False
  allow_tpu_summary: bool = False
  # Trainer intervals.
  steps_per_loop: int = 1000
  summary_interval: int = 1000
  checkpoint_interval: int = 1000
  # Checkpoint manager.
  max_to_keep: int = 5
  continuous_eval_timeout: int = 60 * 60
  # Train/Eval routines.
  train_steps: int = 0
  # Sets validation steps to be -1 to evaluate the entire dataset.
  validation_steps: int = -1
  validation_interval: int = 1000
  # Best checkpoint export.
  best_checkpoint_export_subdir: str = ""
  best_checkpoint_eval_metric: str = ""
  best_checkpoint_metric_comp: str = "higher"
  # Blowup recovery.
  loss_upper_bound: float = 1e6
  recovery_begin_steps: int = 0  # Enforcing the loss bound after these steps.
  # When max trials < 0, no recovery module; max trials = 0, we will check
  # the condition and fail the job if the condition happens; max trials > 0,
  # we will retore the model states.
  recovery_max_trials: int = 0
  validation_summary_subdir: str = "validation"


@dataclasses.dataclass
class TaskConfig(base_config.Config):
  """Config passed to task."""
  init_checkpoint: str = ""
  model: Optional[base_config.Config] = None
  train_data: DataConfig = DataConfig()
  validation_data: DataConfig = DataConfig()
  name: Optional[str] = None
  # Configs for differential privacy
  # These configuration are only effective if you use create_optimizer in
  # tensorflow_models/official/core/base_task.py
  differential_privacy_config: Optional[
      dp_configs.DifferentialPrivacyConfig] = None


@dataclasses.dataclass
class ExperimentConfig(base_config.Config):
  """Top-level configuration."""
  task: TaskConfig = TaskConfig()
  trainer: TrainerConfig = TrainerConfig()
  runtime: RuntimeConfig = RuntimeConfig()

