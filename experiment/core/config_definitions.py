"""Common configuration settings."""

import dataclasses
from typing import Optional, Sequence, Union

from experiment.hyperparams import base_config
from experiment.optimization import OptimizationConfig
from experiment.model.privacy import configs as dp_configs


@dataclasses.dataclass
class DataConfig(base_config.Config):
  """The base configuration for building datasets.

  :attribute input_path: The path to the input. It can be either (1) a str indicating a
      file path/pattern, or (2) a str indicating multiple file paths/patterns
      separated by comma (e.g "a, b, c" or no spaces "a,b,c"), or (3) a list of
      str, each of which is a file path/pattern or multiple file paths/patterns
      separated by comma, or (4) a dictionary of the previous three approaches
      for more advanced data mixing using named access. It should not be
      specified when the following `tfds_name` is specified.
  :attribute tfds_name: The name of the tensorflow dataset (TFDS). It should not be
      specified when the above `input_path` is specified.
  :attribute tfds_split: A str indicating which split of the data to load from TFDS. It
      is required when above `tfds_name` is specified.
  :attribute global_batch_size: The global batch size across all replicas.
  :attribute is_training: Whether this data is used for training or not. This flag is
      useful for consumers of this object to determine whether the data should
      be repeated or shuffled.
  :attribute drop_remainder: Whether the last batch should be dropped in the case it has
      fewer than `global_batch_size` elements.
  :attribute shuffle_buffer_size: The buffer size used for shuffling training data.
  :attribute cache: Whether to cache dataset examples. If `True`, we will cache the
      dataset after applying the decode_fn and parse_fn. It can be used to avoid
      re-reading from disk, re-decoding and re-parsing the example on the second
      epoch, but it requires significant memory overhead.
  :attribute cycle_length: The number of files that will be processed concurrently when
      interleaving files.
  :attribute block_length: The number of consecutive elements to produce from each input
      element before cycling to another input element when interleaving files.
  :attribute deterministic: A boolean controlling whether determinism should be enforced.
  :attribute sharding: Whether sharding is used in the input pipeline.
  :attribute enable_tf_data_service: A boolean indicating whether to enable tf.data
      service for the input pipeline.
  :attribute tf_data_service_address: The URI of a tf.data service to offload
      preprocessing onto during training. The URI should be in the format
      "protocol://address", e.g. "grpc://tf-data-service:5050". It can be
        overridden by `FLAGS.tf_data_service` flag in the binary.
  :attribute tf_data_service_job_name: The name of the tf.data service job. This argument
      makes it possible for multiple datasets to share the same job. The default
      behavior is that the dataset creates anonymous, exclusively owned jobs.
  :attribute tfds_data_dir: A str specifying the directory to read/write TFDS data.
  :attribute tfds_as_supervised: A bool. When loading dataset from TFDS, if True, the
      returned tf.data.Dataset will have a 2-tuple structure (input, label)
      according to builder.info.supervised_keys; if False, the default, the
      returned tf.data.Dataset will have a dictionary with all the features.
  :attribute tfds_skip_decoding_feature: A str to indicate which features are skipped for
      decoding when loading dataset from TFDS. Use comma to separate multiple
      features. The main use case is to skip the image/video decoding for better
      performance.
  :attribute seed: An optional seed to use for deterministic shuffling/preprocessing.
  :attribute prefetch_buffer_size: An int specifying the buffer size of prefetch
      datasets. If None, the buffer size is autotuned. Specifying this is useful
      in case autotuning uses up too much memory by making the buffer size too
      high.
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

  :attribute distribution_strategy: e.g. 'mirrored', 'tpu', etc.
  :attribute enable_xla: Whether or not to enable XLA.
  :attribute per_gpu_thread_count: thread count per GPU.
  :attribute gpu_thread_mode: Whether and how the GPU device uses its own threadpool.
  :attribute dataset_num_private_threads: Number of threads for a private threadpool
      created for all datasets computation.
  :attribute tpu: The address of the TPU to use, if any.
  :attribute num_gpus: The number of GPUs to use, if any.
  :attribute worker_hosts: comma-separated list of worker ip:port pairs for running
      multi-worker models with DistributionStrategy.
  :attribute task_index: If multi-worker training, the task index of this worker.
  :attribute all_reduce_alg: Defines the algorithm for performing all-reduce.
  :attribute num_packs: Sets `num_packs` in the cross device ops used in
      MirroredStrategy.  For details, see tf.distribute.NcclAllReduce.
  :attribute mixed_precision_dtype: dtype of mixed precision policy. It can be 'float32',
      'float16', or 'bfloat16'.
  :attribute loss_scale: The type of loss scale, or 'float' value. This is used when
      setting the mixed precision policy.
  :attribute run_eagerly: Whether or not to run the experiment eagerly.
  :attribute batchnorm_spatial_persistent: Whether or not to enable the spatial
      persistent mode for CuDNN batch norm kernel for improved GPU performance.
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

  :attribute optimizer_config: optimizer config, it includes optimizer, learning rate,
      and warmup schedule configuration.
  :attribute train_tf_while_loop: whether or not to use tf while loop.
  :attribute train_tf_function: whether or not to use tf_function for training loop.
  :attribute eval_tf_function: whether or not to use tf_function for eval.
  :attribute allow_tpu_summary: Whether to allow summary happen inside the XLA program
      runs on TPU through automatic outside compilation.
  :attribute steps_per_loop: number of steps per loop to report training metrics. This
      can also be used to reduce host worker communication in a TPU setup.
  :attribute summary_interval: number of steps between each summary.
  :attribute checkpoint_interval: number of steps between checkpoints.
  :attribute max_to_keep: max checkpoints to keep.
  :attribute continuous_eval_timeout: maximum number of seconds to wait between
      checkpoints, if set to None, continuous eval will wait indefinitely. This
      is only used continuous_train_and_eval and continuous_eval modes. Default
      value is 1 hrs.
  :attribute train_steps: number of train steps.
  :attribute validation_steps: number of eval steps. If -1, the entire eval dataset is
      used.
  :attribute validation_interval: number of training steps to run between evaluations.
  :attribute best_checkpoint_export_subdir: if set, the trainer will keep track of the
      best evaluation metric, and export the corresponding best checkpoint under
      `model_dir/best_checkpoint_export_subdir`. Note that this only works if
      mode contains eval (such as `train_and_eval`, `continuous_eval`, and
      `continuous_train_and_eval`).
  :attribute best_checkpoint_eval_metric: for exporting the best checkpoint, which
      evaluation metric the trainer should monitor. This can be any evaluation
      metric appears on tensorboard.
  :attribute best_checkpoint_metric_comp: for exporting the best checkpoint, how the
      trainer should compare the evaluation metrics. This can be either `higher`
      (higher the better) or `lower` (lower the better).
    validation_summary_subdir: A 'str', sub directory for saving eval summary.
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

