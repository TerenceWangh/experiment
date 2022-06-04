from absl import app, flags
import gin

from experiment.common import distribute_utils
from experiment.common import flags as tfm_flags
from experiment.core import task_factory
from experiment.core import train_lib
from experiment.core import train_utils
from experiment.model import performance

import tensorflow as tf
tf.get_logger().setLevel('INFO')

FLAGS = flags.FLAGS


def main(_):
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_params)
  params = train_utils.parse_configuration(FLAGS)
  model_dir = FLAGS.model_dir

  if 'train' in FLAGS.mode:
    # Pure eval modes do not output yaml files. Otherwise continuous eval job
    # may race against the train job for writing the same file.
    train_utils.serialize_config(params, model_dir)

  # Sets mixed_precision policy. Using 'mixed_float16' or 'mixed_bfloat16'
  # can have significant impact on model speeds by utilizing float16 in case of
  # GPUs. loss_scale takes effect only when dtype if `float16`.
  if params.runtime.mixed_precision_dtype:
    performance.set_mixed_precision_policy(params.runtime.mixed_precision_dtype)
  distribution_strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=params.runtime.distribution_strategy,
      all_reduce_alg=params.runtime.all_reduce_alg,
      num_gpus=params.runtime.num_gpus)
  with distribution_strategy.scope():
    task = task_factory.get_task(params.task, logging_dir=model_dir)

  train_lib.run_experiment(
      distribution_strategy=distribution_strategy,
      task=task,
      mode=FLAGS.mode,
      params=params,
      model_dir=model_dir)

  train_utils.save_gin_config(FLAGS.mode, model_dir)


if __name__ == '__main__':
  tfm_flags.define_flags()
  app.run(main)

