import json
import os
import pprint

import numpy as np
import tensorflow as tf

from experiment import hyperparams
from experiment.core import exp_factory
from experiment.core import train_utils
from experiment.utils.testing import test_utils


@exp_factory.register_config_factory('foo')
def foo():
  """Multitask experiment for test."""
  experiment_config = hyperparams.Config(
      default_params={
          'runtime': {
              'tpu': 'fake',
          },
          'task': {
              'model': {
                  'model_id': 'bar',
              },
          },
          'trainer': {
              'train_steps': -1,
              'validation_steps': -1,
          },
      })
  return experiment_config


class TrainUtilsTest(tf.test.TestCase):

  def test_get_leaf_nested_dict(self):
    d = {'a': {'i': {'x': 5}}}
    self.assertEqual(train_utils.get_leaf_nested_dict(d, ['a', 'i', 'x']), 5)

  def test_get_leaf_nested_dict_not_leaf(self):
    with self.assertRaisesRegex(KeyError, 'The value extracted with keys.*'):
      d = {'a': {'i': {'x': 5}}}
      train_utils.get_leaf_nested_dict(d, ['a', 'i'])

  def test_get_leaf_nested_dict_path_not_exist_missing_key(self):
    with self.assertRaisesRegex(KeyError, 'Path not exist while traversing .*'):
      d = {'a': {'i': {'x': 5}}}
      train_utils.get_leaf_nested_dict(d, ['a', 'i', 'y'])

  def test_get_leaf_nested_dict_path_not_exist_out_of_range(self):
    with self.assertRaisesRegex(KeyError, 'Path not exist while traversing .*'):
      d = {'a': {'i': {'x': 5}}}
      train_utils.get_leaf_nested_dict(d, ['a', 'i', 'z'])

  def test_get_leaf_nested_dict_path_not_exist_meets_leaf(self):
    with self.assertRaisesRegex(KeyError, 'Path not exist while traversing .*'):
      d = {'a': {'i': 5}}
      train_utils.get_leaf_nested_dict(d, ['a', 'i', 'z'])

  def test_cast_leaf_nested_dict(self):
    d = {'a': {'i': {'x': '123'}}, 'b': 456.5}
    d = train_utils.cast_leaf_nested_dict(d, int)
    self.assertEqual(d['a']['i']['x'], 123)
    self.assertEqual(d['b'], 456)

  def test_write_model_params_keras_model(self):
    inputs = np.zeros([2, 3])
    model = test_utils.FakeKerasModel()
    model(inputs)  # Must do forward pass to build the model.

    filepath = os.path.join(self.create_tempdir(), 'model_params.txt')
    train_utils.write_model_params(model, filepath)
    actual = tf.io.gfile.GFile(filepath, 'r').read().splitlines()

    expected = [
        'fake_keras_model/dense/kernel:0 [3, 4]',
        'fake_keras_model/dense/bias:0 [4]',
        'fake_keras_model/dense_1/kernel:0 [4, 4]',
        'fake_keras_model/dense_1/bias:0 [4]',
        '',
        'Total params: 36',
    ]
    self.assertEqual(actual, expected)

  def test_write_model_params_module(self):
    inputs = np.zeros([2, 3], dtype=np.float32)
    model = test_utils.FakeModule(3, name='fake_module')
    model(inputs)  # Must do forward pass to build the model.

    filepath = os.path.join(self.create_tempdir(), 'model_params.txt')
    train_utils.write_model_params(model, filepath)
    actual = tf.io.gfile.GFile(filepath, 'r').read().splitlines()

    expected = [
        'fake_module/dense/b:0 [4]',
        'fake_module/dense/w:0 [3, 4]',
        'fake_module/dense_1/b:0 [4]',
        'fake_module/dense_1/w:0 [4, 4]',
        '',
        'Total params: 36',
    ]
    self.assertEqual(actual, expected)

  def test_construct_experiment_from_flags(self):
    options = train_utils.ParseConfigOptions(
        experiment='foo',
        config_file=[],
        tpu='bar',
        tf_data_service='',
        params_override='task.model.model_id=new,'
                        'trainer.train_steps=10,'
                        'trainer.validation_steps=11')
    builder = train_utils.ExperimentParser(options)
    params_from_obj = builder.parse()
    params_from_func = train_utils.parse_configuration(options)
    pp = pprint.PrettyPrinter()
    self.assertEqual(
        pp.pformat(params_from_obj.as_dict()),
        pp.pformat(params_from_func.as_dict()))
    self.assertEqual(params_from_obj.runtime.tpu, 'bar')
    self.assertEqual(params_from_obj.task.model.model_id, 'new')
    self.assertEqual(params_from_obj.trainer.train_steps, 10)
    self.assertEqual(params_from_obj.trainer.validation_steps, 11)


class BestCheckpointExporterTest(tf.test.TestCase):

  def test_maybe_export(self):
    model_dir = self.create_tempdir().full_path
    best_ckpt_path = os.path.join(model_dir, 'best_ckpt-1')
    metric_name = 'test_metric|metric_1'
    exporter = train_utils.BestCheckpointExporter(
        model_dir, metric_name, 'higher')
    v = tf.Variable(1.0)
    checkpoint = tf.train.Checkpoint(v=v)
    ret = exporter.maybe_export_checkpoint(
        checkpoint, {'test_metric': {'metric_1': 5.0}}, 100)
    with self.subTest(name='Successful first save.'):
      self.assertEqual(ret, True)
      v_2 = tf.Variable(2.0)
      checkpoint_2 = tf.train.Checkpoint(v=v_2)
      checkpoint_2.restore(best_ckpt_path)
      self.assertEqual(v_2.numpy(), 1.0)

    v = tf.Variable(3.0)
    checkpoint = tf.train.Checkpoint(v=v)
    ret = exporter.maybe_export_checkpoint(
        checkpoint, {'test_metric': {'metric_1': 6.0}}, 200)
    with self.subTest(name='Successful better metic save.'):
      self.assertEqual(ret, True)
      v_2 = tf.Variable(2.0)
      checkpoint_2 = tf.train.Checkpoint(v=v_2)
      checkpoint_2.restore(best_ckpt_path)
      self.assertEqual(v_2.numpy(), 3.0)

    v = tf.Variable(5.0)
    checkpoint = tf.train.Checkpoint(v=v)
    ret = exporter.maybe_export_checkpoint(
        checkpoint, {'test_metric': {'metric_1': 1.0}}, 300)
    with self.subTest(name='Worse metic no save.'):
      self.assertEqual(ret, False)
      v_2 = tf.Variable(2.0)
      checkpoint_2 = tf.train.Checkpoint(v=v_2)
      checkpoint_2.restore(best_ckpt_path)
      self.assertEqual(v_2.numpy(), 3.0)

  def test_export_best_eval_metric(self):
    model_dir = self.create_tempdir().full_path
    metric_name = 'test_metric|metric_1'
    exporter = train_utils.BestCheckpointExporter(model_dir, metric_name,
                                                  'higher')
    exporter.export_best_eval_metric({'test_metric': {'metric_1': 5.0}}, 100)
    with tf.io.gfile.GFile(os.path.join(model_dir, 'info.json'),
                           'rb') as reader:
      metric = json.loads(reader.read())
      self.assertAllEqual(
          metric,
          {'test_metric': {'metric_1': 5.0}, 'best_ckpt_global_step': 100.0})


if __name__ == '__main__':
  tf.test.main()
