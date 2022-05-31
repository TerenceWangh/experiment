# pylint: disable=unused-import
from absl.testing import parameterized
import tensorflow as tf

from experiment.core import config_definitions as cfg
from experiment.core import exp_factory
from experiment.configuration import image_classification as exp_cfg


class ImageClassificationConfigTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      ('resnet_imagenet',),
      ('resnet_rs_imagenet',),
      ('revnet_imagenet',),
      ('mobilenet_imagenet'),
  )
  def test_image_classification_configs(self, config_name):
    config = exp_factory.get_exp_config(config_name)
    self.assertIsInstance(config, cfg.ExperimentConfig)
    self.assertIsInstance(config.task, exp_cfg.ImageClassificationTask)
    self.assertIsInstance(config.task.model,
                          exp_cfg.ImageClassificationModel)
    self.assertIsInstance(config.task.train_data, exp_cfg.DataConfig)
    config.validate()
    config.task.train_data.is_training = None
    with self.assertRaises(KeyError):
      config.validate()


if __name__ == '__main__':
  tf.test.main()
