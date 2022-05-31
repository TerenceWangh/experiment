import tensorflow as tf

from experiment.optimization.configuration import learning_rate_config as lr_cfg
from experiment.optimization.configuration import optimization_config
from experiment.optimization.configuration import optimizer_config as opt_cfg


class OptimizerConfigTest(tf.test.TestCase):

  def test_no_optimizer(self):
    optimizer = optimization_config.OptimizationConfig({}).optimizer.get()
    self.assertIsNone(optimizer)

  def test_no_lr_schedule(self):
    lr = optimization_config.OptimizationConfig({}).learning_rate.get()
    self.assertIsNone(lr)

  def test_no_warmup_schedule(self):
    warmup = optimization_config.OptimizationConfig({}).warmup.get()
    self.assertIsNone(warmup)

  def test_config(self):
    opt_config = optimization_config.OptimizationConfig({
      'optimizer'    : {
        'type': 'sgd',
        'sgd' : {}  # default config
      },
      'learning_rate': {
        'type'      : 'polynomial',
        'polynomial': {}
      },
      'warmup'       : {
        'type': 'linear'
      }
    })
    self.assertEqual(opt_config.optimizer.get(), opt_cfg.SGDConfig())
    self.assertEqual(opt_config.learning_rate.get(),
                     lr_cfg.PolynomialLrConfig())
    self.assertEqual(opt_config.warmup.get(), lr_cfg.LinearWarmupConfig())


if __name__ == '__main__':
  tf.test.main()
