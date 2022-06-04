import tensorflow as tf
from experiment.model.privacy import configs


class ConfigsTest(tf.test.TestCase):

  def test_clipping_norm_default(self):
    clipping_norm = configs.DifferentialPrivacyConfig().clipping_norm
    self.assertEqual(1000000000.0, clipping_norm)

  def test_noise_multiplier_default(self):
    noise_multiplier = configs.DifferentialPrivacyConfig().noise_multiplier
    self.assertEqual(0.0, noise_multiplier)

  def test_config(self):
    dp_config = configs.DifferentialPrivacyConfig(
        clipping_norm=1.0,
        noise_multiplier=1.0,
    )
    self.assertEqual(1.0, dp_config.clipping_norm)
    self.assertEqual(1.0, dp_config.noise_multiplier)


if __name__ == '__main__':
  tf.test.main()
