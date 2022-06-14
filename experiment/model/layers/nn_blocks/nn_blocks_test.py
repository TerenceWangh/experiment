"""Tests for nn_blocks"""

from typing import Any, Iterable, Tuple

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations

from experiment.model.layers import nn_blocks


def distribution_strategy_combinations() -> Iterable[Tuple[Any, ...]]:
  """Return the combinations of end-to-end tests to run."""
  return combinations.combine(
      distribution=[
        strategy_combinations.default_strategy,
        strategy_combinations.cloud_tpu_strategy,
        strategy_combinations.one_device_strategy_gpu,
      ],
  )


class NNBlocksTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (nn_blocks.InvertedBottleneckBlock, 1, 1, None, None),
      (nn_blocks.InvertedBottleneckBlock, 6, 1, None, None),
      (nn_blocks.InvertedBottleneckBlock, 1, 2, None, None),
      (nn_blocks.InvertedBottleneckBlock, 1, 1, 0.2, None),
      (nn_blocks.InvertedBottleneckBlock, 1, 1, None, 0.2),
  )
  def test_inverted_bottleneck_block(self,
                                     block_fn,
                                     expand_ratio,
                                     strides,
                                     se_ratio,
                                     stochastic_depth_drop_rate):
    input_size = 128
    in_filters = 24
    out_filters = 40

    inputs = tf.keras.Input(
        shape=(input_size, input_size, in_filters), batch_size=1)
    block = block_fn(
        in_filters=in_filters,
        out_filters=out_filters,
        expand_ratio=expand_ratio,
        strides=strides,
        se_ratio=se_ratio,
        stochastic_depth_drop_rate=stochastic_depth_drop_rate,
    )

    features = block(inputs)

    self.assertAllEqual(
        [1, input_size // strides, input_size // strides, out_filters],
        features.shape.as_list())


class DepthwiseSeparableConvBlockTest(parameterized.TestCase, tf.test.TestCase):

  @combinations.generate(distribution_strategy_combinations())
  def test_shape(self, distribution):
    batch_size, height, width, num_channels = 8, 32, 32, 32
    num_filters = 64
    strides = 2

    input_tensor = tf.random.normal(
        shape=[batch_size, height, width, num_channels])

    with distribution.scope():
      block = nn_blocks.DepthwiseSeparableConvBlock(
          filters=num_filters, strides=strides)
      config_dict = block.get_config()
      recreate_block = nn_blocks.DepthwiseSeparableConvBlock(**config_dict)

    output_tensor = block(input_tensor)
    expected_output_shape = [
      batch_size, height // strides, width // strides, num_filters]
    self.assertAllEqual(output_tensor.shape.as_list(), expected_output_shape)

    output_tensor = recreate_block(input_tensor)
    self.assertEqual(output_tensor.shape.as_list(), expected_output_shape)


if __name__ == '__main__':
  tf.test.main()
