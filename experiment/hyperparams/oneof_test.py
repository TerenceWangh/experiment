import dataclasses
import tensorflow as tf
from experiment.hyperparams import base_config
from experiment.hyperparams import oneof

@dataclasses.dataclass
class ResNet(base_config.Config):
  model_depth: int = 50


@dataclasses.dataclass
class Backbone(oneof.OneOfConfig):
  type: str = 'resnet'
  resnet: ResNet = ResNet()
  not_resnet: int = 2


@dataclasses.dataclass
class OutputLayer(oneof.OneOfConfig):
  type: str = 'single'
  single: int = 1
  multi_head: int = 2


@dataclasses.dataclass
class Network(base_config.Config):
  backbone: Backbone = Backbone()
  output_layer: OutputLayer = OutputLayer()


class OneOfTest(tf.test.TestCase):

  def test_to_dict(self):
    network_params = {
        'backbone': {
            'type': 'resnet',
            'resnet': {
                'model_depth': 50
            }
        },
        'output_layer': {
            'type': 'single',
            'single': 1000
        }
    }
    network_config = Network(network_params)
    self.assertEqual(network_config.as_dict(), network_params)

  def test_get_oneof(self):
    backbone = Backbone()
    self.assertIsInstance(backbone.get(), ResNet)
    self.assertEqual(backbone.get().as_dict(), {'model_depth': 50})


if __name__ == '__main__':
  tf.test.main()
