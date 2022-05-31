from typing import Optional
import tensorflow as tf

from experiment.configuration import image_classification as classification_cfg

# models
from experiment.model import backbones
from experiment.model import classification_model

def build_classification_model(
    input_specs: tf.keras.layers.InputSpec,
    model_config: classification_cfg.ImageClassificationModel,
    l2_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
    skip_logits_layer: bool = False,
    backbone: Optional[tf.keras.Model] = None) -> tf.keras.Model:
  """Builds the classification model."""
  norm_activation_config = model_config.norm_activation
  if not backbone:
    backbone = backbones.build_backbone(
        input_specs=input_specs,
        backbone_config=model_config.backbone,
        norm_activation_config=norm_activation_config,
        l2_regularizer=l2_regularizer)

  model = classification_model.ClassificationModel(
      backbone=backbone,
      num_classes=model_config.num_classes,
      input_specs=input_specs,
      dropout_rate=model_config.dropout_rate,
      kernel_initializer=model_config.kernel_initializer,
      kernel_regularizer=l2_regularizer,
      add_head_batch_norm=model_config.add_head_batch_norm,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      skip_logits_layer=skip_logits_layer)
  return model
