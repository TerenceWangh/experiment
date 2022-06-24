from typing import Optional
import tensorflow as tf

from experiment.configuration import image_classification as classification_cfg
from experiment.configuration import maskrcnn as maskrcnn_cfg

# models
from experiment.model import backbones
from experiment.model import decoders
from experiment.model import heads
from experiment.model.layers import mask_sampler
from experiment.model.layers import roi_aligner
from experiment.model.layers import roi_generator
from experiment.model.layers import roi_sampler

from experiment.model import classification_model
from experiment.model.segmentations import mrcnn


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


def build_maskrcnn(
    input_specs: tf.keras.layers.InputSpec,
    model_config: maskrcnn_cfg.MaskRCNN,
    l2_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
    backbone: Optional[tf.keras.Model] = None,
    decoder: Optional[tf.keras.Model] = None) -> tf.keras.Model:
  """Builds Mask R-CNN model."""
  norm_activation_config = model_config.norm_activation
  if not backbone:
    backbone = backbones.factory.build_backbone(
        input_specs=input_specs,
        backbone_config=model_config.backbone,
        norm_activation_config=norm_activation_config,
        l2_regularizer=l2_regularizer)
  backbone_features = backbone(tf.keras.Input(input_specs.shape[1:]))

  if not decoder:
    decoder = decoders.factory.build_decoder(
        input_specs=backbone.output_specs,
        model_config=model_config,
        l2_regularizer=l2_regularizer)

  rpn_head_config = model_config.rpn_head
  roi_generator_config = model_config.roi_generator
  roi_sampler_config = model_config.roi_sampler
  roi_aligner_config = model_config.roi_aligner
  detection_head_config = model_config.detection_head
  generator_config = model_config.detection_generator
  num_anchors_per_location = (
      len(model_config.anchor.aspect_ratios) * model_config.anchor.num_scales)

  rpn_head = heads.dense_prediction_heads.RPNHead(
      min_level=model_config.min_level,
      max_level=model_config.max_level,
      num_anchors_per_location=num_anchors_per_location,
      num_convs=rpn_head_config.num_convs,
      num_filters=rpn_head_config.num_filters,
      use_separable_conv=rpn_head_config.use_separable_conv,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)

  detection_head = heads.instance_heads.DetectionHead(
      num_classes=model_config.num_classes,
      num_convs=detection_head_config.num_convs,
      num_filters=detection_head_config.num_filters,
      use_separable_conv=detection_head_config.use_separable_conv,
      num_fcs=detection_head_config.num_fcs,
      fc_dims=detection_head_config.fc_dims,
      class_agnostic_bbox_pred=detection_head_config.class_agnostic_bbox_pred,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer,
      name='detection_head')

  if decoder:
    decoder_features = decoder(backbone_features)
    rpn_head(decoder_features)

  if roi_sampler_config.cascade_iou_thresholds:
    detection_head_cascade = [detection_head]
    for cascade_num in range(len(roi_sampler_config.cascade_iou_thresholds)):
      detection_head = heads.instance_heads.DetectionHead(
          num_classes=model_config.num_classes,
          num_convs=detection_head_config.num_convs,
          num_filters=detection_head_config.num_filters,
          use_separable_conv=detection_head_config.use_separable_conv,
          num_fcs=detection_head_config.num_fcs,
          fc_dims=detection_head_config.fc_dims,
          class_agnostic_bbox_pred=detection_head_config
            .class_agnostic_bbox_pred,
          activation=norm_activation_config.activation,
          use_sync_bn=norm_activation_config.use_sync_bn,
          norm_momentum=norm_activation_config.norm_momentum,
          norm_epsilon=norm_activation_config.norm_epsilon,
          kernel_regularizer=l2_regularizer,
          name='detection_head_{}'.format(cascade_num + 1))

      detection_head_cascade.append(detection_head)
    detection_head = detection_head_cascade

  roi_generator_obj = roi_generator.MultilevelROIGenerator(
      pre_nms_top_k=roi_generator_config.pre_nms_top_k,
      pre_nms_score_threshold=roi_generator_config.pre_nms_score_threshold,
      pre_nms_min_size_threshold=(
          roi_generator_config.pre_nms_min_size_threshold),
      nms_iou_threshold=roi_generator_config.nms_iou_threshold,
      num_proposals=roi_generator_config.num_proposals,
      test_pre_nms_top_k=roi_generator_config.test_pre_nms_top_k,
      test_pre_nms_score_threshold=(
          roi_generator_config.test_pre_nms_score_threshold),
      test_pre_nms_min_size_threshold=(
          roi_generator_config.test_pre_nms_min_size_threshold),
      test_nms_iou_threshold=roi_generator_config.test_nms_iou_threshold,
      test_num_proposals=roi_generator_config.test_num_proposals,
      use_batched_nms=roi_generator_config.use_batched_nms)

  roi_sampler_cascade = []
  roi_sampler_obj = roi_sampler.ROISampler(
      mix_gt_boxes=roi_sampler_config.mix_gt_boxes,
      num_sampled_rois=roi_sampler_config.num_sampled_rois,
      foreground_fraction=roi_sampler_config.foreground_fraction,
      foreground_iou_threshold=roi_sampler_config.foreground_iou_threshold,
      background_iou_high_threshold=(
          roi_sampler_config.background_iou_high_threshold),
      background_iou_low_threshold=(
          roi_sampler_config.background_iou_low_threshold))
  roi_sampler_cascade.append(roi_sampler_obj)
  # Initialize additional roi simplers for cascade heads.
  if roi_sampler_config.cascade_iou_thresholds:
    for iou in roi_sampler_config.cascade_iou_thresholds:
      roi_sampler_obj = roi_sampler.ROISampler(
          mix_gt_boxes=False,
          num_sampled_rois=roi_sampler_config.num_sampled_rois,
          foreground_iou_threshold=iou,
          background_iou_high_threshold=iou,
          background_iou_low_threshold=0.0,
          skip_subsampling=True)
      roi_sampler_cascade.append(roi_sampler_obj)

  roi_aligner_obj = roi_aligner.MultiLevelRoIAligner(
      crop_size=roi_aligner_config.crop_size,
      sample_offset=roi_aligner_config.sample_offset)

  detection_generator_obj = heads.detection_generator.DetectionGenerator(
      apply_nms=generator_config.apply_nms,
      pre_nms_top_k=generator_config.pre_nms_top_k,
      pre_nms_score_threshold=generator_config.pre_nms_score_threshold,
      nms_iou_threshold=generator_config.nms_iou_threshold,
      max_num_detections=generator_config.max_num_detections,
      nms_version=generator_config.nms_version,
      use_cpu_nms=generator_config.use_cpu_nms,
      soft_nms_sigma=generator_config.soft_nms_sigma)

  if model_config.include_mask:
    mask_head = heads.instance_heads.MaskHead(
        num_classes=model_config.num_classes,
        upsample_factor=model_config.mask_head.upsample_factor,
        num_convs=model_config.mask_head.num_convs,
        num_filters=model_config.mask_head.num_filters,
        use_separable_conv=model_config.mask_head.use_separable_conv,
        activation=model_config.norm_activation.activation,
        norm_momentum=model_config.norm_activation.norm_momentum,
        norm_epsilon=model_config.norm_activation.norm_epsilon,
        kernel_regularizer=l2_regularizer,
        class_agnostic=model_config.mask_head.class_agnostic)

    mask_sampler_obj = mask_sampler.MaskSampler(
        mask_target_size=(
            model_config.mask_roi_aligner.crop_size *
            model_config.mask_head.upsample_factor),
        num_sampled_masks=model_config.mask_sampler.num_sampled_masks)

    mask_roi_aligner_obj = roi_aligner.MultiLevelRoIAligner(
        crop_size=model_config.mask_roi_aligner.crop_size,
        sample_offset=model_config.mask_roi_aligner.sample_offset)
  else:
    mask_head = None
    mask_sampler_obj = None
    mask_roi_aligner_obj = None

  model = mrcnn.MaskRCNNModel(
      backbone=backbone,
      decoder=decoder,
      rpn_head=rpn_head,
      detection_head=detection_head,
      roi_generator=roi_generator_obj,
      roi_sampler=roi_sampler_cascade,
      roi_aligner=roi_aligner_obj,
      detection_generator=detection_generator_obj,
      mask_head=mask_head,
      mask_sampler=mask_sampler_obj,
      mask_roi_aligner=mask_roi_aligner_obj,
      class_agnostic_bbox_pred=detection_head_config.class_agnostic_bbox_pred,
      cascade_class_ensemble=detection_head_config.cascade_class_ensemble,
      min_level=model_config.min_level,
      max_level=model_config.max_level,
      num_scales=model_config.anchor.num_scales,
      aspect_ratios=model_config.anchor.aspect_ratios,
      anchor_size=model_config.anchor.anchor_size)
  return model
