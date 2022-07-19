"""R-CNN(-RS) models."""

from typing import Any, List, Mapping, Optional, Tuple, Union

import tensorflow as tf

from experiment.ops import anchor
from experiment.ops import box_ops

layers = tf.keras.layers

MASK_RCNN_OUTPUT = Tuple[Mapping[str, tf.Tensor], Mapping[str, tf.Tensor]]


@tf.keras.utils.register_keras_serializable(package='experiment')
class MaskRCNNModel(tf.keras.Model):
  """The Mask R-CNN(-RS) and Cascade RCNN-RS models."""

  def __init__(self,
               backbone: tf.keras.Model,
               decoder: tf.keras.Model,
               rpn_head: layers.Layer,
               detection_head: Union[layers.Layer, List[layers.Layer]],
               roi_generator: layers.Layer,
               roi_sampler: Union[layers.Layer, List[layers.Layer]],
               roi_aligner: layers.Layer,
               detection_generator: layers.Layer,
               mask_head: Optional[layers.Layer] = None,
               mask_sampler: Optional[layers.Layer] = None,
               mask_roi_aligner: Optional[layers.Layer] = None,
               class_agnostic_bbox_pred: bool = False,
               cascade_class_ensemble: bool = False,
               min_level: Optional[int] = None,
               max_level: Optional[int] = None,
               num_scales: Optional[int] = None,
               aspect_ratios: Optional[List[float]] = None,
               anchor_size: Optional[List[float]] = None,
               **kwargs):
    """Initializes the R-CNN(-RS) model.

    Arguments
    =========
    backbone : tf.keras.Model
        The backbone network.
    decoder : tf.keras.Model
        The decoder network.
    rpn_head : layers.Layer
        The RPN head.
    detection_head : Union[layers.Layer, List[layers.Layer]]
        The detection head or a list of heads.
    roi_generator : layers.Layer
        The ROI generator.
    roi_sampler : layers.Layer
        A single ROI sampler or a list of ROI samplers for cascade detection
        heads.
    roi_aligner : layers.Layer
        The ROI aligner.
    detection_generator: layers.Layer
        The detection generator.
    mask_head : Optional[layers.Layer],
        The mask head.
    mask_sampler : Optional[layers.Layer],
        The mask sampler.
    mask_roi_aligner : Optional[layers.Layer]
        The ROI aligner for mask prediction.
    class_agnostic_bbox_pred : bool, default False
        If True, perform class agnostic bounding box prediction. Needs to be
        `True` for Cascade RCNN models.
    cascade_class_ensemble : bool, default False
        If True, ensemble classification scores over all detection heads.
    min_level : Optional[int]
        Minimum level in output feature maps.
    max_level : Optional[int]
        Maximum level in output feature maps.
    num_scales : Optional[int]
        A number representing intermediate scales added on each level.
        For instances, num_scales=2 adds one additional intermediate anchor
        scales [2^0, 2^0.5] on each level.
    aspect_ratios : Optional[List[float]]
        A list representing the aspect ratio anchors added on each level.
        The number indicates the ratio of width to height. For instances,
        aspect_ratios=[1.0, 2.0, 0.5] adds three anchors on each scale level.
    anchor_size : Optional[List[float]]
        A number representing the scale of size of the base anchor to the
        feature stride 2^level.
    """
    super(MaskRCNNModel, self).__init__(**kwargs)
    self._config_dict = {
        'backbone': backbone,
        'decoder': decoder,
        'rpn_head': rpn_head,
        'detection_head': detection_head,
        'roi_generator': roi_generator,
        'roi_sampler': roi_sampler,
        'roi_aligner': roi_aligner,
        'detection_generator': detection_generator,
        'mask_head': mask_head,
        'mask_sampler': mask_sampler,
        'mask_roi_aligner': mask_roi_aligner,
        'class_agnostic_bbox_pred': class_agnostic_bbox_pred,
        'cascade_class_ensemble': cascade_class_ensemble,
        'min_level': min_level,
        'max_level': max_level,
        'num_scales': num_scales,
        'aspect_ratios': aspect_ratios,
        'anchor_size': anchor_size,
    }
    self._backbone = backbone
    self._decoder = decoder
    self._rpn_head = rpn_head
    if not isinstance(detection_head, (list, tuple)):
      self._detection_head = [detection_head]
    else:
      self._detection_head = detection_head
    self._roi_generator = roi_generator
    if not isinstance(roi_sampler, (list, tuple)):
      self._roi_sampler = [roi_sampler]
    else:
      self._roi_sampler = roi_sampler
    if len(self._roi_sampler) > 1 and not class_agnostic_bbox_pred:
      raise ValueError('`class_agnostic_bbox_pred` needs to be True if'
                       'multiple detection heads are specified.')
    self._roi_aligner = roi_aligner
    self._detection_generator = detection_generator
    self._include_mask = mask_head is not None
    self._mask_head = mask_head
    if self._include_mask and mask_sampler is None:
      raise ValueError('`mask_sampler` is not provided in Mask R-CNN.')
    self._mask_sampler = mask_sampler
    if self._include_mask and mask_roi_aligner is None:
      raise ValueError('`mask_roi_aligner` is not provided in Mask R-CNN.')
    self._mask_roi_aligner = mask_roi_aligner
    # Weights for the regression losses for each FRCNN layer.
    self._cascade_layer_to_weights = [
        [10.0, 10.0, 5.0, 5.0],
        [20.0, 20.0, 10.0, 10.0],
        [30.0, 30.0, 15.0, 15.0],
    ]

  def call(self,
           images: tf.Tensor,
           image_shape: tf.Tensor,
           anchor_boxes: Optional[Mapping[str, tf.Tensor]] = None,
           gt_boxes: Optional[tf.Tensor] = None,
           gt_classes: Optional[tf.Tensor] = None,
           gt_masks: Optional[tf.Tensor] = None,
           training: Optional[bool] = None) -> Mapping[str, tf.Tensor]:
    model_outputs, intermediate_outputs = self._call_box_outputs(
        images=images,
        image_shape=image_shape,
        anchor_boxes=anchor_boxes,
        gt_boxes=gt_boxes,
        gt_classes=gt_classes,
        training=training)
    if not self._include_mask:
      return model_outputs

    model_mask_outputs = self._call_mask_outputs(
        model_box_outputs=model_outputs,
        features=model_outputs['decoder_features'],
        current_rois=intermediate_outputs['current_rois'],
        matched_gt_indices=intermediate_outputs['matched_gt_indices'],
        matched_gt_boxes=intermediate_outputs['matched_gt_boxes'],
        matched_gt_classes=intermediate_outputs['matched_gt_classes'],
        gt_masks=gt_masks,
        training=training)
    model_outputs.update(model_mask_outputs)
    return model_outputs

  def _get_backbone_and_decoder_features(self, images):
    backbone_features = self._backbone(images)
    if self._decoder:
      features = self._decoder(backbone_features)
    else:
      features = backbone_features
    return backbone_features, features

  def _call_box_outputs(self,
                        images: tf.Tensor,
                        image_shape: tf.Tensor,
                        anchor_boxes: Optional[Mapping[str, tf.Tensor]] = None,
                        gt_boxes: Optional[tf.Tensor] = None,
                        gt_classes: Optional[tf.Tensor] = None,
                        training: Optional[bool] = None) -> MASK_RCNN_OUTPUT:
    """Implementation of the Faster-RCNN logic for boxes."""
    model_outputs = {}

    # Feature extraction.
    backbone_features, decoder_features = \
        self._get_backbone_and_decoder_features(images)

    # Region proposal network.
    rpn_scores, rpn_boxes = self._rpn_head(decoder_features)

    model_outputs.update({
        'backbone_features': backbone_features,
        'decoder_features': decoder_features,
        'rpn_boxes': rpn_boxes,
        'rpn_scores': rpn_scores,
    })

    # Generate anchor boxes for this batch if not provided.
    if anchor_boxes is None:
      _, image_height, image_width, _ = images.get_shape().as_list()
      anchor_boxes = anchor.Anchor(
          min_level=self._config_dict['min_level'],
          max_level=self._config_dict['max_level'],
          num_scales=self._config_dict['num_scales'],
          aspect_ratios=self._config_dict['aspect_ratios'],
          anchor_size=self._config_dict['anchor_size'],
          image_size=(image_height, image_width)
      ).multi_level_boxes

      for l in anchor_boxes:
        anchor_boxes[l] = tf.tile(
            tf.expand_dims(anchor_boxes[l], axis=0),
            [tf.shape(images)[0], 1, 1, 1])

    # Generate RoIs.
    current_rois, _ = self._roi_generator(
        rpn_boxes, rpn_scores, anchor_boxes, image_shape, training)

    next_rois = current_rois
    all_class_outputs = []
    for cascade_num in range(len(self._roi_sampler)):
      # In cascade RCNN we want the higher layers to have different regression
      # weights as the predicted deltas become smaller and smaller.
      regression_weights = self._cascade_layer_to_weights[cascade_num]
      current_rois = next_rois

      values = self._run_frcnn_head(
          features=decoder_features,
          rois=current_rois,
          gt_boxes=gt_boxes,
          gt_classes=gt_classes,
          training=training,
          model_outputs=model_outputs,
          cascade_num=cascade_num,
          regression_weights=regression_weights)

      class_outputs = values[0]
      box_outputs = values[1]
      model_outputs = values[2]
      matched_gt_boxes = values[3]
      matched_gt_classes = values[4]
      matched_gt_indices = values[5]
      current_rois = values[6]
      all_class_outputs.append(class_outputs)

      # Generate ROIs for the next cascade head if there is any.
      if cascade_num < len(self._roi_sampler) - 1:
        next_rois = box_ops.decode_boxes(
            tf.cast(box_outputs, tf.float32),
            current_rois,
            weights=regression_weights)
        next_rois = box_ops.clip_boxes(
            next_rois, tf.expand_dims(image_shape, axis=1))

    if not training:
      if self._config_dict['cascade_class_ensemble']:
        class_outputs = tf.add_n(all_class_outputs) / len(all_class_outputs)

      detections = self._detection_generator(
          box_outputs,
          class_outputs,
          current_rois,
          image_shape,
          regression_weights,
          bbox_per_class=(not self._config_dict['class_agnostic_bbox_pred']))
      model_outputs.update({
          'cls_outputs': class_outputs,
          'box_outputs': box_outputs,
      })
      if self._detection_generator.get_config()['apply_nms']:
        model_outputs.update({
            'detection_boxes': detections['detection_boxes'],
            'detection_scores': detections['detection_scores'],
            'detection_classes': detections['detection_classes'],
            'num_detections': detections['num_detections']
        })
      else:
        model_outputs.update({
            'decoded_boxes': detections['decoded_boxes'],
            'decoded_box_scores': detections['decoded_box_scores']
        })

    intermediate_outputs = {
        'matched_gt_boxes': matched_gt_boxes,
        'matched_gt_indices': matched_gt_indices,
        'matched_gt_classes': matched_gt_classes,
        'current_rois': current_rois,
    }
    return model_outputs, intermediate_outputs

  def _call_mask_outputs(self,
                         model_box_outputs: Mapping[str, tf.Tensor],
                         features: tf.Tensor,
                         current_rois: tf.Tensor,
                         matched_gt_indices: tf.Tensor,
                         matched_gt_boxes: tf.Tensor,
                         matched_gt_classes: tf.Tensor,
                         gt_masks: tf.Tensor,
                         training: Optional[bool] = None
                         ) -> Mapping[str, tf.Tensor]:
    """Mask-RCNN mask prediction logic."""
    model_outputs = dict(model_box_outputs)
    if training:
      current_rois, roi_classes, roi_masks = self._mask_sampler(
          current_rois,
          matched_gt_boxes,
          matched_gt_classes,
          matched_gt_indices,
          gt_masks)
      roi_masks = tf.stop_gradient(roi_masks)

      model_outputs.update({
          'mask_class_targets': roi_classes,
          'mask_targets': roi_masks,
      })
    else:
      current_rois = model_outputs['detection_boxes']
      roi_classes = model_outputs['detection_classes']

    mask_logits, mask_probs = self._features_to_mask_outputs(
        features, current_rois, roi_classes)

    if training:
      model_outputs.update({
          'mask_outputs': mask_logits,
      })
    else:
      model_outputs.update({
          'detection_masks': mask_probs,
      })
    return model_outputs

  def _run_frcnn_head(self, features, rois, gt_boxes, gt_classes, training,
                      model_outputs, cascade_num, regression_weights):
    matched_gt_boxes = None
    matched_gt_indices = None
    matched_gt_classes = None

    name_format = lambda name, num: '{}_{}'.format(name, num) if num else name

    if training and gt_boxes is not None:
      rois = tf.stop_gradient(rois)
      current_roi_sampler = self._roi_sampler[cascade_num]
      rois, matched_gt_boxes, matched_gt_classes, matched_gt_indices = \
          current_roi_sampler(rois, gt_boxes, gt_classes)
      # Create bounding box training targets
      box_targets = box_ops.encode_boxes(matched_gt_boxes, rois,
                                         weights=regression_weights)
      # If the target is background, the box target is set to all 0s.
      box_targets = tf.where(
          tf.tile(
              tf.expand_dims(tf.equal(matched_gt_classes, 0), axis=-1),
              [1, 1, 4]
          ),
          tf.zeros_like(box_targets), box_targets)

      model_outputs.update({
          name_format('class_targets', cascade_num): matched_gt_classes,
          name_format('box_targets', cascade_num): box_targets,
      })

    # Get roi features.
    roi_features = self._roi_aligner(features, rois)

    # Run frcnn head to get class and bbox predictions.
    current_detection_head = self._detection_head[cascade_num]
    class_outputs, box_outputs = current_detection_head(roi_features)

    model_outputs.update({
        name_format('class_outputs', cascade_num): class_outputs,
        name_format('box_outputs', cascade_num): box_outputs,
    })
    return class_outputs, box_outputs, model_outputs, matched_gt_boxes, \
        matched_gt_classes, matched_gt_indices, rois

  def _features_to_mask_outputs(self, features, rois, roi_classes):
    # Mask RoI align.
    mask_roi_features = self._mask_roi_aligner(features, rois)
    # Mask Head
    raw_masks = self._mask_head([mask_roi_features, roi_classes])
    return raw_masks, tf.nn.sigmoid(raw_masks)

  @property
  def checkpoint_items(self) -> Mapping[
      str, Union[tf.keras.Model, tf.keras.layers.Layer]]:
    items = dict(
        backbone=self._backbone,
        rpn_head=self._rpn_head,
        detection_head=self._detection_head)
    if self._decoder is not None:
      items.update(decoder=self._decoder)
    if self._include_mask:
      items.update(mask_head=self._mask_head)

    return items

  @property
  def backbone(self) -> tf.keras.Model:
    return self._backbone

  @property
  def decoder(self) -> tf.keras.Model:
    return self._decoder

  def get_config(self) -> Mapping[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)
