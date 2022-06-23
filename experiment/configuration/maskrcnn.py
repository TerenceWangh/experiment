"""R-CNN(-RS) configuration definition."""

import dataclasses
import os
from typing import List, Optional, Union

from experiment.hyperparams import Config
from experiment.core.config_definitions import DataConfig as BaseDataConfig
from experiment.core.config_definitions import TaskConfig
from experiment.core.config_definitions import RuntimeConfig
from experiment.core.config_definitions import TrainerConfig
from experiment.core.config_definitions import ExperimentConfig

from experiment.core import exp_factory
from experiment.configuration.common import DataDecoder
from experiment.configuration.common import NormActivation
from experiment.configuration.backbones import Backbone, ResNet
from experiment.configuration.decoders import Decoder, FPN
from experiment.optimization.configuration import OptimizationConfig


# pylint: disable=missing-class-docstring
@dataclasses.dataclass
class Parser(Config):
  num_channels: int = 3
  match_threshold: float = 0.5
  unmatched_threshold: float = 0.5
  aug_rand_hflip: bool = False
  aug_scale_min: float = 1.0
  aug_scale_max: float = 1.0
  skip_crowd_during_training: bool = True
  max_num_instances: int = 100
  rpn_match_threshold: float = 0.7
  rpn_unmatched_threshold: float = 0.3
  rpn_batch_size_per_im: int = 256
  rpn_fg_fraction: float = 0.5
  mask_crop_size: int = 112


@dataclasses.dataclass
class DataConfig(BaseDataConfig):
  """Input config for training."""
  input_path: str = ''
  global_batch_size: int = 0
  is_training: bool = False
  dtype: str = 'bfloat16'
  decoder: DataDecoder = DataDecoder()
  parser: Parser = Parser()
  shuffle_buffer_size: int = 10000
  file_type: str = 'tfrecord'
  drop_remainder: bool = True
  # Number of examples in the data set, it's used to create the annotation file.
  num_examples: int = -1


@dataclasses.dataclass
class Anchor(Config):
  num_scales: int = 1
  aspect_ratios: List[float] = dataclasses.field(
      default_factory=lambda: [0.5, 1.0, 2.0])
  anchor_size: float = 8.0


@dataclasses.dataclass
class RPNHead(Config):
  num_convs: int = 1
  num_filters: int = 256
  use_separable_conv: bool = False


@dataclasses.dataclass
class DetectionHead(Config):
  num_convs: int = 4
  num_filters: int = 256
  use_separable_conv: bool = False
  num_fcs: int = 1
  fc_dims: int = 1024
  class_agnostic_bbox_pred: bool = False
  # If additional IoUs are passed in 'cascade_iou_thresholds'
  # then ensemble the class probabilities from all heads.
  cascade_class_ensemble: bool = False


@dataclasses.dataclass
class ROIGenerator(Config):
  pre_nms_top_k: int = 2000
  pre_nms_score_threshold: float = 0.0
  pre_nms_min_size_threshold: float = 0.0
  nms_iou_threshold: float = 0.7
  num_proposals: int = 1000
  test_pre_nms_top_k: int = 1000
  test_pre_nms_score_threshold: float = 0.0
  test_pre_nms_min_size_threshold: float = 0.0
  test_nms_iou_threshold: float = 0.7
  test_num_proposals: int = 1000
  use_batched_nms: bool = False


@dataclasses.dataclass
class ROISampler(Config):
  mix_gt_boxes: bool = True
  num_sampled_rois: int = 512
  foreground_fraction: float = 0.25
  foreground_iou_threshold: float = 0.5
  background_iou_high_threshold: float = 0.5
  background_iou_low_threshold: float = 0.0
  # IoU thresholds for additional FRCNN heads in Cascade mode.
  # `foreground_iou_threshold` is the first threshold.
  cascade_iou_thresholds: Optional[List[float]] = None


@dataclasses.dataclass
class ROIAligner(Config):
  crop_size: int = 7
  sample_offset: float = 0.5


@dataclasses.dataclass
class DetectionGenerator(Config):
  apply_nms: bool = True
  pre_nms_top_k: int = 5000
  pre_nms_score_threshold: float = 0.05
  nms_iou_threshold: float = 0.5
  max_num_detections: int = 100
  nms_version: str = 'v2'  # `v2`, `v1`, `batched`
  use_cpu_nms: bool = False
  soft_nms_sigma: Optional[float] = None  # Only works when nms_version='v1'.


@dataclasses.dataclass
class MaskHead(Config):
  upsample_factor: int = 2
  num_convs: int = 4
  num_filters: int = 256
  use_separable_conv: bool = False
  class_agnostic: bool = False


@dataclasses.dataclass
class MaskSampler(Config):
  num_sampled_masks: int = 128


@dataclasses.dataclass
class MaskROIAligner(Config):
  crop_size: int = 128
  sample_offset: float = 0.5


@dataclasses.dataclass
class MaskRCNN(Config):
  num_classes: int = 0
  input_size: List[int] = dataclasses.field(default_factory=list)
  min_level: int = 2
  max_level: int = 6
  anchor: Anchor = Anchor()
  include_mask: bool = True
  backbone: Backbone = Backbone(type='resnet', resnet=ResNet())
  decoder: Decoder = Decoder(type='fpn', fpn=FPN())
  rpn_head: RPNHead = RPNHead()
  detection_head: DetectionHead = DetectionHead()
  roi_generator: ROIGenerator = ROIGenerator()
  roi_sampler: ROISampler = ROISampler()
  roi_aligner: ROIAligner = ROIAligner()
  detection_generator: DetectionGenerator = DetectionGenerator()
  mask_head: Optional[MaskHead] = MaskHead()
  mask_sampler: Optional[MaskSampler] = MaskSampler()
  mask_roi_aligner: Optional[MaskROIAligner] = MaskROIAligner()
  norm_activation: NormActivation = NormActivation(
      norm_momentum=0.997,
      norm_epsilon=0.0001,
      use_sync_bn=True)


@dataclasses.dataclass
class Losses(Config):
  loss_weight: float = 1.0
  rpn_huber_loss_delta: float = 1. / 9.
  frcnn_huber_loss_delta: float = 1.
  l2_weight_decay: float = 0.0
  rpn_score_weight: float = 1.0
  rpn_box_weight: float = 1.0
  frcnn_class_weight: float = 1.0
  frcnn_box_weight: float = 1.0
  mask_weight: float = 1.0


@dataclasses.dataclass
class MaskRCNNTask(TaskConfig):
  model: MaskRCNN = MaskRCNN()
  train_data: DataConfig = DataConfig(is_training=True)
  validation_data: DataConfig = DataConfig(is_training=False,
                                           drop_remainder=False)
  losses: Losses = Losses()
  init_checkpoint: Optional[str] = None
  init_checkpoint_modules: Union[
    str, List[str]] = 'all'  # all, backbone, and/or decoder
  annotation_file: Optional[str] = None
  per_category_metrics: bool = False
  # If set, we only use masks for the specified class IDs.
  allowed_mask_class_ids: Optional[List[int]] = None
  # If set, the COCO metrics will be computed.
  use_coco_metrics: bool = True
  # If set, the Waymo Open Dataset evaluator would be used.
  use_wod_metrics: bool = False

  # If set, freezes the backbone during training.
  freeze_backbone: bool = False


COCO_INPUT_PATH_BASE = 'coco'


@exp_factory.register_config_factory('fasterrcnn_resnetfpn_coco')
def fasterrcnn_resnetfpn_coco() -> ExperimentConfig:
  """COCO object detection with Faster R-CNN."""
  steps_per_epoch = 500
  coco_val_samples = 5000
  train_batch_size = 64
  eval_batch_size = 8

  config = ExperimentConfig(
      runtime=RuntimeConfig(mixed_precision_dtype='bfloat16'),
      task=MaskRCNNTask(
          # pylint: disable=line-too-long
          init_checkpoint='gs://cloud-tpu-checkpoints/vision-2.0/resnet50_imagenet/ckpt-28080',
          init_checkpoint_modules='backbone',
          annotation_file=os.path.join(COCO_INPUT_PATH_BASE, 'instances_val2017.json'),
          # pylint: enable=line-too-long
          model=MaskRCNN(
              num_classes=91,
              input_size=[1024, 1024, 3],
              include_mask=False,
              mask_head=None,
              mask_sampler=None,
              mask_roi_aligner=None),
          losses=Losses(l2_weight_decay=0.00004),
          train_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'train*'),
              is_training=True,
              global_batch_size=train_batch_size,
              parser=Parser(
                  aug_rand_hflip=True,
                  aug_scale_min=0.8,
                  aug_scale_max=1.25,
              ),
          ),
          validation_data=DataConfig(
              input_path=os.path.join(COCO_INPUT_PATH_BASE, 'val*'),
              is_training=False,
              global_batch_size=eval_batch_size,
              drop_remainder=False,
          ),
      ),
      trainer=TrainerConfig(
          train_steps=22500,
          validation_steps=coco_val_samples // eval_batch_size,
          steps_per_loop=steps_per_epoch,
          summary_interval=steps_per_epoch,
          checkpoint_interval=steps_per_epoch,
          optimizer_config=OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9,
                  },
              },
              'learning_rate': {
                  'type': 'stepwise',
                  'stepwise': {
                      'boundaries': [15000, 20000],
                      'values': [0.12, 0.012, 0.0012],
                  },
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 500,
                      'warmup_learning_rate': 0.0067,
                  },
              },
          }),
          restrictions=[
              'task.train_data.is_training != None',
              'task.validation_data.is_training != None',
          ]
      )
  )
  return config
