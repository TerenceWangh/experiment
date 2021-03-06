import os
# Import libraries
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from experiment.dataloaders import TfExampleDecoderLabelMap
from experiment.dataloaders import tf_example_utils

LABEL_MAP_CSV_CONTENT = '0,class_0\n1,class_1\n2,class_2'


class TfExampleDecoderLabelMapTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (100, 100, 0),
      (100, 100, 1),
      (100, 100, 2),
      (100, 100, 0),
      (100, 100, 1),
      (100, 100, 2),
  )
  def test_result_shape(self, image_height, image_width, num_instances):
    label_map_dir = self.get_temp_dir()
    label_map_name = 'label_map.csv'
    label_map_path = os.path.join(label_map_dir, label_map_name)
    with open(label_map_path, 'w') as f:
      f.write(LABEL_MAP_CSV_CONTENT)

    decoder = TfExampleDecoderLabelMap(label_map_path, include_mask=True)

    serialized_example = tf_example_utils.create_detection_test_example(
        image_height=image_height,
        image_width=image_width,
        image_channel=3,
        num_instances=num_instances).SerializeToString()
    decoded_tensors = decoder.decode(
        tf.convert_to_tensor(value=serialized_example))

    results = tf.nest.map_structure(lambda x: x.numpy(), decoded_tensors)

    self.assertAllEqual(
        (image_height, image_width, 3), results['image'].shape)
    self.assertEqual(tf_example_utils.DUMP_SOURCE_ID, results['source_id'])
    self.assertEqual(image_height, results['height'])
    self.assertEqual(image_width, results['width'])
    self.assertAllEqual(
        (num_instances,), results['ground_truth_classes'].shape)
    self.assertAllEqual(
        (num_instances,), results['ground_truth_is_crowd'].shape)
    self.assertAllEqual(
        (num_instances,), results['ground_truth_area'].shape)
    self.assertAllEqual(
        (num_instances, 4), results['ground_truth_boxes'].shape)
    self.assertAllEqual(
        (num_instances, image_height, image_width),
        results['ground_truth_instance_masks'].shape)
    self.assertAllEqual(
        (num_instances,), results['ground_truth_instance_masks_png'].shape)

  def test_result_content(self):
    label_map_dir = self.get_temp_dir()
    label_map_name = 'label_map.csv'
    label_map_path = os.path.join(label_map_dir, label_map_name)
    with open(label_map_path, 'w') as f:
      f.write(LABEL_MAP_CSV_CONTENT)

    decoder = TfExampleDecoderLabelMap(label_map_path, include_mask=True)

    image_content = [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                     [[0, 0, 0], [255, 255, 255], [255, 255, 255], [0, 0, 0]],
                     [[0, 0, 0], [255, 255, 255], [255, 255, 255], [0, 0, 0]],
                     [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]
    image = tf_example_utils.encode_image(np.uint8(image_content), fmt='PNG')
    image_height = 4
    image_width = 4
    num_instances = 2
    xmins = [0, 0.25]
    xmaxs = [0.5, 1.0]
    ymins = [0, 0]
    ymaxs = [0.5, 1.0]
    labels = [b'class_2', b'class_0']
    areas = [
        0.25 * image_height * image_width, 0.75 * image_height * image_width
    ]
    is_crowds = [1, 0]
    mask_content = [[[255, 255, 0, 0],
                     [255, 255, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]],
                    [[0, 255, 255, 255],
                     [0, 255, 255, 255],
                     [0, 255, 255, 255],
                     [0, 255, 255, 255]]]
    masks = [
        tf_example_utils.encode_image(np.uint8(m), fmt='PNG')
        for m in list(mask_content)
    ]
    # pylint: disable=bad-whitespace
    serialized_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/encoded'          : (tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[image]))),
                'image/source_id'        : (tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[tf_example_utils.DUMP_SOURCE_ID]))),
                'image/height'           : (tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[image_height]))),
                'image/width'            : (tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[image_width]))),
                'image/object/bbox/xmin' : (tf.train.Feature(
                    float_list=tf.train.FloatList(value=xmins))),
                'image/object/bbox/xmax' : (tf.train.Feature(
                    float_list=tf.train.FloatList(value=xmaxs))),
                'image/object/bbox/ymin' : (tf.train.Feature(
                    float_list=tf.train.FloatList(value=ymins))),
                'image/object/bbox/ymax' : (tf.train.Feature(
                    float_list=tf.train.FloatList(value=ymaxs))),
                'image/object/class/text': (tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=labels))),
                'image/object/is_crowd'  : (tf.train.Feature(
                    int64_list=tf.train.Int64List(value=is_crowds))),
                'image/object/area'      : (tf.train.Feature(
                    float_list=tf.train.FloatList(value=areas))),
                'image/object/mask'      : (tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=masks))),
            })).SerializeToString()
    # pylint: enable=bad-whitespace
    decoded_tensors = decoder.decode(
        tf.convert_to_tensor(value=serialized_example))

    results = tf.nest.map_structure(lambda x: x.numpy(), decoded_tensors)

    self.assertAllEqual(
        (image_height, image_width, 3), results['image'].shape)
    self.assertAllEqual(image_content, results['image'])
    self.assertEqual(tf_example_utils.DUMP_SOURCE_ID, results['source_id'])
    self.assertEqual(image_height, results['height'])
    self.assertEqual(image_width, results['width'])
    self.assertAllEqual(
        (num_instances,), results['ground_truth_classes'].shape)
    self.assertAllEqual(
        (num_instances,), results['ground_truth_is_crowd'].shape)
    self.assertAllEqual(
        (num_instances,), results['ground_truth_area'].shape)
    self.assertAllEqual(
        (num_instances, 4), results['ground_truth_boxes'].shape)
    self.assertAllEqual(
        (num_instances, image_height, image_width),
        results['ground_truth_instance_masks'].shape)
    self.assertAllEqual(
        (num_instances,), results['ground_truth_instance_masks_png'].shape)
    self.assertAllEqual(
        [2, 0], results['ground_truth_classes'])
    self.assertAllEqual(
        [True, False], results['ground_truth_is_crowd'])
    self.assertNDArrayNear(
        [0.25 * image_height * image_width, 0.75 * image_height * image_width],
        results['ground_truth_area'], 1e-4)
    self.assertNDArrayNear(
        [[0, 0, 0.5, 0.5], [0, 0.25, 1.0, 1.0]],
        results['ground_truth_boxes'], 1e-4)
    self.assertNDArrayNear(
        mask_content, results['ground_truth_instance_masks'], 1e-4)
    self.assertAllEqual(
        masks, results['ground_truth_instance_masks_png'])


if __name__ == '__main__':
  tf.test.main()
