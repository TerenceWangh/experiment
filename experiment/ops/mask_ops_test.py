import numpy as np
import tensorflow as tf
from experiment.ops import mask_ops


class MaskUtilsTest(tf.test.TestCase):

  def testPasteInstanceMasks(self):
    image_height = 10
    image_width = 10
    mask_height = 6
    mask_width = 6
    masks = np.random.randint(0, 255, (1, mask_height, mask_width))
    detected_boxes = np.array([[0.0, 2.0, mask_width, mask_height]])

    _ = mask_ops.paste_instance_masks(
        masks, detected_boxes, image_height, image_width)

  def testPasteInstanceMasksV2(self):
    image_height = 10
    image_width = 10
    mask_height = 6
    mask_width = 6
    masks = np.random.randint(0, 255, (1, mask_height, mask_width))
    detected_boxes = np.array([[0.0, 2.0, mask_width, mask_height]])

    image_masks = mask_ops.paste_instance_masks_v2(
        masks, detected_boxes, image_height, image_width)

    self.assertNDArrayNear(
        image_masks[:, 2:8, 0:6],
        np.array(masks > 0.5, dtype=np.uint8),
        1e-5)


if __name__ == '__main__':
  tf.test.main()
