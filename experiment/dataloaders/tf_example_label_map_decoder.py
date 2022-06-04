"""Tensorflow Example proto decoder for object detection.

A decoder to decode string tensors containing serialized tensorflow.Example
protos for object detection.
"""
import csv
import tensorflow as tf

from experiment.dataloaders import TfExampleDecoder


class TfExampleDecoderLabelMap(TfExampleDecoder):
  """Tensorflow Example proto decoder."""

  def __init__(self,
               label_map,
               include_mask=False,
               regenerate_source_id=False,
               mask_binarize_threshold=None):
    super(TfExampleDecoderLabelMap, self).__init__(
        include_mask=include_mask,
        regenerate_source_id=regenerate_source_id,
        mask_binarize_threshold=mask_binarize_threshold)

    self._keys_to_features.update({
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
    })
    name_to_id = self._process_label_map(label_map)
    self._name_to_id_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(list(name_to_id.keys()), dtype=tf.string),
            values=tf.constant(list(name_to_id.values()), dtype=tf.int64)),
        default_value=-1)

  def _process_label_map(self, label_map):
    if label_map.endswith('.csv'):
      name_to_id = self._process_csv(label_map)
    else:
      raise ValueError('The label map file is in incorrect format.')
    return name_to_id

  def _process_csv(self, label_map):
    name_to_id = {}
    with tf.io.gfile.GFile(label_map, 'r') as f:
      reader = csv.reader(f, delimiter=',')
      for row in reader:
        if len(row) != 2:
          raise ValueError('Each row of the csv label map file must be in '
                           '`id,name` format. length = {}'.format(len(row)))
        idx = int(row[0])
        name = row[1]
        name_to_id[name] = idx
    return name_to_id

  def _decode_classes(self, parsed_tensors):
    return self._name_to_id_table.lookup(
        parsed_tensors['image/object/class/text'])
