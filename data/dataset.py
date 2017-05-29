"""Provides data for rumor detection"""

import os
import tensorflow as tf

slim = tf.contrib.slim

# TODO(nsilberman): Add tfrecord file type once the script is updated.
_FILE_PATTERN = '%s-*'

_SPLITS_TO_SIZES = {
    'train': 793,
    'validation': 99,
    'test': 100
}

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'label': 'The label id of the image, integer between 0 and 999',
    'label_text': 'The text of the label.',
    'object/bbox': 'A list of bounding boxes.',
    'object/label': 'A list of labels, one per each object.',
}

_NUM_CLASSES = 2


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading ImageNet.

  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in _SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'tweets/ix0': tf.VarLenFeature(
          dtype=tf.int64),
      'tweets/ix1': tf.VarLenFeature(
          dtype=tf.int64),
      'tweets/values': tf.VarLenFeature(
          dtype=tf.float32),
      'tweets/shape': tf.FixedLenFeature(
          [2], dtype=tf.int64),
      'label': tf.FixedLenFeature(
          [], dtype=tf.int64),
      'file_path': tf.FixedLenFeature(
          [], dtype=tf.string, default_value='')
  }

  items_to_handlers = {
      'tweets/ix0': slim.tfexample_decoder.Tensor('tweets/ix0'),
      'tweets/ix1': slim.tfexample_decoder.Tensor('tweets/ix1'),
      'tweets/values': slim.tfexample_decoder.Tensor('tweets/values'),
      'tweets/shape': slim.tfexample_decoder.Tensor('tweets/shape'),
      'label': slim.tfexample_decoder.Tensor('label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES)
