""" Convert twitter data to TFRecords """
import tensorflow as tf

import argparse
from data.dataset import *


def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    Args:
    batch_size: The batch size will be baked into both placeholders.
    Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    FLAGS = flags.FLAGS
    inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, None, FLAGS.K))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

    return inputs_placeholder, labels_placeholder


def fill_feed_dict(data_set, inputs_placeholder, labels_placeholder):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
    }
    Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
    Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.

    features_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
    feed_dict = {
        inputs_placeholder: features_feed,
        labels_placeholder: labels_feed,
    }
    return feed_dict


def main(_):
    """ ENTRY POINT """
    print(FLAGS.hello)

if __name__ == 'main':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_features',
        type=int,
        default=0,
        help='Number of Features'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run()

