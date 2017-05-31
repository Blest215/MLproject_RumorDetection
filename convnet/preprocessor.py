""" Preprocessing for ConvNet """
import tensorflow as tf


def preprocess(input, input_shape, size):
    def repeat(t, ratio):
        # [-1, 5000] --> [-1, 1, 5000]
        t = tf.expand_dims(t, 1)
        # [-1, 1, 5000] --> [-1, ratio, 5000]
        t = tf.tile(t, [1, ratio, 1])
        return tf.reshape(t, [-1, 5000])

    # Resize
    ratio = tf.div(size, tf.to_int32(input_shape[0])) + 1
    input = tf.cond(tf.greater(ratio, 1),
                    lambda: repeat(input, ratio),
                    lambda: input)

    return tf.random_crop(input, [size, 5000])


def get_preprocessing(is_training):
    if is_training:
        return preprocess