""" Preprocessing for ConvNet """
import tensorflow as tf


def preprocess(input, input_shape, size):
    def repeat(t, ratio):
        # [-1, 5000] --> [-1, 1, 5000]
        t = tf.expand_dims(t, 1)
        # [-1, 1, 5000] --> [-1, ratio, 5000]
        t = tf.tile(t, [1, ratio, 1])
        return tf.reshape(t, [-1, 5000])

    def shrink(t, length):
        rand = tf.random_uniform([1])
        start = tf.cond(tf.less(rand[0], 0.5), lambda: 0, lambda: 1)
        t = tf.strided_slice(t, [start, 0], [length, 5000], [2, 1])
        return t

    # Resize
    length = tf.to_int32(input_shape[0])
    ratio = tf.div(size, length) + 1
    inverse_ratio = tf.div(length, size)

    to_repeat = tf.greater(ratio, 1)
    to_shrink = tf.greater(inverse_ratio, 10)

    input = tf.case({
        to_repeat: lambda: repeat(input, ratio),
        to_shrink: lambda: shrink(input, length)
    }, default=lambda: input, exclusive=True)

    # input = tf.cond(to_repeat,
    #                 lambda: repeat(input, ratio),
    #                 lambda: input)

    return tf.random_crop(input, [size, 5000])


def get_preprocessing(is_training):
    if is_training:
        return preprocess