""" Convert twitter data to TFRecords """
import tensorflow as tf

tf.app.flags.DEFINE_string('hello', 'Hello World!', 'Example argument')

FLAGS = tf.app.flags.FLAGS


def main(_):
    """ ENTRY POINT """
    print FLAGS.hello

if __name__ == 'main':
    tf.app.run()
