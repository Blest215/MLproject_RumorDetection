from data import read_data_sets
from data import placeholder_inputs, fill_feed_dict
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
tf.app.flags.DEFINE_string('hello', 'Hello World!', 'Example argument')
flags.DEFINE_integer('K', 5000, 'Number of top-K words')
flags.DEFINE_integer('batch_size', 10, 'Size of batch')
flags.DEFINE_integer('max_steps', 1000, 'Maximum number of steps')

train, validation, test = read_data_sets(train_ratio=.8, validation_ratio=.1, interval=(5, 15))

input_placeholder, label_placeholder = placeholder_inputs(FLAGS.batch_size)
sess = tf.Session()

for step in xrange(FLAGS.max_steps):
    feed_dict = fill_feed_dict(train, input_placeholder, label_placeholder)
    sess.run([], feed_dict=feed_dict)
    # input sequence length -> dimension = input_length * 5000
    input_length = FLAGS.input_length
