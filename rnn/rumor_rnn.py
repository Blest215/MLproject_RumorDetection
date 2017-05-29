import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib.slim import xavier_initializer
import logging
import os.path

logging.basicConfig(level=logging.DEBUG,
        format = "%(levelname) -1s %(asctime)s %(module)s:%(lineno)s %(funcName)s %(message)s")
logger = logging.getLogger(__name__)

logger.info('tensorflow version : ' + tf.__version__)

tf.app.flags.DEFINE_integer('num_feature', 5000, 'Number of features')
tf.app.flags.DEFINE_integer('num_hidden', 128, 'Number of hidden units of RNN cell')
tf.app.flags.DEFINE_integer('num_layer', 3, 'Number of hidden layers')
tf.app.flags.DEFINE_integer('batch_size', 16, 'Mini-batch size')
tf.app.flags.DEFINE_integer('train_epochs', 200, 'Number of training epoch')
tf.app.flags.DEFINE_float('lr', 0.001, 'Learning rate')
tf.app.flags.DEFINE_float('keep_prob', 0.5, 'Dropout keep probability')
tf.app.flags.DEFINE_string('tf_records', '../data', 'TF records file dir path')
tf.app.flags.DEFINE_string(
    'checkpoint_dir', '/tmp/rumor_rnn',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')
tf.app.flags.DEFINE_boolean('is_train', True, 'Determine train or test')

FLAGS = tf.app.flags.FLAGS

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'valid.tfrecords'
TEST_FILE = 'test.tfrecords'

def random_interval(tweet, length, interval):
    min_len, max_len = interval

    next_len = tf.constant(0,dtype=tf.int32)
    current_len = tf.constant(0,dtype=tf.int32)
    count = tf.constant(0,dtype=tf.int32)
    temp_tweet = tf.constant(0, dtype=tf.float32, shape=[1,FLAGS.num_feature])

    def condition(length, next_len, current_len, tweet, temp_tweet, count):
        return tf.less(next_len, length)

    def body(length, next_len, current_len, tweet, temp_tweet, count):
        interval = tf.random_uniform([], min_len, max_len+1, dtype=tf.int32)
        next_len = current_len + interval
        next_len = tf.cond(tf.less(length, next_len),
                lambda : length,
                lambda : next_len)
        summed = tf.reduce_sum(tweet[current_len:next_len], axis=0)
        normed = tf.nn.l2_normalize(summed, dim=0)
        temp = tf.reshape(normed, shape=[1, FLAGS.num_feature])
        temp_tweet = tf.cond(tf.equal(count,0),
                lambda : temp, 
                lambda : tf.concat([temp_tweet, temp], axis=0))
        current_len += interval
        count += 1
        return length, next_len, current_len, tweet, temp_tweet, count
    
    loop_vars = [length, next_len, current_len, tweet, temp_tweet, count]
    shape_inv = [length.get_shape(), next_len.get_shape(),
            current_len.get_shape(), tweet.get_shape(), 
            tf.TensorShape([None, FLAGS.num_feature]), count.get_shape()]

    _, _, _, _, temp_tweet, _ = tf.while_loop(condition, body, loop_vars, 
            shape_invariants=shape_inv)
    
    count = tf.shape(temp_tweet)[0]

    return temp_tweet, count
       
def read_and_decode(filename_queue, interval):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'tweets': tf.FixedLenFeature([], tf.string),
          'length': tf.FixedLenFeature([], tf.int64),
          'vector_size': tf.FixedLenFeature([], tf.int64),
          'label': tf.FixedLenFeature([], tf.int64),
          'file_path': tf.FixedLenFeature([], tf.string),
      })

  tweet = tf.decode_raw(features['tweets'], tf.float32)
  label = tf.cast(features['label'], tf.int32)
  length = 2*tf.cast(features['length'], tf.int32)
  tweet = tf.reshape(tweet, [length, FLAGS.num_feature])
  path = features['file_path']

  tweet, length = random_interval(tweet, length, interval)

  return tweet, length, label, path


def inputs(train, batch_size, num_epochs, num_threads, capacity,
        interval=(10,10)):
  if not num_epochs: num_epochs = None

  if train=='train':
    filename = os.path.join(FLAGS.tf_records, TRAIN_FILE)
  elif train=='valid':
    filename = os.path.join(FLAGS.tf_records, VALIDATION_FILE)
  elif train=='test': 
    filename = os.path.join(FLAGS.tf_records, TEST_FILE)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=num_epochs)

    tweet, length, label, path = read_and_decode(filename_queue, interval)

    tweets, lengths, labels, paths = tf.train.batch(
            [tweet, length, label, path], batch_size=FLAGS.batch_size,
            num_threads=num_threads, capacity=capacity, dynamic_pad = True)
        # Ensures a minimum amount of shuffling of examples.
        # min_after_dequeue=1000)

    return tweets, lengths, labels, paths

def main(_):

    # if not FLAGS.tf_records:
    #     raise ValueError("You must supply the PATH of TF records file with --tf_records")

    batch_size = FLAGS.batch_size
    num_feature = FLAGS.num_feature

    graph = tf.Graph()

    with graph.as_default():

        train_inputs, train_inputs_length, train_labels, train_paths = inputs('train', batch_size, FLAGS.train_epochs, 2, 3*batch_size, (6,14))

        logger.info(train_inputs)

        valid_inputs, valid_inputs_length, valid_labels, valid_paths = inputs('valid', 10, None, 1, 20)

        test_inputs, test_inputs_length, test_labels, test_paths = inputs('test', 10, None, 1, 20)
        
        keep_prob = tf.placeholder(tf.float32)

        def model(inputs, inputs_length, reuse=None):
            embedded = tf.layers.dense(inputs=inputs,
                                       units=FLAGS.num_hidden,
                                       kernel_initializer=xavier_initializer(),
                                       name='embedding_layer',
                                       reuse=reuse)

            cell = rnn.GRUCell(FLAGS.num_hidden, reuse=reuse)
            cell = rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            cell = rnn.MultiRNNCell([rnn.ResidualWrapper(cell) for i in range(FLAGS.num_layer)])

            outputs, final_state = tf.nn.dynamic_rnn(cell,
                                                     embedded,
                                                     inputs_length,
                                                     dtype=tf.float32)
                                                    #  cell.zero_state(batch_size, dtype=tf.float32))

            results = tf.layers.dense(inputs=final_state[FLAGS.num_layer - 1],
                                      units=2,
                                      kernel_initializer=xavier_initializer(),
                                      name='output_layer',
                                      reuse=reuse)

            return results

        logits = model(train_inputs, train_inputs_length)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(train_labels, depth=2, dtype=tf.float32), 
            logits=logits)
        
        loss = tf.reduce_mean(cross_entropy)

        train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

        train_prediction = tf.argmax(logits, 1)
        valid_prediction = tf.argmax(model(valid_inputs, valid_inputs_length, True), 1)
        test_prediction = tf.argmax(model(test_inputs, test_inputs_length, True), 1)

        train_correct = tf.equal(tf.cast(train_prediction, tf.int32), train_labels)
        valid_correct = tf.equal(tf.cast(valid_prediction, tf.int32), valid_labels)
        test_correct = tf.equal(tf.cast(test_prediction, tf.int32), test_labels)

        train_accuracy = tf.reduce_mean(tf.cast(train_correct, tf.float32))
        valid_accuracy = tf.reduce_mean(tf.cast(valid_correct, tf.float32))
        test_accuracy = tf.reduce_mean(tf.cast(test_correct, tf.float32))

        train_merged = tf.summary.merge([tf.summary.scalar('train_loss', loss),
                                         tf.summary.scalar('train_accuracy', train_accuracy)])
        
        valid_acc = tf.placeholder(tf.float32)
        test_acc = tf.placeholder(tf.float32)
        # test_merged = tf.summary.merge([tf.summary.scalar('test_accuracy', test_accuracy), tf.summary.scalar('valid_accuracy', valid_accuracy)])
        valid_summary = tf.summary.scalar('valid_accuracy', valid_acc)
        test_summary = tf.summary.scalar('test_accuracy', test_acc)

        saver = tf.train.Saver()
    
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        logger.info('Trainable vars:')
        for v in var_list:
            logger.info('  %s %s', v.name, v.get_shape())

        ## end of graph

    with tf.Session(graph=graph) as sess:

        if FLAGS.is_train:
            logdir = FLAGS.checkpoint_dir + '/train'
            if tf.gfile.Exists(logdir):
                logger.info('Delete logdir')
                tf.gfile.DeleteRecursively(logdir)
            tf.gfile.MakeDirs(logdir)

            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            sess.run(init_op)
            logger.info("initialized")

            train_writer = tf.summary.FileWriter(logdir, graph=sess.graph)

            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                step = 0
                while not coord.should_stop():
                    train_feed = { keep_prob: FLAGS.keep_prob}
                    summary, _, path = sess.run([train_merged, train_op, train_paths], feed_dict=train_feed)

                    if step % 10 == 0:
                        train_writer.add_summary(summary, step)
            
                    if step % 100 == 0:
                        temp_v = 0
                        temp_t = 0
                        for i in xrange(10):
                            valid, test = sess.run([valid_accuracy, test_accuracy], {keep_prob:1.0})
                            temp_v += valid
                            temp_t += test
                        temp_v = temp_v / 10.0
                        temp_t = temp_t / 10.0

                        test_, valid_ = sess.run([test_summary, valid_summary], {test_acc:temp_t, valid_acc:temp_v})
                        train_writer.add_summary(test_, step)
                        train_writer.add_summary(valid_, step)

                    step += 1

            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' % (FLAGS.train_epochs, step))
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)

            # for step in xrange(FLAGS.training_epoch): # training_epoch should be changed to num_steps
            #     batch = None
            #     if not batch:
            #         raise ValueError("There are no batch data")

            #     feed = batch.feed_next()
            #     _ = sess.run([train_op], feed)

            #     if step % 50 == 0:
            #         summary = sess.run([train_merged], feed)
            #         train_writer.add_summary(summary, step)
            
            #     if step % 1000 == 0:
            #         summary = sess.run([test_merged], feed)
            #         train_writer.add_summary(summary, step)
    
            model_path = FLAGS.checkpoint_dir + '/model.ckpt'
            save_path = saver.save(sess, model_path)
            logger.info("Model saved in file %s" % save_path)
            train_writer.close()

        else:
            model_path = FLAGS.checkpoint_dir + '/model.ckpt'
            saver.restore(sess, model_path)
            logger.info("Model restored.")

if __name__ == '__main__':
    tf.app.run()
