import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib.slim import xavier_initializer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info('tensorflow version : ' + tf.__version__)

tf.app.flags.DEFINE_integer('num_feature', 5000, 'Number of features')
tf.app.flags.DEFINE_integer('num_hidden', 128, 'Number of hidden units of RNN cell')
tf.app.flags.DEFINE_integer('num_layer', 3, 'Number of hidden layers')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Mini-batch size')
tf.app.flags.DEFINE_integer('training_epoch', 300, 'Number of training epoch')
tf.app.flags.DEFINE_float('lr', 0.001, 'Learning rate')
tf.app.flags.DEFINE_float('keep_prob', 1.0, 'Dropout keep probability')
tf.app.flags.DEFINE_string('tf_records', None, 'TF records file path')
tf.app.flags.DEFINE_string(
    'checkpoint_dir', '/tmp/rumor_rnn/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')
tf.app.flags.DEFINE_boolean('is_train', True, 'Determine train or test')

FLAGS = tf.app.flags.FLAGS

def main(_):

    # if not FLAGS.tf_records:
    #     raise ValueError("You must supply the PATH of TF records file with --tf_records")

    batch_size = FLAGS.batch_size
    num_feature = FLAGS.num_feature

    graph = tf.Graph()

    with graph.as_default():

        train_inputs = tf.placeholder(shape=(batch_size, None, num_feature),
                                      dtype=tf.float32,
                                      name='train_inputs')

        train_inputs_length = tf.placeholder(shape=(batch_size,),
                                             dtype=tf.int32,
                                             name='train_inputs_length')

        train_labels = tf.placeholder(shape=(batch_size, None),
                                      dtype=tf.int32,
                                      name='train_label')
                                      
        valid_inputs = tf.placeholder(shape=(None, None, num_feature),
                                      dtype=tf.float32,
                                      name='valid_inputs')

        valid_inputs_length = tf.placeholder(shape=(None,),
                                             dtype=tf.int32,
                                             name='valid_inputs_length')

        valid_labels = tf.placeholder(shape=(None, None),
                                      dtype=tf.int32,
                                      name='valid_label')

        test_inputs = tf.placeholder(shape=(None, None, num_feature),
                                     dtype=tf.float32,
                                     name='test_inputs')
        
        test_inputs_length = tf.placeholder(shape=(None,),
                                            dtype=tf.int32,
                                            name='test_inputs_length')

        test_labels = tf.placeholder(shape=(None, None),
                                     dtype=tf.int32,
                                     name='test_label')
        
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

        test_merged = tf.summary.merge([tf.summary.scalar('valid_accuracy', valid_accuracy),
                                        tf.summary.scalar('test_accuracy', test_accuracy)])

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
            tf.gfile.MakeDirs('logdir')

            tf.global_variables_initializer().run()
            logger.info("initialized")

            train_writer = tf.summary.FileWriter(logdir, graph=sess.graph)

            for step in xrange(FLAGS.training_epoch): # training_epoch should be changed to num_steps
                batch = None
                if not batch:
                    raise ValueError("There are no batch data")

                feed = batch.feed_next()
                _ = sess.run([train_op], feed)

                if step % 50 == 0:
                    summary = sess.run([train_merged], feed)
                    train_writer.add_summary(summary, step)
            
                if step % 1000 == 0:
                    summary = sess.run([test_merged], feed)
                    train_writer.add_summary(summary, step)
    
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