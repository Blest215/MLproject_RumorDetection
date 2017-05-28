import os
import sys
import json
import operator
import re
import math
import random
import threading

import tensorflow as tf
import numpy as np

from datetime import datetime, timedelta
from textblob import TextBlob as tb

word_counter = {}
longest_topic = 0

tf.app.flags.DEFINE_string('dataset_directory', '/tmp/',
                           'Data directory')
tf.app.flags.DEFINE_string('output_directory', '/tmp/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 64,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 8,
                            'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('test_shards', 8,
                            'Number of shards in test TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 4,
                            'Number of threads to preprocess the topics.')
FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_words(text):
    return re.compile('\w+').findall(text)


class Topic:
    def __init__(self, file_path, label):
        # parsing data
        self.parse_data = []
        self.file_path = file_path
        self.label = label
        data_file = open(file_path)
        counter = 0
        global word_counter
        for l in data_file:
            try:
                j = json.loads(l)
                text = j['text'] #.decode()
                self.parse_data.append((datetime.strptime(j['created_at'], "%a %b %d %H:%M:%S +0000 %Y"), text))
                words = get_words(text)
                for word in words:
                    if word in word_counter:
                        word_counter[word] += 1
                    else:
                        word_counter[word] = 1
                counter += 1
            except:
                pass
        # sort according to date
        global longest_topic
        if counter > longest_topic:
            longest_topic = counter
        self.parse_data.sort()

    # output : array of feature
    def get_feature(self, length):
        # each document_set

        def tf(word, blob):
            return float(blob.words.count(word)) / float(len(blob.words))

        def n_containing(word, bloblist):
            return float(sum(1 for blob in bloblist if word in blob.words))

        def idf(word, bloblist):
            return float(math.log(len(bloblist)) / float(1 + n_containing(word, bloblist)))

        def tfidf(word, blob, bloblist):
            return float(tf(word, blob) * float(idf(word, bloblist)))

        def twit_to_tf_dict(text):
            blob = tb(text)
            return {word: tf(word, blob) for word in blob.words}

        self.tf_data = []
        for date, text in self.parse_data:
            self.tf_data.append(twit_to_tf_dict(text))

        features = []
        counter = 0
        for data in self.tf_data:
            global word_counter
            vector = {}
            for w in word_counter:
                vector[w] = 0.0
            for w in data:
                if w in vector:
                    vector[w] += data[w]
            features.append(np.array(list(vector.values())).reshape((1, 5000)))
            counter += 1

        return features, counter, self.label, self.file_path


def read_data_sets():
    """read data from files in directory"""
    data_dir = FLAGS.dataset_directory

    print('Determining list of input files from %s.' % data_dir)

    jpeg_file_path = '%s/*' % data_dir
    matching_files = tf.gfile.Glob(jpeg_file_path)
    topics = map(lambda filepath: Topic(filepath, int('nonrumor' not in filepath)),
        matching_files)
    num_topics = len(topics)

    # shuffle topics
    random.shuffle(topics)

    # sort word_counter
    global word_counter

    # extract top FLAGS.K words
    word_counter = sorted(word_counter.items(), key=operator.itemgetter(1), reverse=True)[:5000]
    word_counter = [w[0] for w in word_counter]

    train_ratio = 0.8
    train_size = int(train_ratio*num_topics)
    validation_ratio = 0.1
    validation_size = int(validation_ratio*num_topics)

    train = topics[:train_size]
    validation = topics[train_size:train_size+validation_size]
    test = topics[train_size+validation_size:]

    print('Found %d JSON files inside %s.' %
          (len(matching_files), data_dir))

    # write_tfrecord(train, "train", FLAGS.train_shards)
    # write_tfrecord(validation, "valid", FLAGS.validation_shards)
    write_tfrecord(test, "test", FLAGS.test_shards)


def write_tfrecord(topics, name, num_shards):
    num_topics = len(topics)

    # Break all topics into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, num_topics, FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    threads = []
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, topics, num_shards)
        t = threading.Thread(target=batch_write_tfrecord, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d topics in data set.' %
          (datetime.now(), num_topics))
    sys.stdout.flush()


def batch_write_tfrecord(thread_index, ranges, name, topics, num_shards):
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        topics_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in topics_in_shard:
            topic = topics[i]
            features, length, label, path = topic.get_feature(longest_topic)
            example = tf.train.Example(features=tf.train.Features(feature={
                'tweets': _bytes_feature([f.tostring() for f in features]),
                'length': _int64_feature(length),
                'vector_size': _int64_feature(5000),
                'label': _int64_feature(label),
                'file_path': _byte_feature(tf.compat.as_bytes(path))
            }))

            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d topics in thread batch.' %
                      (datetime.now(), thread_index, counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d topics to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d topics to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def main(_):
    read_data_sets()

if __name__ == '__main__':
    tf.app.run()
