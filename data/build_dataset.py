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
from multiprocessing import Pool
from textblob import TextBlob as tb
from collections import Counter

tf.app.flags.DEFINE_string('dataset_directory', '/tmp/',
                           'Data directory')
tf.app.flags.DEFINE_string('output_directory', '/tmp/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 32,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 8,
                            'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('test_shards', 8,
                            'Number of shards in test TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 4,
                            'Number of threads to preprocess the topics.')

tf.app.flags.DEFINE_integer('num_words', 5000,
                            'Number of words to use.')
FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_words(text):
    return re.compile('\w+').findall(text)


class Topic:
    word_counts = None
    def __init__(self, file_path, label):
        # parsing data
        self.file_path = file_path
        self.label = label
        self.word_counter = Counter()

        def line2textblob(line):
            """Convert line(tweet) to TextBlob"""
            json_obj = json.loads(line)
            text = json_obj['text']
            created_at = datetime.strptime(json_obj['created_at'], "%a %b %d %H:%M:%S +0000 %Y")
            textblob = tb(text)
            self.word_counter.update(textblob.word_counts)

            return created_at, textblob.word_counts, len(textblob.words)

        topic_file = open(file_path)
        self.tweets = map(line2textblob, topic_file)
        self.tweets.sort()

    # output : array of feature
    def get_feature(self):
        ix0 = []
        ix1 = []
        values = []
        for l, tweet in enumerate(self.tweets):
            word_counts = tweet[1]
            num_words = tweet[2]
            for word, count in word_counts.items():
                for d, tup in enumerate(Topic.word_counts):
                    if word == tup[0]:
                        ix0.append(l)
                        ix1.append(d)
                        values.append(float(count) / float(num_words))
                        break

        return ix0, ix1, values, len(self.tweets), self.label, self.file_path


def filepath2topic(filepath):
    return Topic(filepath, int('nonrumor' not in filepath))

def read_data_sets():
    """read data from files in directory"""
    data_dir = FLAGS.dataset_directory

    print('Determining list of input files from %s.' % data_dir)

    jpeg_file_path = '%s/*' % data_dir
    matching_files = tf.gfile.Glob(jpeg_file_path)
    topics = Pool(FLAGS.num_threads).map(filepath2topic, matching_files)
    # shuffle topics
    random.shuffle(topics)

    def merge_word_counters(accum, topic):
        accum.update(topic.word_counter)
        return accum

    merged_counters = reduce(merge_word_counters, topics, Counter())

    # sort word_counter and extract top FLAGS.K words
    Topic.word_counts = merged_counters.most_common(FLAGS.num_words)

    # Split dataset
    num_topics = len(topics)
    train_ratio = 0.8
    train_size = int(train_ratio*num_topics)
    validation_ratio = 0.1
    validation_size = int(validation_ratio*num_topics)

    train = topics[:train_size]
    validation = topics[train_size:train_size+validation_size]
    test = topics[train_size+validation_size:]

    print('Found %d JSON files inside %s.' %
          (len(matching_files), data_dir))

    write_tfrecord(train, "train", FLAGS.train_shards)
    write_tfrecord(validation, "valid", FLAGS.validation_shards)
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
            ix0, ix1, values, length, label, path = topic.get_feature()
            example = tf.train.Example(features=tf.train.Features(feature={
                'tweets/ix0': _int64_feature(ix0),
                'tweets/ix1': _int64_feature(ix1),
                'tweets/values': _float_feature(values),
                'tweets/shape': _int64_feature([length, FLAGS.num_words]),
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
