from tensorflow.contrib.learn.python.learn.datasets import base
import tensorflow as tf
import os
from datetime import datetime, timedelta
import numpy
import json
import operator
import re
import math
import random
from textblob import TextBlob as tb

word_counter = {}
tweet_counter = 0

flags = tf.app.flags
FLAGS = flags.FLAGS

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def getWords(text):
    return re.compile('\w+').findall(text)


class Topic:
    def __init__(self, file_path):
        # parsing data
        self.parse_data = []
        data_file = open(file_path)
        for l in data_file:
            try:
                j = json.loads(l)
                text = j['text'].decode()
                self.parse_data.append((datetime.strptime(j['created_at'], "%a %b %d %H:%M:%S +0000 %Y"), text))
                words = getWords(text)
                for word in words:
                    if word in word_counter:
                        word_counter[word] += 1
                    else:
                        word_counter[word] = 1
                global tweet_counter
                tweet_counter += 1
            except:
                pass
        # sort according to date
        self.parse_data.sort()

    # output : array of feature
    def get_feature(self):
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
        for data in self.tf_data:
            global word_counter
            vector = {}
            for w in word_counter:
                vector[w] = 0.0
            for w in data:
                if w in vector:
                    vector[w] += data[w]
            features.append(numpy.array(vector.values()))
        features = numpy.array(features)
        print(features.shape)
        return features


class DataSet:
    def __init__(self, features, labels):
        self._features = features
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = len(features)
        print(self._num_examples)

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._features = [self._features[i] for i in perm0]
            self._labels = [self._labels[i] for i in perm0]
        # Go to the next epoch
        if start + batch_size >= self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            features_rest_part = self._features[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._features = [self._features[i] for i in perm]
                self._labels = [self._labels[i] for i in perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            if end == 0:
                return numpy.array(features_rest_part), numpy.array(labels_rest_part)
            features_new_part = self._features[start:end]
            labels_new_part = self._labels[start:end]
            return numpy.concatenate((features_rest_part, features_new_part), axis=0), \
                   numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._features[start:end], self._labels[start:end]


# read data from files in directory
def read_data_sets():
    features = []  # array of features
    topics = {} # array of topics
    labels = [] # array of labels

    for dirname, dirnames, filenames in os.walk('..'):
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            # only files that contain 'Information' or 'Rumor' in its name
            if bool(re.search('\w*rumor_\d*.json$', file_path)):
                print(file_path)
                new_topic = Topic(file_path)
                topics[file_path] = new_topic

    # sort word_counter
    global word_counter, tweet_counter

    # extract top FLAGS.K words
    word_counter = sorted(word_counter.items(), key=operator.itemgetter(1), reverse=True)[:5000]
    word_counter = [w[0] for w in word_counter]
    print(word_counter)

    # print total number of tweets
    print("Processed total %s tweets" % tweet_counter)

    # get feature after count finishes
    for file_path in topics:
        print("feature from %s" % file_path)
        features.append(topics[file_path].get_feature())
        if "nonrumor" in file_path:
            labels.append(0)
        else:
            labels.append(1)

    # get longest and label should be same
    assert len(features) == len(labels)

    train_ratio = 0.8
    validation_ratio = 0.1
    # split train/validation/test set according to ratio
    train_size = int(train_ratio*len(features))
    train_features = features[:train_size]
    train_labels = labels[:train_size]
    write_tfrecord(train_features, train_labels, "train")
    # train = DataSet(train_features, train_labels)

    validation_size = int(validation_ratio*len(features))
    validation_features = features[train_size:train_size+validation_size]
    validation_labels = labels[train_size:train_size+validation_size]
    write_tfrecord(validation_features, validation_labels, "valid")
    # validation = DataSet(validation_features, validation_labels)

    test_features = features[train_size+validation_size:]
    test_labels = labels[train_size+validation_size:]
    write_tfrecord(test_features, test_labels, "test")
    # test = DataSet(test_features, test_labels)


def write_tfrecord(features, labels, name):
    # generate tf.record
    writer = tf.python_io.TFRecordWriter("%s.tfrecords" % name)
    for i in range(len(features)):
        example = tf.train.Example(features=tf.train.Features(feature={
            'tweets': _bytes_feature(features[i].tostring()),
            'length': _int64_feature(len(features[i])),
            'vector_size': _int64_feature(5000),
            'label': _int64_feature(labels[i])
        }))
        writer.write(example.SerializeToString())
    writer.close()

read_data_sets()
