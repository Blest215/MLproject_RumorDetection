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
longest_topic = 0

flags = tf.app.flags
FLAGS = flags.FLAGS

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def getWords(text):
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
                words = getWords(text)
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
            features.append(numpy.array(list(vector.values())))
            counter += 1

        return numpy.array(features), len(features), self.label, self.file_path

# read data from files in directory
def read_data_sets():
    topics = [] # array of topics
    num_topic = 0

    for dirname, dirnames, filenames in os.walk('..'):
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            # only files that contain 'Information' or 'Rumor' in its name
            if bool(re.search('\w*rumor_\d*.json$', file_path)):
                print(file_path)
                num_topic += 1
                if "nonrumor" in file_path:
                    new_topic = Topic(file_path, 0)
                else:
                    new_topic = Topic(file_path, 1)
                topics.append(new_topic)

    # shuffle topics
    random.shuffle(topics)

    # sort word_counter
    global word_counter

    # extract top FLAGS.K words
    word_counter = sorted(word_counter.items(), key=operator.itemgetter(1), reverse=True)[:5000]
    word_counter = [w[0] for w in word_counter]

    train_ratio = 0.8
    train_size = int(train_ratio*num_topic)
    validation_ratio = 0.1
    validation_size = int(validation_ratio*num_topic)

    train = []
    validation = []
    test = []
    for i in range(train_size):
        train.append(topics.pop())
    for i in range(validation_size):
        validation.append(topics.pop())
    for t in topics:
        test.append(t)

    write_tfrecord(train, "train")
    write_tfrecord(validation, "valid")
    write_tfrecord(test, "test")

def write_tfrecord(topics, name):
    # generate tf.record for train, validation, test group
    writer = tf.python_io.TFRecordWriter("%s.tfrecords" % name)
    for t in topics:
        features, length, label, path = t.get_feature(longest_topic)
        example = tf.train.Example(features=tf.train.Features(feature={
            'tweets': _bytes_feature(features.tostring()),
            'length': _int64_feature(length),
            'vector_size': _int64_feature(5000),
            'label': _int64_feature(label),
            'file_path': _byte_feature(tf.compat.as_bytes(path))
        }))
        writer.write(example.SerializeToString())
    writer.close()

read_data_sets()
