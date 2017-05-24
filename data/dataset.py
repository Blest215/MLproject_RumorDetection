from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
import os
from datetime import datetime, timedelta
import numpy
import json
import operator
import re
import math
import nltk
from textblob import TextBlob as tb

word_counter = {}


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
                text = j['text']
                self.parse_data.append((datetime.strptime(j['created_at'], "%a %b %d %H:%M:%S +0000 %Y"), text))
                words = getWords(text)
                for word in words:
                    if word in word_counter:
                        word_counter[word] += 1
                    else:
                        word_counter[word] = 1
            except:
                pass

        # sort according to date
        self.parse_data.sort()

    # get feature from the topic : interval - hour
    # output : array of feature
    def get_feature(self, interval=5):
        # each document_set is time interval
        document_sets = []
        document_set = []
        start_datetime = self.parse_data[0][0]
        for date, text in self.parse_data:
            diff = date - start_datetime
            if diff < timedelta(hours=interval):
                document_set.append(tb(text))
            else:
                document_sets.append(document_set)
                start_datetime = start_datetime + timedelta(hours=interval)
                while date - start_datetime > timedelta(hours=interval):
                    document_sets.append([])
                    start_datetime = start_datetime + timedelta(hours=interval)
                document_set = [tb(text)]

        def tf(word, blob):
            return float(blob.words.count(word)) / float(len(blob.words))

        def n_containing(word, bloblist):
            return float(sum(1 for blob in bloblist if word in blob.words))

        def idf(word, bloblist):
            return float(math.log(len(bloblist)) / float(1 + n_containing(word, bloblist)))

        def tfidf(word, blob, bloblist):
            return float(tf(word, blob) * float(idf(word, bloblist)))


        # print document_set

        for bloblist in document_sets:
            for i, blob in enumerate(bloblist):
                print("Top words in document {}".format(i + 1))
                scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
                sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for word, score in sorted_words[:]:
                    print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))

        # TODO : calculate tf-idf value
        # each document_set in document_sets become one element of input of neural network
        # so length of one input becomes len(document_sets)
        # calculate average tf-idf value of each document_set
        return []

    # def print(self):
    #     for t in self.parse_data:
    #         print(t)


class DataSet:
    def __init__(self, features, labels):
        self._features = features
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = len(features)

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._features = self._features[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._features[start:end], self._labels[start:end]


def extract_features(f):
    pass

def extract_labels(f):
    pass


# read data from files in directory
def read_data_sets(train_ratio, interval=5):
    features = []  # array of features
    topics = [] # array of topics
    labels = [] # array of labels
    for dirname, dirnames, filenames in os.walk('.'):
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            # only files that contain 'Information' or 'Rumor' in its name
            if "nonrumor" in file_path or "rumor" in file_path:
                print(file_path)
                new_topic = Topic(file_path)
                topics.append(new_topic)
                features.append(new_topic.get_feature(interval=interval))
                if "nonrumor" in file_path:
                    labels.append(0)
                else:
                    labels.append(1)

    # length of feature and label should be same
    assert len(features) == len(labels)

    # sort word_counter
    global word_counter

    word_counter = sorted(word_counter.items(), key=operator.itemgetter(1))[:5000]

    # split train/test set according to ratio
    train_size = int(train_ratio*len(features))
    train_features = features[:train_size]
    train_labels = labels[:train_size]
    test_features = features[train_size:]
    test_labels = labels[train_size:]
    train = DataSet(train_features, train_labels)
    validation = DataSet([], [])
    test = DataSet(test_features, test_labels)
    return base.Datasets(train=train, validation=validation, test=test)

read_data_sets(0.8, 5)

