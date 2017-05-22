from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
import os
from datetime import datetime, timedelta
import numpy


class Topic:
    def __init__(self, file_path):
        # parsing data
        self.parse_data = []
        with open(file_path) as f:
            for l in f:
                # print(l)
                try:
                    date = l.split("<d>")[1].split("</d>")[0]
                    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                    text = l.split("<t>")[1].split("</t>")[0]
                except:
                    pass
                self.parse_data.append((date, text))
        # sort according to date
        self.parse_data.sort()

    # get feature from the topic : interval - hour
    # output : array of feature
    def get_feature(self, interval=5):
        documents = []
        document = []
        start_datetime = self.parse_data[0][0]
        for date, text in self.parse_data:
            diff = date - start_datetime
            if diff < timedelta(hours=interval):
                document.append(text)
            else:
                documents.append(document)
                start_datetime = start_datetime + timedelta(hours=interval)
                while date - start_datetime > timedelta(hours=interval):
                    documents.append([])
                    start_datetime = start_datetime + timedelta(hours=interval)
                document = [text]

        # TODO : calculate tf-idf value
        return []

    def print(self):
        for t in self.parse_data:
            print(t)


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
            if "Information" in file_path or "Rumor" in file_path:
                new_topic = Topic(file_path)
                topics.append(new_topic)
                features.append(new_topic.get_feature(interval=interval))
                if "Information" in file_path:
                    labels.append(0)
                else:
                    labels.append(1)

    # length of feature and label should be same
    assert len(features) == len(labels)

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

