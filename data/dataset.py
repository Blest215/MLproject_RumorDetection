from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
import os
from datetime import datetime, timedelta
import numpy


class Topic:
    def __init__(self, file_path):
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
        self.parse_data.sort()

    def get_feature(self, interval=5):
        documents = []
        document = []
        start_datetime = self.parse_data[0][0]
        print(start_datetime)
        for date, text in self.parse_data:
            print(date, text)
            diff = date - start_datetime
            if diff < timedelta(hours=interval):
                document.append(text)
            else:
                documents.append(document)
                start_datetime = start_datetime + timedelta(hours=interval)
                while date - start_datetime > timedelta(hours=interval):
                    documents.append([])
                    start_datetime = start_datetime + timedelta(hours=interval)
                    print("cut and ", start_datetime)
                document = [text]
        for d in documents:
            print(d)


        pass

    def print(self):
        for t in self.parse_data:
            print(t)


class DataSet:
    def __init__(self, features, labels, num_examples):
        self._features = features
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = num_examples

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


def read_data_sets():
    topics = []
    for dirname, dirnames, filenames in os.walk('.'):
        # print path to all subdirectories first.
        # for subdirname in dirnames:
        #    print(os.path.join(dirname, subdirname))

        # print path to all filenames.
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            if "Information" in file_path or "Rumor" in file_path:
                new_topic = Topic(file_path)
                #new_topic.print()
                topics.append(new_topic)
    topics[0].get_feature()
    train = DataSet()
    validation = DataSet()
    test = DataSet()
    return base.Datasets(train=train, validation=validation, test=test)

read_data_sets()

