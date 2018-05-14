import os
import urllib.request
import gzip
import pickle
import numpy as np


def get_data():
    path = os.path.join(os.getcwd(), 'mnist.pkl.gz')
    if not os.path.isfile(path):
        data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
        urllib.request.urlretrieve(data_url, "mnist.pkl.gz")
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

        return train_set, valid_set, test_set


def split_set_to_features_and_output(data_set):
    return data_set[0], data_set[1]


def standardize_input(train_x, valid_x, test_x):
    train_mean = np.mean(train_x, axis=0)  # TODO: should it be axewise?
    train_std = np.std(train_x)
    train_x = (train_x - train_mean) / train_std
    valid_x = (valid_x - train_mean) / train_std
    test_x = (test_x - train_mean) / train_std

    return train_x, valid_x, test_x


def array_to_one_hot(y, number_of_classes=None):
    if number_of_classes is None:
        number_of_classes = np.max(y) + 1

    number_of_samples = y.shape[0]
    one_hot = np.zeros((number_of_samples, number_of_classes))
    one_hot[np.arange(number_of_samples), y] = 1

    return one_hot
