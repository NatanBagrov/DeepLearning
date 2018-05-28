from collections import defaultdict

import numpy as np
from keras import Sequential
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import matplotlib.pyplot as plt

from transfer_learning.cifar100vgg import cifar100vgg


def main():
    pretrained = cifar100vgg(train=False)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    x_train = pretrained.normalize_production(x_train)
    x_test = pretrained.normalize_production(x_test)
    pretrained.model.summary()
    pretrained_head = Sequential(pretrained.model.layers[:-2])
    pretrained_head.summary()
    x_train = pretrained_head.predict(x_train)
    embeded_train = KNeighborsClassifier()

    for number_of_samples in (100, 1000, 10000):
        current_x_train, _, current_y_train, _ = train_test_split(x_train, y_train,
                                                  train_size=number_of_samples, random_state=42, stratify=y_train)
        parameters = {} # tune_knn_parameters(pretrained.normalize_production, pretrained.model, x_train, y_train)
        embeded_train.fit(current_x_train, current_y_train)
        train_accuracy = embeded_train.score(x_train, y_train)
        print('Train score after training on %d samples is %f' % (number_of_samples, train_accuracy))
        test_accuracy = embeded_train.score(x_test, y_test)
        print('Test score after training on %d samples is %f' % ( number_of_samples, test_accuracy))


if "__main__" == __name__:
    main()

