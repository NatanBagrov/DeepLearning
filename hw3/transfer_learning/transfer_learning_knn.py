from collections import defaultdict
import pickle
import os

import numpy as np
from keras import Sequential
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import matplotlib.pyplot as plt

from transfer_learning.cifar100vgg import cifar100vgg
from plotting import plot_parameter_influence_score


def get_best_parameters(cv_result: dict):
    index = np.argmax(cv_result['mean_test_score'])

    return cv_result['params'][index], cv_result['mean_test_score'][index]


def search_parameters_for_model(x, y, classifier, parameters: dict):
    print('search_parameters_for_model')
    clf = GridSearchCV(
        classifier(),
        parameters,
        n_jobs=1,
        verbose=2**31,
        cv=StratifiedShuffleSplit(n_splits=3, random_state=42),
        return_train_score=True,
    )
    clf.fit(x, y)

    print(clf)

    best_parameters, best_score = get_best_parameters(clf.cv_results_)
    print('%r with %r achieves %f' %(classifier, best_parameters, best_score))
    plot_parameter_influence_score(classifier.__name__, clf.cv_results_)

    return best_parameters, best_score


def search_parameters_for_knn(x, y):
    print('search_parameters_for_knn')
    parameters1, score1 = search_parameters_for_model(
        x, y,
        KNeighborsClassifier,
        {
            'n_neighbors': np.arange(1, 10, 2),
        }
    )
    print('parameters=%r score=%r' % (parameters1, score1))
    return parameters1


def main():
    if not os.path.isfile('feature-map.pkl'):
        pretrained = cifar100vgg(train=False)

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        print(x_train.mean(), x_train.std(), x_test.mean(), x_test.std(), x_train.shape)
        x_train = pretrained.normalize_production(x_train)
        x_test = pretrained.normalize_production(x_test)
        print(x_train.mean(), x_train.std(), x_test.mean(), x_test.std(), x_train.shape)
        pretrained.model.summary()
        pretrained_head = Sequential(pretrained.model.layers[:-2])
        pretrained_head.summary()
        print(x_train.mean(), x_train.std(), x_test.mean(), x_test.std(), x_train.shape)
        x_train = pretrained_head.predict(x_train)
        x_test = pretrained_head.predict(x_test)

        with open('feature-map.pkl', 'wb') as fh:
            pickle.dump((x_train, y_train, x_test, y_test), fh)  # TODO: change to numpy.dump
    else:
        with open('feature-map.pkl', 'rb') as fh:
            x_train, y_train, x_test, y_test = pickle.load(fh)  # TODO: change to numpy.load

    print(x_train.mean(), x_train.std(), x_test.mean(), x_test.std(), x_train.shape)

    for number_of_samples in (100, 1000, 10000):
        print('number_of_samples=%d' % number_of_samples)
        current_x_train, _, current_y_train, _ = train_test_split(x_train, y_train,
                                                  train_size=number_of_samples, random_state=42, stratify=y_train)
        parameters = {'n_neighbors': 9}# search_parameters_for_knn(x_train, y_train)
        embedded_tail = KNeighborsClassifier(**parameters)
        embedded_tail.fit(current_x_train, current_y_train)
        train_accuracy = embedded_tail.score(x_train, y_train)
        print('Train score after training on %d samples is %f' % (number_of_samples, train_accuracy))
        test_accuracy = embedded_tail.score(x_test, y_test)
        print('Test score after training on %d samples is %f' % ( number_of_samples, test_accuracy))


if "__main__" == __name__:
    main()

