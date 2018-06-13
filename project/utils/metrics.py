from keras import backend as K
import numpy as np


def keras_zero_one_loss(y_true, y_predicted):
    return K.mean(K.equal(y_true, y_predicted))


def numpy_zero_one_loss(y_true, y_predicted):
    return np.mean(np.equal(np.array(y_true), np.array(y_predicted)))