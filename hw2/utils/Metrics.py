import numpy as np


def accuracy(y_true, y_predicted):
    assert y_true.shape == y_predicted.shape

    true_classes = np.argmax(y_true, axis=1)
    predicted_classes = np.argmax(y_predicted, axis=1)
    correctly_predicted = np.count_nonzero(true_classes == predicted_classes) / y_true.shape[0]

    return correctly_predicted
