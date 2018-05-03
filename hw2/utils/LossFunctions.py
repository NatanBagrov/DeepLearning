from abc import abstractmethod

import numpy as np


class LossFunction:
    @abstractmethod
    def calculate(self, y_predicted, y_true):
        """
                :param y_predicted: the predicted label (probabilities, other methods)
                :param y_true: the true label (1-hot)
                :return: a TUPLE of (loss, d/dy_predicted of loss)
        """
        pass


class MSE(LossFunction):
    
    def calculate(self, y_predicted, y_true):
        n = y_predicted.size
        # TODO: should diff be calculated as y_predicted - y_true? that changes the dl_dy...
        diff = y_true - y_predicted
        loss = np.square(diff) / n
        dl_dy = (2/n) * diff
        return loss, dl_dy


class CrossEntropy(LossFunction):
    # TODO: implement this
    def calculate(self, y_predicted, y_true):
        pass


def test_mse():
    pass


def test_cross_entropy():
    pass


if __name__ == '__main__':
    test_mse()
    test_cross_entropy()