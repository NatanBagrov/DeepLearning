from abc import abstractmethod

import numpy as np

from graph.GraphNode import GraphNode
from graph.Operation import Operation, Add, Multiply, ReduceMean, HadamardMult
from graph.Variable import Variable


class LossFunction(Operation):

    def __init__(self, n1: GraphNode, n2: GraphNode):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def _inner_backward(self, grad=None):
        pass

    @abstractmethod
    def _inner_reset(self):
        pass


class MSE(LossFunction):

    def __init__(self, label: GraphNode, predicted: GraphNode):
        super().__init__(label, predicted)
        self._label = label
        self._predicted = predicted

    def forward(self):
        self._value = np.mean(np.square(np.subtract(self._label.forward(), self._predicted.forward())))

        return self._value

    def _inner_backward(self, grad=None):
        if grad is None:
            grad = 1.0

        dl_dlabel = 2.0 * np.subtract(self._label.forward(), self._predicted.forward()) / self._label._value.size
        dl_dpredicted = -dl_dlabel

        self._label.backward(dl_dlabel)
        self._predicted.backward(dl_dpredicted)

    def _inner_reset(self):
        pass


class CrossEntropy(LossFunction):
    def __init__(self, label: GraphNode, predicted: GraphNode):
        super(CrossEntropy, self).__init__(label, predicted)
        Log()

    def forward(self):
        pass

    def _inner_backward(self, grad=None):
        pass


loss_name_to_class = {
    'MSE': MSE,
    'cross-entropy': CrossEntropy
}


def test_mse():
    y = np.zeros(10)
    y[2] = 1  # digit 2
    y_hat = np.random.dirichlet(np.ones(10), size=1)
    vy, vyh = Variable(y), Variable(y_hat)
    mse = MSE(vy, vyh)
    np.testing.assert_allclose(mse.forward(), (1/y.shape[0]) * np.linalg.norm((y_hat-y)) ** 2)


def test_cross_entropy():
    assert False  # TODO


if __name__ == '__main__':
    test_mse()
    # test_cross_entropy()
