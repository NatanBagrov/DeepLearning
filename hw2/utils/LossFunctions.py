from abc import abstractmethod

import numpy as np

from graph.GraphNode import GraphNode
from graph.Operation import Operation, Add, Multiply, HadamardMult, Divide, Subtract
from graph.UnaryOperations import Transpose, ReduceSize
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
        diff = Subtract(predicted, label)
        squared_error = Multiply(diff, Transpose(diff))
        self._size_node = ReduceSize(diff, axis=0)
        mse = Divide(squared_error, self._size_node)
        self._node = mse

    def forward(self):
        self._value = self._node.forward()
        return self._value

    def _inner_backward(self, grad=None):
        self._node.backward(self._gradient)

    def _inner_reset(self):
        self._node.reset()
        self._size_node.reset()


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


def test_mse_basic():
    y = np.zeros(10)
    y_hat = np.ones(10)
    # MSE = 10 * 1 / 10 = 1
    vy, vyh = Variable(y), Variable(y_hat)
    mse = MSE(vy, vyh)
    np.testing.assert_allclose(mse.forward(), 1, rtol=1e-5)
    mse.backward(1)
    #dL/dyhat = dL/dMSE * dMSE/dyhat = 1 * 2/10 * 1 = 0.2
    np.testing.assert_allclose(vyh.get_gradient(), 2/10 * (y_hat-y))

def test_cross_entropy():
    assert False  # TODO


if __name__ == '__main__':
    test_mse_basic()
    # test_cross_entropy()
