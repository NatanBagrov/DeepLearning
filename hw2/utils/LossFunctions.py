from abc import abstractmethod

import numpy as np

from graph.GraphNode import GraphNode
from graph.Operation import Operation


class LossFunction(Operation):

    def __init__(self, n1: GraphNode, n2: GraphNode):
        super().__init__()
        self._node1 = n1
        self._node2 = n2

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def _inner_backward(self, grad=None):
        pass

    def _inner_reset(self):
        self._node1.reset()
        self._node2.reset()


class MSE(LossFunction):

    def __init__(self, n1: GraphNode, n2: GraphNode):
        super().__init__(n1, n2)
        self._size = None

    def forward(self):
        v1 = self._node1.forward()  # serves as y_hat
        v2 = self._node2.forward()  # serves as y_truth
        assert v1.shape == v2.shape
        self._size = v1.size
        self._value = np.sum((self._node1.forward() - self._node2.forward()) ** 2) / self._size
        return self._value

    def _inner_backward(self, grad=None):
        # TODO: is this the right implementation? should we store 2 values from forward? should we add a getter?
        n1_value = self._node1.forward()
        n2_value = self._node2.forward()
        # TODO: should we divide by size or shouldnt?
        self._node1.backward((2/self._size) * self._gradient * n1_value)
        self._node2.backward(-(2/self._size) * self._gradient * n2_value)


class CrossEntropy(LossFunction):

    def forward(self):
        pass

    def _inner_backward(self, grad=None):
        pass


def test_mse():
    assert False  # TODO


def test_cross_entropy():
    assert False  # TODO


if __name__ == '__main__':
    test_mse()
    test_cross_entropy()
