from abc import abstractmethod

import numpy as np

from graph.GraphNode import GraphNode
from graph.Operation import Operation
from graph.Variable import Variable


class RegularizationMethod(Operation):

    def __init__(self, node: GraphNode):
        super().__init__()
        self._node = node

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def _inner_backward(self, grad=None):
        pass

    def _inner_reset(self):
        self._node.reset()


class L2(RegularizationMethod):

    def forward(self):
        self._value = np.linalg.norm(self._node.forward()) ** 2
        return self._value

    def _inner_backward(self, grad=None):
        self._node.backward(2 * self._gradient)


class L1(RegularizationMethod):

    def forward(self):
        self._value = np.linalg.norm(self._node.forward(), 1)
        return self._value

    def _inner_backward(self, grad=None):
        # TODO: passing 0 for zero entry in the grad is ok?
        self._node.backward(np.sign(self._gradient))


def test_l1():
    x = np.random.rand(10) - 0.5
    v = Variable(x)
    l1 = L1(v)
    np.testing.assert_allclose(l1.forward(), np.sum(np.abs(x)), rtol=1e-5)
    l1.backward(x)
    np.testing.assert_equal(v.get_gradient(), np.sign(x))


def test_l2():
    x = np.random.rand(10) - 0.5
    v = Variable(x)
    l2 = L2(v)
    np.testing.assert_allclose(l2.forward(), np.sum(np.abs(x) ** 2), rtol=1e-5)
    l2.backward(x)
    np.testing.assert_allclose(v.get_gradient(), 2 * x, rtol=1e-5)


if __name__ == '__main__':
    test_l1()
    test_l2()
