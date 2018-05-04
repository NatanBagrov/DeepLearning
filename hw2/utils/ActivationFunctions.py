from abc import abstractmethod

import numpy as np

from graph.GraphNode import GraphNode
from graph.Operation import Operation
from graph.Variable import Variable


class ActivationFunction(Operation):

    def __init__(self, node: GraphNode):
        super().__init__()
        self._node = node

    @abstractmethod
    def _inner_backward(self, grad=None):
        pass

    @abstractmethod
    def forward(self):
        pass

    def _inner_reset(self):
        self._node.reset()


class Sigmoid(ActivationFunction):
    def sigma(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        self._value = self.sigma(self._node.forward())
        return self._value

    def _inner_backward(self, grad=None):
        grad_sig = self.sigma(self._gradient)
        self._node.backward(grad_sig * (1.0 - grad_sig))


class ReLU(ActivationFunction):

    def forward(self):
        self._value = np.maximum(self._node.forward(), 0)
        return self._value

    def _inner_backward(self, grad=None):
        self._node.backward((self._gradient > 0).astype(float))  # int also ok, maybe better?


class Softmax(ActivationFunction):

    def forward(self):
        x = self._node.forward()
        # numerically stable softmax
        e_x = np.exp(x - np.max(x))
        self._value = e_x / e_x.sum()
        return self._value

    def _inner_backward(self, grad=None):
        # TODO: derivative of x (np.array) returns a martix, shouldnt it return an array?
        s = self._gradient.reshape(-1, 1)
        self._node.backward(np.diagflat(s) - np.dot(s, s.T))


class Identity(ActivationFunction):

    def forward(self):
        return self._node.forward()

    def _inner_backward(self, grad=None):
        self._node.backward(self._gradient)


def test_sigmoid():
    x = np.array([1, 1.1, 1.2, 5.5])
    v = Variable(x)
    sig = Sigmoid(v)
    np.testing.assert_allclose(sig.forward(), [0.73105858, 0.75026011, 0.76852478, 0.99592986], rtol=1e-5)
    sig.backward(x)
    np.testing.assert_allclose(v.get_gradient(), [0.19661193, 0.18736987, 0.17789444, 0.00405357], rtol=1e-5)


def test_relu():
    x = np.random.random(10) - 0.5
    x[1] = -2  # in case all positive
    v = Variable(x)
    relu = ReLU(v)
    expected = np.array(x)
    expected[x < 0] = 0
    np.testing.assert_allclose(relu.forward(), expected, rtol=1e-5)
    relu.backward(x)
    np.testing.assert_equal(v.get_gradient(), np.sign(expected))


def test_softmax():
    x = np.array([3, -1, 1])
    v = Variable(x)
    sm = Softmax(v)
    expected = np.array([0.86681333, 0.01587623, 0.11731042])
    np.testing.assert_allclose(sm.forward(), expected, rtol=1e-5)
    # TODO: add a test for the derivative, once the implementation is approved to be correct


def test_identity():
    x = np.random.rand(10)
    v = Variable(x)
    none = Identity(v)
    np.testing.assert_allclose(none.forward(), x, rtol=1e-5)
    none.backward(x)
    np.testing.assert_equal(v.get_gradient(), x)


if __name__ == '__main__':
    test_sigmoid()
    test_relu()
    test_softmax()
    test_identity()
