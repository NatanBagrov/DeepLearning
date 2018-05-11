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
        raise NotImplementedError()

    @abstractmethod
    def forward(self):
        raise NotImplementedError()

    def _inner_reset(self):
        self._node.reset()


class Sigmoid(ActivationFunction):
    def sigma(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        self._value = self.sigma(self._node.forward())
        return self._value

    def _inner_backward(self, grad=None):
        grad_sig = self.sigma(self._node.get_value())
        self._node.backward(self._gradient * (grad_sig * (1.0 - grad_sig)))


class ReLU(ActivationFunction):

    def forward(self):
        self._value = np.maximum(self._node.forward(), 0)
        return self._value

    def _inner_backward(self, grad=None):
        self._node.backward(self._gradient * ((self.get_value() > 0).astype(float)))  # int also ok, maybe better?


class Softmax(ActivationFunction):

    def forward(self):
        x = self._node.forward()
        # numerically stable softmax
        e_x = np.exp((x.T - np.max(x, axis=1)).T)
        self._value = (e_x.T / e_x.sum(axis=1)).T
        return self._value

    def _inner_backward(self, grad=None):
        weighted_value = self._gradient * self.get_value()
        gradient_by_node = weighted_value - np.transpose(np.transpose(self.get_value()) * np.sum(weighted_value, axis=1))
        self._node.backward(gradient_by_node)


class Identity(ActivationFunction):

    def forward(self):
        self._value = self._node.forward()

        return self._value

    def _inner_backward(self, grad=None):
        self._node.backward(self._gradient)


activation_function_name_to_class = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'sot-max': Softmax,  # TODO: is it typo in document?
    'none': Identity,
}


def test_sigmoid():
    x = np.array([1, 1.1, 1.2, 5.5])
    v = Variable(x)
    sig = Sigmoid(v)
    np.testing.assert_allclose(sig.forward(), [0.73105858, 0.75026011, 0.76852478, 0.99592986], rtol=1e-5)
    sig.backward(np.ones(4))
    np.testing.assert_allclose(v.get_gradient(), [0.19661193, 0.18736987, 0.17789444, 0.00405357], rtol=1e-5)


def test_relu():
    x = np.random.random(10) - 0.5
    x[1] = -2  # in case all positive
    v = Variable(x)
    relu = ReLU(v)
    expected = np.array(x)
    expected[x < 0] = 0
    np.testing.assert_allclose(relu.forward(), expected, rtol=1e-5)
    relu.backward(np.ones(10))
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
