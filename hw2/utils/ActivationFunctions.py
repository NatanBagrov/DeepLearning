import math
from abc import abstractmethod

import numpy as np


class ActivationFunction:

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, x):
        pass


class Sigmoid(ActivationFunction):

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, y):
        return y * (1.0 - y)


class ReLU(ActivationFunction):

    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, y):
        return (y > 0).astype(float)  # int also ok, maybe better?


class Softmax(ActivationFunction):

    def forward(self, x):
        # numerically stable softmax
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def backward(self, x):  # TODO: derivative of x (np.array) returns a martix, shouldnt it return an array?
        s = x.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)


class Identity(ActivationFunction):

    def forward(self, x):
        return x

    def backward(self, x):
        return np.ones(x.shape)


def test_sigmoid():
    sig = Sigmoid()
    x = np.array([1, 1.1, 1.2, 5.5])
    np.testing.assert_allclose(sig.forward(x), [0.73105858, 0.75026011, 0.76852478, 0.99592986], rtol=1e-5)
    assert sig.backward(0.5) == 0.25


def test_relu():
    relu = ReLU()
    x = np.random.random(10) - 0.5
    x[1] = -2  # in case all positive
    expected = np.array(x)
    expected[x < 0] = 0
    np.testing.assert_allclose(relu.forward(x), expected, rtol=1e-5)
    np.testing.assert_equal(relu.backward(x), np.sign(expected))


def test_softmax():
    sm = Softmax()
    x = np.array([3, -1, 1])
    expected = np.array([0.86681333, 0.01587623, 0.11731042])
    np.testing.assert_allclose(sm.forward(x), expected, rtol=1e-5)


def test_identity():
    none = Identity()
    x = np.random.rand(10)
    np.testing.assert_allclose(none.forward(x), x, rtol=1e-5)
    np.testing.assert_equal(none.backward(x), np.ones(10))
    # TODO: add a test for the derivative, once the implementation is approved to be correct


if __name__ == '__main__':
    test_sigmoid()
    test_relu()
    test_softmax()
    test_identity()
