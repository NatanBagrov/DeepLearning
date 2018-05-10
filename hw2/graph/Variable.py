from graph.GraphNode import GraphNode
import numpy as np


class Variable(GraphNode):
    def __init__(self, value):
        super().__init__()
        self._value = value
        self._gradient = 0
        self._gradient_sum = 0

    def forward(self):
        return self._value

    def set_value(self, value):
        self._value = value

    def backward(self, grad=None):
        if grad is None:
            self._gradient = 1.0
        else:
            self._gradient_sum += grad
            self._gradient += grad

    def reset(self):
        self._gradient = 0

    def update_grad(self, eta):
        self._value = self._value - 1.0 * eta * self._gradient_sum
        self._gradient_sum = 0

    def get_gradient(self):
        # assert isinstance(self._value, float) and or  isinstance(self._gradient, float) or \
        #        self._value.shape == self._gradient.shape

        return self._gradient
