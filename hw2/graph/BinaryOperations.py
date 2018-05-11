from abc import abstractmethod

import numpy as np

from graph.GraphNode import GraphNode
from graph.Operation import Operation


class BinaryOperation(Operation):
    def __init__(self, left, right):
        super().__init__()

        assert isinstance(left, GraphNode)
        assert isinstance(right, GraphNode)

        self._left = left
        self._right = right

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def _inner_backward(self, grad=None):
        # This should implement the backward specific logic
        pass

    def _inner_reset(self):
        self._left.reset()
        self._right.reset()

    def _do_backward(self, d_current_d_left, d_current_d_right):
        # TODO: i think it is bad idea to unite them
        self._left.backward(self._gradient * d_current_d_left)
        self._right.backward(self._gradient * d_current_d_right)


class Add(BinaryOperation):
    def __init__(self, left, right):
        super().__init__(left, right)

    def forward(self):
        # TODO: BEWARE OF bad broadcasting
        self._value = self._left.forward() + self._right.forward()

        return self._value

    def _inner_backward(self, grad=None):
        # TODO: assuming broadcasting here, expect bugs
        d_current_d_left = 1.0
        d_current_d_right = 1.0

        self._do_backward(d_current_d_left, d_current_d_right)

    def _do_backward(self, d_current_d_left, d_current_d_right):
        def get_dimensions_number(x):
            return 1 if isinstance(x, float) else len(x.shape)

        extra_dimensions_left = get_dimensions_number(self.get_value()) - get_dimensions_number(self._left.get_value())
        extra_dimensions_right = get_dimensions_number(self.get_value()) - get_dimensions_number(self._right.get_value())

        self._left.backward(np.sum(self._gradient, axis=tuple(range(extra_dimensions_left))) * d_current_d_left)
        self._right.backward(np.sum(self._gradient, axis=tuple(range(extra_dimensions_right))) * d_current_d_right)


class Multiply(BinaryOperation):
    def __init__(self, left: GraphNode, right: GraphNode):
        super().__init__(left, right)

    def forward(self):
        left = self._left.forward()
        right = self._right.forward()
        if isinstance(left, float):
            assert isinstance(right, float)

            self._value = left * right
        else:
            self._value = left @ right

        return self._value

    def _inner_backward(self, grad=None):
        d_current_d_left = self._right.get_value()
        d_current_d_right = self._left.get_value()

        self._do_backward(d_current_d_left, d_current_d_right)

    def _do_backward(self, d_current_d_left, d_current_d_right):
        if isinstance(d_current_d_left, float):
            self._left.backward(self._gradient * d_current_d_left)
        else:
            self._left.backward(self._gradient @ np.transpose(d_current_d_left))

        if isinstance(d_current_d_right, float):
            self._right.backward(self._gradient * d_current_d_right)
        else:
            self._right.backward(np.transpose(d_current_d_right) @ self._gradient)


class HadamardMult(BinaryOperation):
    def __init__(self, left: GraphNode, right: GraphNode):
        super().__init__(left, right)

    def forward(self):
        self._value = self._left.forward() * self._right.forward()

        return self._value

    def _inner_backward(self, grad=None):
        d_current_d_left = self._right.get_value()
        d_current_d_right = self._left.get_value()

        self._do_backward(d_current_d_left, d_current_d_right)


class Divide(BinaryOperation):

    def __init__(self, numerator, denominator):
        super().__init__(numerator, denominator)

    def forward(self):
        self._value = self._left.forward() / self._right.forward()
        return self._value

    def _inner_backward(self, grad=None):
        nom = self._left.get_value()
        denom = self._right.get_value()
        d_current_d_nom = 1.0 / denom
        d_current_d_denom = - (nom / (denom ** 2))
        self._do_backward(d_current_d_nom, d_current_d_denom)