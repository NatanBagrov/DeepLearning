from abc import abstractmethod

import numpy as np

from graph.GraphNode import GraphNode


class Operation(GraphNode):
    def __init__(self):
        super().__init__()
        self._value = 0
        self._gradient = 0

    @abstractmethod
    def forward(self):
        pass

    def backward(self, grad=None):
        if grad is None:
            self._gradient = 1
        else:
            self._gradient += grad

        # Specific logic to be implemented in the concrete classes, 'grad' passed for safety (might be needed?)
        self._inner_backward(grad)

    def reset(self):
        self._gradient = 0
        self._inner_reset()

    @abstractmethod
    def _inner_backward(self, grad=None):
        # This should implement the backward specific logic
        pass

    @abstractmethod
    def _inner_reset(self):
        # This should implement the reset specific logic
        pass


class UnaryOperation(Operation):

    def __init__(self, node: GraphNode):
        super().__init__()
        assert isinstance(node, GraphNode)
        self._node = node

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def _inner_backward(self, grad=None):
        pass

    def _inner_reset(self):
        self._node.reset()


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
        self._left.backward(self._gradient * d_current_d_left)
        self._right.backward(self._gradient * d_current_d_right)


class Add(BinaryOperation):
    def __init__(self, left, right):
        super().__init__(left, right)

    def forward(self):
        self._value = self._left.forward() + self._right.forward()

        return self._value

    def _inner_backward(self, grad=None):
        d_current_d_left = 1
        d_current_d_right = 1

        self._do_backward(d_current_d_left, d_current_d_right)


class Multiply(BinaryOperation):
    def __init__(self, left: GraphNode, right: GraphNode):
        super().__init__(left, right)

    def forward(self):
        self._value = self._left.forward() @ self._right.forward()

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
        d_current_d_nom = 1 / denom
        d_current_d_denom = - (nom / (denom ** 2))
        self._do_backward(d_current_d_nom, d_current_d_denom)


class RowCount(UnaryOperation):
    def forward(self):
        node_value = self._node.forward()
        self._value = node_value.shape[0] if isinstance(node_value, np.ndarray) else 1
        return self._value

    def _inner_backward(self, grad=None):
        pass  # Does nothing


class SumOverRows(UnaryOperation):
    # TODO: how to sum? should sum over rows? cols? both? maybe create SumOverAxisFactory?
    def forward(self):
        node_value = self._node.forward()
        self._value = np.sum(node_value, axis=0) if isinstance(node_value, np.ndarray) else node_value
        return self._value

    def _inner_backward(self, grad=None):
        # TODO: implement with respect to the upper TODO.
        self._node.backward(self._gradient)
