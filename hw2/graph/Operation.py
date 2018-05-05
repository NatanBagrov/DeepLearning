from abc import abstractmethod

import numpy as np

from graph.GraphNode import GraphNode
from graph.Variable import Variable


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


class ReductionOperation(UnaryOperation):

    def __init__(self, node: GraphNode, axis: int):
        super().__init__(node)
        self._axis = axis

    @abstractmethod
    def _inner_backward(self, grad=None):
        pass

    def _inner_reset(self):
        self._node.reset()


class ReduceSum(ReductionOperation):
    def __init__(self, node: GraphNode, axis: int):
        super().__init__(node, axis)
        self._size = ReduceSize(node, axis)

    def forward(self):
        node_value = self._node.forward()
        self._value = np.sum(node_value, axis=self._axis) if isinstance(node_value, np.ndarray) else node_value
        return self._value

    def _inner_backward(self, grad=None):
        self._node.backward(self._gradient * np.ones(self._size.forward()))


class ReduceSize(ReductionOperation):

    def __init__(self, node: GraphNode, axis: int):
        super().__init__(node, axis)

    def forward(self):
        node_value = self._node.forward()
        self._value = node_value.shape[self._axis] if isinstance(node_value, np.ndarray) else 1
        return self._value

    def _inner_backward(self, grad=None):
        pass


class ReduceMean(ReductionOperation):
    # TODO: write few tests
    def __init__(self, node: GraphNode, axis: int):
        sum = ReduceSum(node, axis)
        size = ReduceSize(node, axis)
        one_div_size = Divide(Variable(1), size)
        res = Multiply(sum, one_div_size)  # TODO: BROKEN. float @ np.float not supported.
        super().__init__(res, axis)

    def forward(self):
        self._value = self._node.forward()
        return self._value

    def _inner_backward(self, grad=None):
        self._node.backward(self._gradient)


def test_reduce_sum():
    # Matrix
    x = np.array([[1, 2, 3], [11, 12, 13]])
    v = Variable(x)
    rs = ReduceSum(v, 1)
    np.testing.assert_allclose(rs.forward(), np.array([6, 36]), rtol=1e-5)
    rs2 = ReduceSum(v, 0)
    np.testing.assert_allclose(rs2.forward(), np.array([12, 14, 16]), rtol=1e-5)
    op_sum = ReduceSum(ReduceSum(v, 0), 0)
    np.testing.assert_allclose(op_sum.forward(), np.sum(x), rtol=1e-5)
    # Array
    y = np.array([-0.5, 1, 2.5])
    v2 = Variable(y)
    r = ReduceSum(v2, 0)
    np.testing.assert_allclose(r.forward(), 3.0, rtol=1e-5)
    r.backward(1)
    np.testing.assert_equal(v2.get_gradient(), [1, 1, 1])


def test_reduce_mean():
    # Array
    y = np.array([-0.5, 1, 2.5])
    v2 = Variable(y)
    m = ReduceMean(v2, 0)
    np.testing.assert_allclose(m.forward(), 1.0, rtol=1e-5)
    m.backward(1)
    np.testing.assert_equal(v2.get_gradient(), [1/3, 1/3, 1/3])


if __name__ == '__main__':
    test_reduce_sum()
    test_reduce_mean()
