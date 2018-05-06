from abc import abstractmethod
from random import random

import numpy as np

from graph.GraphNode import GraphNode
from graph.Operation import Divide, HadamardMult, UnaryOperation
from graph.Variable import Variable


class Transpose(UnaryOperation):

    def __init__(self, node: GraphNode):
        super().__init__(node)

    def forward(self):
        node_value = self._node.forward()
        self._value = node_value.T if isinstance(node_value, np.ndarray) else node_value
        return self._value

    def _inner_backward(self, grad=None):
        node_value = self._node.forward()
        if isinstance(node_value, np.ndarray):
            self._node.backward(self._gradient.T)
        else:
            self._node.backward(self._gradient)


class ReductionOperation(UnaryOperation):

    def __init__(self, node: GraphNode, axis: int = None):
        super().__init__(node)
        self._axis = axis

    @abstractmethod
    def _inner_backward(self, grad=None):
        pass

    def _inner_reset(self):
        self._node.reset()


class ReduceSize(ReductionOperation):

    def __init__(self, node: GraphNode, axis=None):
        super().__init__(node, axis)

    def forward(self):
        node_value = self._node.forward()
        if not isinstance(node_value, np.ndarray):
            self._value = 1
            return self._value
        self._value = node_value.size if self._axis is None else node_value.shape[self._axis]
        return self._value

    def _inner_backward(self, grad=None):
        pass


class ReduceSum(ReductionOperation):
    def __init__(self, node: GraphNode, axis):
        super().__init__(node, axis)
        self._size = ReduceSize(node, axis)

    def forward(self):
        node_value = self._node.forward()
        self._value = np.sum(node_value, axis=self._axis) if isinstance(node_value, np.ndarray) else node_value
        return self._value

    def _inner_backward(self, grad=None):
        self._node.backward(self._gradient * np.ones(self._size.forward()))


class ReduceMean(ReductionOperation):
    def __init__(self, node: GraphNode, axis=None):
        sum = ReduceSum(node, axis)
        size = ReduceSize(node, axis)
        one_div_size = Divide(Variable(1), size)
        res = HadamardMult(sum, one_div_size)
        super().__init__(res, axis)

    def forward(self):
        self._value = self._node.forward()
        return self._value

    def _inner_backward(self, grad=None):
        self._node.backward(self._gradient)


def test_transpose():
    x = np.random.rand(5, 3)
    v = Variable(x)
    t = Transpose(v)
    np.testing.assert_allclose(t.forward(), x.T)
    grads = np.random.rand(3, 5)
    t.backward(grads)
    np.testing.assert_allclose(v.get_gradient(), grads.T)


def test_reduce_size():
    x = np.random.rand(5, 3)
    v = Variable(x)
    rs_full = ReduceSize(v)
    rs_rows = ReduceSize(v, 0)
    rs_cols = ReduceSize(v, 1)
    np.testing.assert_equal(rs_full.forward(), 15)
    np.testing.assert_equal(rs_rows.forward(), 5)
    np.testing.assert_equal(rs_cols.forward(), 3)
    grad_before = v.get_gradient()
    rs_full.backward(np.random.rand(5, 3))
    np.testing.assert_equal(v.get_gradient(), grad_before)


def test_reduce_sum():
    x = np.random.rand(5, 3)
    v1, v2 = Variable(x), Variable(x)
    rs_rows = ReduceSum(v1, 0)
    rs_cols = ReduceSum(v2, 1)
    np.testing.assert_allclose(rs_rows.forward(), np.sum(x, 0))
    np.testing.assert_allclose(rs_cols.forward(), np.sum(x, 1))
    grad = random()
    rs_rows.backward(grad)
    np.testing.assert_allclose(v1.get_gradient(), grad * np.ones((5,)))
    rs_cols.backward(grad)
    np.testing.assert_allclose(v2.get_gradient(), grad * np.ones((3,)))




if __name__ == '__main__':
    test_transpose()
    test_reduce_size()
    test_reduce_sum()
    # test_reduce_mean()
