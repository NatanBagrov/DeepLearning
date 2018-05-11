from abc import abstractmethod
from random import random

import numpy as np

from graph.GraphNode import GraphNode
from graph.Operation import Operation
from graph.BinaryOperations import HadamardMult, Divide
from graph.Variable import Variable


class UnaryOperation(Operation):

    def __init__(self, node: GraphNode):
        super().__init__()
        assert isinstance(node, GraphNode)
        self._node = node

    @abstractmethod
    def forward(self):
        raise NotImplementedError()

    @abstractmethod
    def _inner_backward(self, grad=None):
        raise NotImplementedError()

    def _inner_reset(self):
        self._node.reset()


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
        raise NotImplementedError()

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
        needed_shape = self._node.get_value().shape
        new_shape = [1 if self._axis is None or self._axis == dimension else value for dimension, value in enumerate(needed_shape)]
        reps = [value if self._axis is None or self._axis == dimension else 1 for dimension, value in enumerate(needed_shape)]
        d_current_d_node = np.tile(np.reshape(self._gradient, new_shape), reps)
        assert d_current_d_node.shape == needed_shape

        self._node.backward(d_current_d_node)


class ReduceMean(ReductionOperation):
    def __init__(self, node: GraphNode, axis=None):
        sum = ReduceSum(node, axis)
        size = ReduceSize(node, axis)
        one_div_size = Divide(Variable(1.0), size)
        res = HadamardMult(sum, one_div_size)
        super().__init__(res, axis)

    def forward(self):
        self._value = self._node.forward()
        return self._value

    def _inner_backward(self, grad=None):
        self._node.backward(self._gradient)


class Splitter(UnaryOperation):

    def __init__(self, node: GraphNode, split_size):
        super().__init__(node)
        self._split_size = split_size
        self._curr_grad_count, self._curr_reset_count = 0, 0

    def forward(self):
        self._value = self._node.forward()

        return self._value

    def _inner_backward(self, grad=None):
        self._curr_grad_count += 1
        if self._curr_grad_count == self._split_size:
            self._node.backward(self._gradient)
            self._curr_grad_count = 0

    def _inner_reset(self):
        self._curr_reset_count += 1
        if self._curr_reset_count == self._split_size:
            self._node.reset()
            self._curr_reset_count = 0


class Log(UnaryOperation):
    def __init__(self, node: GraphNode):
        super().__init__(node)

    def forward(self):
        self._value = np.log(self._node.forward())

        return self._value

    def _inner_backward(self, grad=None):
        self._node.backward((1.0 / self._node.get_value()) * grad)
