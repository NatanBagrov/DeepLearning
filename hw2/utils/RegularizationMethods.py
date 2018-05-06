from abc import abstractmethod

import numpy as np

from graph.GraphNode import GraphNode
from graph.Operation import Operation


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
        self._value = np.sum(self._node.forward() ** 2)
        return self._value

    def _inner_backward(self, grad=None):
        self._node.backward(self._gradient * 2 * self._node.get_value())


class L1(RegularizationMethod):

    def forward(self):
        self._value = np.sum(np.abs(self._node.forward()))
        return self._value

    def _inner_backward(self, grad=None):
        self._node.backward(self._gradient * np.sign(self._node.get_value()))


regularization_method_name_to_class = {
    'l1': L1,
    'l2': L2,
}
