from abc import abstractmethod

import numpy as np

from graph.GraphNode import GraphNode
from graph.Operation import Operation


class LossFunction(Operation):

    def __init__(self, n1: GraphNode, n2: GraphNode):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def _inner_backward(self, grad=None):
        pass

    @abstractmethod
    def _inner_reset(self):
        pass


class MSE(LossFunction):

    def __init__(self, label: GraphNode, predicted: GraphNode):
        super().__init__(label, predicted)
        self._label = label
        self._predicted = predicted

    def forward(self):
        self._value = np.mean(np.square(np.subtract(self._label.forward(), self._predicted.forward())))

        return self._value

    def _inner_backward(self, grad=None):
        dl_dlabel = 2.0 * np.subtract(self._label.get_value(), self._predicted.get_value()) / self._label.get_value().size
        dl_dpredicted = -dl_dlabel
        # multiplied by self._gradient just to be on the safe side and not to assume we are the last node.
        self._label.backward(self._gradient * dl_dlabel)
        self._predicted.backward(self._gradient * dl_dpredicted)

    def _inner_reset(self):
        pass


class CrossEntropy(LossFunction):
    def __init__(self, label: GraphNode, predicted: GraphNode):
        super(CrossEntropy, self).__init__(label, predicted)
        Log()

    def forward(self):
        pass

    def _inner_backward(self, grad=None):
        pass


loss_name_to_class = {
    'MSE': MSE,
    'cross-entropy': CrossEntropy
}

