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
            grad = 1.0

        self._gradient += grad
#        assert isinstance(self._value, float) and isinstance(grad, float) or self._value.shape == grad.shape

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
