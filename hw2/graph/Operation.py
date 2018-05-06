from abc import abstractmethod

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
