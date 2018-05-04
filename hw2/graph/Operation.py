from abc import abstractmethod

from graph.GraphNode import GraphNode


class Operation(GraphNode):
    def __init__(self):
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

    def reset(self):
        self._gradient = 0
