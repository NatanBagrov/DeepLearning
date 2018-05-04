from abc import abstractmethod


class GraphNode:
    def __init__(self):
        self._value = None

    def get_value(self):
        return self._value

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self, grads=None):
        pass

    @abstractmethod
    def reset(self):
        pass

