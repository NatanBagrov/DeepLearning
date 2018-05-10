from abc import abstractmethod


class GraphNode:
    def __init__(self):
        self._value = None

    def get_value(self):
        return self._value

    @abstractmethod
    def forward(self):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, grads=None):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()
