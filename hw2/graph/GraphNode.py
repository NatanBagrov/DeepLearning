from abc import abstractmethod


class GraphNode:
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self, grads=None):
        pass

    @abstractmethod
    def reset(self):
        pass

