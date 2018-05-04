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


class BinaryOperation(Operation):
    def __init__(self, left, right):
        super(BinaryOperation, self).__init__()

        self._left = left
        self._right = right

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def _inner_backward(self, grad=None):
        # This should implement the backward specific logic
        pass

    def _inner_reset(self):
        self._left.reset()
        self._right.reset()

    def _do_backward(self, d_current_d_left, d_current_d_right):
        self._left.backward(self._gradient * d_current_d_left)
        self._right.backward(self._gradient * d_current_d_right)


class Add(BinaryOperation):
    def __init__(self, left, right):
        super(Add, self).__init__(left, right)

    def forward(self):
        self._value = self._left.forward() + self._right.forward()

        return self._value

    def _inner_backward(self, grad=None):
        d_current_d_left = 1
        d_current_d_right = 1

        self._do_backward(d_current_d_left, d_current_d_right)


class Multiply(BinaryOperation):
    def __init__(self, left, right):
        super(Multiply, self).__init__(left, right)

    def forward(self):
        self._value = self._left.forward() * self._right.forward()

        return self._value

    def _inner_backward(self, grad=None):
        d_current_d_left = self._right.value
        d_current_d_right = self._left.value

        self._do_backward(d_current_d_left, d_current_d_right)
