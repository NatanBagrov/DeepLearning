from graph.GraphNode import GraphNode


class Variable(GraphNode):
    def __init__(self, value):
        self._value = value
        self._gradient = 0
        self._gradient_sum = 0

    def forward(self):
        return self._value

    def set_value(self, value):
        self._value = value

    def backward(self, grad=None):
        if grad is None:
            self._gradient = 1
        else:
            self._gradient_sum += grad
            self._gradient += grad

    def reset(self):
        self._gradient = 0

    def update_grad(self, eta):
        self._value = self._value - 1 * eta * self._gradient_sum
        self._gradient_sum = 0

    def get_gradient(self):
        return self._gradient
