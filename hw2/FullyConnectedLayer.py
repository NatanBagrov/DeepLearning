import math

import numpy as np

from utils.ActivationFunctions import ActivationFunction
from utils.RegularizationMethods import RegularizationMethod
from graph.BinaryOperations import Add, Multiply
from graph.GraphNode import GraphNode
from graph.Variable import Variable
import logging

logger = logging.getLogger('FullyConnectedLayer')


class FullyConnectedLayer(GraphNode):
    def __init__(self, inputs_num: int, outputs_num: int,
                 activation_function: ActivationFunction.__class__,
                 input_variable=None):
        super().__init__()
        self._af = activation_function
        self._w = Variable(
            np.random.uniform(-1 / math.sqrt(inputs_num), 1 / math.sqrt(inputs_num), (inputs_num, outputs_num)))
        self._b = Variable(np.zeros(outputs_num))
        self._input = input_variable
        self._output = self._af(Add(Multiply(self._input, self._w), self._b))

    def forward(self):
        return self._output.forward()

    def backward(self, grads=None):
        self._output.backward(grads)

    def reset(self):
        self._output.reset()

    def get_value(self):
        return self._output.get_value()

    def update_grad(self, learning_rate):
        # param_scale = np.linalg.norm(self._w.get_value())
        # update_scale = np.linalg.norm(-learning_rate * self._w.get_gradient())
        # logger.info('Update magnitude is %f (desired is about %f)', update_scale / param_scale, 1e-3)

        self._w.update_grad(learning_rate)
        self._b.update_grad(learning_rate)

    def get_weight(self):
        return self._w