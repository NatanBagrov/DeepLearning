import math

import numpy as np

from utils.ActivationFunctions import ActivationFunction
from utils.RegularizationMethods import RegularizationMethod
from graph.Operation import Add, Multiply
from graph.GraphNode import GraphNode
from graph.Variable import Variable


class FullyConnectedLayer(GraphNode):
    def __init__(self, inputs_num: int, outputs_num: int, activation_function: ActivationFunction.__class__,
                 regularization_method: RegularizationMethod, weight_decay: float):
        self._af = activation_function
        self._lambda = weight_decay
        self._rm = regularization_method
        self._w = Variable(
            np.random.uniform(-1 / math.sqrt(inputs_num), 1 / math.sqrt(inputs_num), (inputs_num, outputs_num)))
        self._b = Variable(np.zeros(outputs_num))
        self._input = Variable(None)  # TODO: placeholder if you want
        # TODO: add regularization once its API is clear
        self._output = self._af(Add(Multiply(self._w, self._input), self._b))

    def forward(self):
        return self._output.forward()

    def backward(self, grads=None):
        self._output.backward()

    def reset(self):
        self._output.reset()
