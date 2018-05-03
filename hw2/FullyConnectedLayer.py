import math

import numpy as np

from hw2.utils.ActivationFunctions import ActivationFunction
from hw2.utils.RegularizationMethods import RegularizationMethod


class FullyConnectedLayer:
    def __init__(self, inputs_num: int, outputs_num: int, activation_function: ActivationFunction,
                 regularization_method: RegularizationMethod, weight_decay: float):
        self._af = activation_function
        self._lambda = weight_decay
        self._rm = regularization_method
        self._w = \
            np.random.uniform(-1 / math.sqrt(inputs_num), 1 / math.sqrt(inputs_num), (inputs_num, outputs_num))
        self._b = np.zeros(outputs_num)
        self._output = np.zeros(outputs_num)

    def feed_forward(self, inputs):
        self._output = self._af.forward(np.dot(inputs.T, self._w))
        return self._output

    def feed_back(self, output_from_prev, grads_from_next, step_size):
        self._b += step_size * grads_from_next
        # self._w += TODO: continue here...
