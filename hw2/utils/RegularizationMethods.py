from abc import abstractmethod

import numpy as np


class RegularizationMethod:
    @abstractmethod
    def value(self, x):
        pass


class L2(RegularizationMethod):
    def value(self, x):
        return 2 * x


class L1(RegularizationMethod):
    def value(self, x):
        # TODO: what to do with zeros? is it ok to return 0?
        return np.sign(x)
