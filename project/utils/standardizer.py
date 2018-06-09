import numpy as np


class Standardizer:
    def __init__(self, mean=None, std=None):
        self._mean = mean
        self._std = std

    def __str__(self):
        return 'Standardizer(mean={}, std={})'.format(self._mean, self._std)

    def fit(self, x):
        self._mean = np.mean(x)
        self._std = np.std(x)

        assert np.isclose(self._std, 0.0), 'Input has 0 variance'

    def predict(self, x):
        assert self._mean is not None and self._std is not None, 'Please fit first'

        return (x - self._mean) / self._std

    def save(self, file_path):
        np.savez(file_path, {'mean': self._mean, 'std': self._std})

    def restore(self, file_path):
        loaded = np.load(file_path)
        self._mean = loaded['mean']
        self._std = loaded['std']

