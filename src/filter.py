from abc import ABC
from functools import cached_property
from scipy.signal import convolve2d
import numpy as np


class Filter(ABC):
    def __init__(self, *filters: np.ndarray, threshold):
        self.filters = filters
        self.threshold = threshold

    def __call__(self, array) -> np.ndarray:
        convolved = [convolve2d(array, filter, mode="same", boundary="symm") ** 2 for filter in self.filters]
        convolved = np.sqrt(np.sum(convolved, axis=(0)))
        return np.mean(convolved), convolved

    @cached_property
    def is_homogeneous(self) -> bool:
        self()[0] < self.threshold


class SobelFilter(Filter):
    def __init__(self):
        super().__init__(
            np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
            threshold=50,
        )


class PrewittFilter(Filter):
    def __init__(self):
        super().__init__(
            np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),
            np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),
            threshold=35,
        )


class LaplacianFilter(Filter):
    def __init__(self):
        super().__init__(
            np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
            threshold=20,
        )
