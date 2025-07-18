from abc import ABC
from functools import cached_property
from scipy.signal import convolve2d
import numpy as np
from PIL import Image

from src.dataset.objaverse_dataset3d import ObjaverseDataset3D
from src.dataset.shapenetcore_dataset3d import ShapeNetCoreDataset3D


class Filter(ABC):
    def __init__(self, *filters: np.ndarray, threshold):
        self.filters = filters
        self.threshold = threshold

    def __call__(self, image) -> np.ndarray:
        if issubclass(type(image), Image.Image):
            image = np.array(image.convert("L"), dtype=np.float32)
        convolved = [convolve2d(image, filter, mode="same", boundary="symm") ** 2 for filter in self.filters]
        convolved = np.sqrt(np.sum(convolved, axis=(0)))
        return np.mean(convolved), convolved

    def is_jagged(self, image, type) -> bool:
        self(image)[0] > self.threshold[type]

    def is_plain(self, image) -> bool:
        self(image)[0] < 0.5


class SobelFilter(Filter):
    def __init__(self):
        super().__init__(
            np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
            threshold={ObjaverseDataset3D: 1e6, ShapeNetCoreDataset3D: 52},
        )


class PrewittFilter(Filter):
    def __init__(self):
        super().__init__(
            np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),
            np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),
            threshold={ObjaverseDataset3D: 1e6, ShapeNetCoreDataset3D: 38},
        )


class LaplacianFilter(Filter):
    def __init__(self):
        super().__init__(
            np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
            threshold={ObjaverseDataset3D: 1e6, ShapeNetCoreDataset3D: 23},
        )
