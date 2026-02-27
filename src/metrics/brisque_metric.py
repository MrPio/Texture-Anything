import numpy as np
import brisque
import torch

from .base_metric import Metric


class BRISQUEMetric(Metric):
    def __init__(self, need_renders=True):
        super().__init__(need_renders=need_renders)

    def compute(self, y, gt, captions) -> float:
        y = y.reshape(-1, *y.shape[2:]) if self.need_renders else y
        brisque_model = brisque.BRISQUE()
        y_uint8 = (y.cpu() * 255).permute(0, 2, 3, 1).numpy().astype("uint8")
        scores = [brisque_model.score(y_uint8[i]) for i in range(y_uint8.shape[0])]
        return float(np.mean(scores))
