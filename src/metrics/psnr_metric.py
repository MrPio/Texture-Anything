import torch
from torchmetrics.image import PeakSignalNoiseRatio

from .base_metric import Metric


class PSNRMetric(Metric):
    def __init__(self, need_renders=False):
        super().__init__(need_renders=need_renders)

    def compute(self, y, gt, captions) -> float:
        metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        score = metric(y.to(self.device), gt.to(self.device))
        return float(score.item())
