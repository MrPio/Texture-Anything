import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from .base_metric import Metric


class FIDMetric(Metric):
    def __init__(self):
        super().__init__(need_renders=False)
        
    def compute(self, y, gt) -> float:
        metric = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)
        gt_uint8 = (gt * 255).clamp(0, 255).to(torch.uint8).to(self.device)
        y_uint8 = (y * 255).clamp(0, 255).to(torch.uint8).to(self.device)
        metric.update(gt_uint8, real=True)
        metric.update(y_uint8, real=False)
        return float(metric.compute().item())
