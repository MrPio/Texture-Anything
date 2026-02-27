import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from .base_metric import Metric


class FIDMetric(Metric):
    def __init__(self, need_renders=False):
        super().__init__(need_renders=need_renders)

    def compute(self, y, gt, captions) -> float:
        metric = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)
        gt_uint8 = (gt * 255).clamp(0, 255).to(torch.uint8).to(self.device)
        y_uint8 = (y * 255).clamp(0, 255).to(torch.uint8).to(self.device)
        if self.need_renders:
            scores = []
            for i in range(y.size(1)):
                metric.update(gt_uint8[:, i], real=True)
                metric.update(y_uint8[:, i], real=False)
                scores.append(float(metric.compute().item()))
            score = sum(scores) / len(scores)
        else:
            metric.update(gt_uint8, real=True)
            metric.update(y_uint8, real=False)
            score = float(metric.compute().item())

        return score
