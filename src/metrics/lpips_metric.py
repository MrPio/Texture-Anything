import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .base_metric import Metric


class LPIPSMetric(Metric):
    def __init__(self, need_renders=False):
        super().__init__(need_renders=need_renders)

    def compute(self, y, gt, captions) -> float:
        metric = LearnedPerceptualImagePatchSimilarity(
            net_type="alex",
            normalize=True,
        ).to(self.device)
        if self.need_renders:
            score = sum(
                metric(y[:, i].to(self.device), gt[:, i].to(self.device))
                for i in range(y.size(1))
            ) / y.size(1)
        else:
            score = metric(y.to(self.device), gt.to(self.device))

        return float(score.item())
