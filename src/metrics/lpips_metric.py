import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .base_metric import Metric


class LPIPSMetric(Metric):
    def __init__(self):
        super().__init__(need_renders=False)
        
    def compute(self, y, gt) -> float:
        metric = LearnedPerceptualImagePatchSimilarity(
            net_type="alex",
            normalize=True,
        ).to(self.device)
        score = metric(y.to(self.device), gt.to(self.device))
        return float(score.item())
