import torch
from torchmetrics.multimodal import CLIPImageQualityAssessment

from .base_metric import Metric


class CLIPIQAMetric(Metric):
    def __init__(self):
        super().__init__(need_renders=True)

    def compute(self, y, gt) -> float:
        captions = [("a good texture for a 3D model", "a bad texture for a 3D model")] * y.size(0)

        metric = CLIPImageQualityAssessment(prompts=tuple(captions)).to(self.device)
        images_255 = (y * 255).clamp(0, 255).to(self.device)
        scores = metric(images_255)
        diagonal_scores = [values[idx].item() for idx, values in enumerate(scores.values())]
        return float(sum(diagonal_scores) / len(diagonal_scores))
