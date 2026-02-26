import torch
from torchmetrics.multimodal import CLIPImageQualityAssessment
from tqdm import trange

from .base_metric import Metric


class CLIPIQAMetric(Metric):
    def __init__(self):
        super().__init__(need_renders=True)

    def compute(self, y, gt, captions) -> float:
        scores = []
        for i in trange(y.size(0), leave=False, desc="Computing CLIPIQA"):
            captions = [("a good texture for a 3D model", "a bad texture for a 3D model")] * y.size(1)
            metric = CLIPImageQualityAssessment(prompts=tuple(captions)).to(self.device)
            images_255 = (y[i] * 255).clamp(0, 255).to(self.device)
            score = metric(images_255)
            diagonal_score = [values[idx].item() for idx, values in enumerate(score.values())]
            scores.append(float(sum(diagonal_score) / len(diagonal_score)))
        return sum(scores) / len(scores)
