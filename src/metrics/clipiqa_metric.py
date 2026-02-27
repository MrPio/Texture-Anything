import torch
from torchmetrics.multimodal import CLIPImageQualityAssessment
from tqdm import trange

from .base_metric import Metric


class CLIPIQAMetric(Metric):
    def __init__(self, need_renders=False):
        super().__init__(need_renders=need_renders)

    def compute(self, y, gt, captions) -> float:
        captions = [
            ("a good texture for a 3D model", "a bad texture for a 3D model")
        ] * y.size(1)
        metric = CLIPImageQualityAssessment(prompts=tuple(captions)).to(self.device)
        y_255 = (y * 255).clamp(0, 255).to(self.device)
        scores = []
        if self.need_renders:
            for i in trange(y.size(0), leave=False, desc="Computing CLIPIQA"):
                score = metric(y_255[i])
                diagonal_score = [
                    values[idx].item() for idx, values in enumerate(score.values())
                ]
                scores.append(float(sum(diagonal_score) / len(diagonal_score)))
        else:
            score = metric(y_255)
            diagonal_score = [
                values[idx].item() for idx, values in enumerate(score.values())
            ]
            scores.append(float(sum(diagonal_score) / len(diagonal_score)))

        return sum(scores) / len(scores)
