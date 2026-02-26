import torch
import torch.nn.functional as F
from tqdm import trange
from transformers import CLIPProcessor, CLIPModel
from .base_metric import Metric


class CLIPMetric(Metric):
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        super().__init__(need_renders=True)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def compute(self, y, gt, captions) -> float:
        scores = []
        for i in trange(y.size(0), leave=False, desc="Computing CLIP"):
            inputs = self.processor(
                text=captions[i] * y.size(1),
                images=list(y[i]),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
                do_rescale=False,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                image_embeds = F.normalize(outputs.image_embeds, dim=-1)
                text_embeds = F.normalize(outputs.text_embeds, dim=-1)
                similarity = (image_embeds * text_embeds).sum(dim=-1)
                scores.append(similarity.mean().item())

        return sum(scores) / len(scores)
