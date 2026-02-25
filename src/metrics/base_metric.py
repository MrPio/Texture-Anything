from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Metric(ABC):
    """Base class for all evaluation metrics."""

    def __init__(self, need_renders: bool = False, device: str | None = None) -> None:
        self.need_renders = need_renders
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def compute(self, y: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute and return a scalar metric value."""

    def __call__(self, *args, **kwargs) -> float:
        return self.compute(*args, **kwargs)
