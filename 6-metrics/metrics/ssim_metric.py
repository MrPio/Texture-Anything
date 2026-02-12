"""
SSIM: Structural Similarity Index Measure.
Compares structure, luminance, and contrast.
Higher = more similar (range [0, 1]).
"""
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure


def compute_ssim(
    preds: torch.Tensor,
    targets: torch.Tensor,
    device: str = 'cuda'
) -> float:
    """
    Compute mean SSIM between predictions and ground truth.
    
    Args:
        preds: (N, 3, H, W) in [0, 1]
        targets: (N, 3, H, W) in [0, 1]
        device: 'cuda' or 'cpu'
    
    Returns:
        SSIM score (0-1, higher = better)
    """
    metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    preds = preds.to(device)
    targets = targets.to(device)
    
    score = metric(preds, targets)
    return score.item()
