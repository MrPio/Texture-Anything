"""
LPIPS: Learned Perceptual Image Patch Similarity.
Measures perceptual difference between images using deep features.
Lower = more similar (0 = identical).
"""
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def compute_lpips(
    preds: torch.Tensor,
    targets: torch.Tensor,
    device: str = 'cuda'
) -> float:
    """
    Compute mean LPIPS between predictions and ground truth.
    
    Args:
        preds: (N, 3, H, W) in [0, 1]
        targets: (N, 3, H, W) in [0, 1]
        device: 'cuda' or 'cpu'
    
    Returns:
        LPIPS score (0 = identical, typical range: 0.0 - 1.0)
    """
    metric = LearnedPerceptualImagePatchSimilarity(
        net_type='alex',      # AlexNet backbone (fast)
        normalize=True        # Input in [0,1]
    ).to(device)
    
    preds = preds.to(device)
    targets = targets.to(device)
    
    score = metric(preds, targets)
    return score.item()
