"""
FID: Frechet Inception Distance.
Compares distributions of real vs generated images.
Lower = more similar distributions.
"""
import torch
from torchmetrics.image.fid import FrechetInceptionDistance


def compute_fid(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: str = 'cuda'
) -> float:
    """
    Compute FID between two image distributions.
    
    Args:
        real_images: (N, 3, H, W) in [0, 1] - ground truth
        fake_images: (M, 3, H, W) in [0, 1] - generated
        device: 'cuda' or 'cpu'
    
    Returns:
        FID score (lower = better, 0 = identical distributions)
    """
    metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    # Convert to uint8 [0, 255]
    real_uint8 = (real_images * 255).clamp(0, 255).to(torch.uint8).to(device)
    fake_uint8 = (fake_images * 255).clamp(0, 255).to(torch.uint8).to(device)
    
    # Accumulate distributions
    metric.update(real_uint8, real=True)
    metric.update(fake_uint8, real=False)
    
    score = metric.compute()
    return score.item()
