"""
PSNR: Peak Signal-to-Noise Ratio.
Measures pixel-wise noise/distortion.
Higher = less noise (typical range: 20-50 dB).
"""
import torch
from torchmetrics.image import PeakSignalNoiseRatio


def compute_psnr(
    preds: torch.Tensor,
    targets: torch.Tensor,
    device: str = 'cuda'
) -> float:
    """
    Compute mean PSNR between predictions and ground truth.
    
    Args:
        preds: (N, 3, H, W) in [0, 1]
        targets: (N, 3, H, W) in [0, 1]
        device: 'cuda' or 'cpu'
    
    Returns:
        PSNR score in dB (higher = better)
    """
    metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    
    preds = preds.to(device)
    targets = targets.to(device)
    
    score = metric(preds, targets)
    return score.item()
