"""
BRISQUE: Blind/Referenceless Image Spatial Quality Evaluator.
Evaluates distortions (blur, noise, compression) without ground truth.
Uses 3D renderings of objects with applied textures.
Lower = better quality (range 0-100).
"""
import torch
import numpy as np
import brisque

def compute_brisque(
    rendered_images: torch.Tensor
) -> float:
    """
    Evaluate quality of 3D renderings using BRISQUE.
    
    Args:
        rendered_images: (N, 3, H, W) in [0, 1] - renderings with texture
    
    Returns:
        Mean BRISQUE score (0-100, lower = better)
    """
    
    brisque_model = brisque.BRISQUE()
    
    # Convert to numpy uint8
    if rendered_images.is_cuda:
        rendered_images = rendered_images.cpu()
    
    images_np = (rendered_images * 255).numpy().astype('uint8')
    
    scores = []
    for i in range(images_np.shape[0]):
        # BRISQUE expects (H, W, C)
        img = images_np[i].transpose(1, 2, 0)
        score = brisque_model.score(img)
        scores.append(score)
    
    return np.mean(scores)
