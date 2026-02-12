"""
CLIP-IQA: CLIP-based Image Quality Assessment.
Evaluates how well rendered textures match their generation captions.
Uses 3D renderings of objects with applied textures.
Higher = better prompt fidelity.
"""
import torch
from torchmetrics.multimodal import CLIPImageQualityAssessment


def compute_clipiqa(
    rendered_images: torch.Tensor,
    captions: list,
    device: str = 'cuda'
) -> float:
    """
    Evaluate how well 3D renderings match their generation captions.
    
    Args:
        rendered_images: (N, 3, H, W) in [0, 1] - renderings with texture
        captions: List of N captions used to generate each texture
        device: 'cuda' or 'cpu'
    
    Returns:
        Mean prompt fidelity score (0-1, higher = better match)
        
    Example:
        captions = [
            "a wooden rustic texture with worn details",
            "metallic brushed steel surface"
        ]
        score = compute_clipiqa(renders, captions)
        # → 0.87
    """
    assert len(captions) == rendered_images.shape[0], \
        f"Number of captions ({len(captions)}) must match images ({rendered_images.shape[0]})"
    
    metric = CLIPImageQualityAssessment(prompts=tuple(captions)).to(device)
    
    # CLIP-IQA expects [0, 255]
    images_255 = (rendered_images * 255).clamp(0, 255).to(device)
    
    scores = metric(images_255)
    
    # Extract diagonal: each image vs its own caption
    per_image_scores = []
    for idx, (key, val) in enumerate(scores.items()):
        # val is shape (N,) - get score for idx-th image vs idx-th caption
        per_image_scores.append(val[idx].item())
    
    # Return mean score
    return sum(per_image_scores) / len(per_image_scores)
