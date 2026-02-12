"""
Utility functions for loading images from folders.
"""
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Tuple


def load_images_from_folder(
    folder_path: str,
    size: Tuple[int, int] = (512, 512),
    normalize: bool = True
) -> Tuple[torch.Tensor, List[str]]:
    """
    Load all images from a folder into a batch tensor.
    
    Args:
        folder_path: Path to folder containing images
        size: (height, width) for resizing
        normalize: If True, output in [0,1], else [0,255]
    
    Returns:
        tensor: (N, 3, H, W) batch of images
        filenames: List of image filenames (for debugging)
    """
    folder = Path(folder_path)
    image_files = sorted(folder.glob('*.png')) + sorted(folder.glob('*.jpg'))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {folder_path}")
    
    transform_list = [
        transforms.Resize(size),
        transforms.ToTensor()  # Converts to [0, 1]
    ]
    
    if not normalize:
        # Convert to [0, 255]
        transform_list.append(transforms.Lambda(lambda x: x * 255))
    
    transform = transforms.Compose(transform_list)
    
    images = []
    filenames = []
    
    for img_path in image_files:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        images.append(img_tensor)
        filenames.append(img_path.name)
    
    return torch.stack(images), filenames


def match_image_pairs(
    folder_pred: str,
    folder_gt: str,
    size: Tuple[int, int] = (512, 512)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load paired images by matching filenames from two folders.
    
    Args:
        folder_pred: Folder with predicted images
        folder_gt: Folder with ground truth images
        size: Resize dimensions
    
    Returns:
        preds: (N, 3, H, W) predictions
        targets: (N, 3, H, W) ground truths
    """
    preds, pred_names = load_images_from_folder(folder_pred, size, normalize=True)
    targets, gt_names = load_images_from_folder(folder_gt, size, normalize=True)
    
    # Check that filenames match
    assert len(pred_names) == len(gt_names), \
        f"Different number of images: {len(pred_names)} preds vs {len(gt_names)} GT"
    
    for p, g in zip(pred_names, gt_names):
        if p != g:
            raise ValueError(f"Files don't match: {p} vs {g}")
    
    return preds, targets
