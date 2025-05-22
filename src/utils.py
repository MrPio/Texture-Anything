import math
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def plot_images(images: list[Image.Image | str | Path] | dict[Image.Image | str | Path], size=3, cols: int = None):
    """Plot a list of PIL images in a grid

    Args:
        images (list[Image.Image]): the list of images to show
        size (int, optional): the size in inch of the images
        col (int, optional): The number of columns of the grid. Defaults to 1.
    """
    titles = None
    if isinstance(images, dict):
        titles, images = list(images.keys()), list(images.values())
    for i in range(len(images)):
        if not isinstance(images[i], (Image.Image, np.ndarray)):
            images[i] = Image.open(images[i])

    if not cols:
        cols = min(10, len(images))
    rows = math.ceil(len(images) / cols)
    max_ratio = max(
        (image.size[0] / image.size[1] if isinstance(image, (Image.Image)) else image.shape[0] / image.shape[1])
        for image in images
    )
    _, axes = plt.subplots(rows, cols, figsize=(cols * size, int(rows * size / max_ratio)))
    if rows > 1 or cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    for i, img in enumerate(images):
        axes[i].imshow(img)
        if titles:
            axes[i].set_title(titles[i])
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


def compute_image_density(img: Image.Image, threshold=0) -> float:
    """Get the fraction of non-transparent pixels of an image. Can be used to determine how dense a UV map is."""
    # Ensure image is in RGBA mode
    img = img.convert("RGBA")
    pixels = img.getdata()
    non_transparent = sum(1 for px in pixels if px[3] > threshold)

    return non_transparent / len(pixels)


def is_textured(mesh):
    """Returns True if the material uses an image texture (not a flat color)"""

    for mat_slot in mesh.material_slots:
        mat = mat_slot.material
        if not mat.use_nodes:
            continue

        principled = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if not principled:
            continue

        base_color_input = principled.inputs.get("Base Color")
        if base_color_input and base_color_input.is_linked:
            from_node = base_color_input.links[0].from_node
            if from_node.type == "TEX_IMAGE":
                return True

    return False


def binary_mask_tensor(image_path: str) -> torch.Tensor:
    """
    Load a 1-bit black-and-white image and convert it to a PyTorch mask tensor.

    Each pixel in the input image:
      - Black  (pixel value == 0)   → mask value = 1
      - White  (pixel value == 255) → mask value = 0

    Returns:
      A torch.Tensor of shape (1, H, W), dtype=torch.uint8, with values {0,1}.
    """
    # Open image and convert to 1-bit BW: pixels are either 0 (black) or 255 (white)
    pil_mask = Image.open(image_path).convert("1")

    # Create a NumPy array from the PIL image: shape (H, W), values in {0,255}
    arr = np.array(pil_mask)

    # Build a binary array: black→1, white→0
    binary = (arr == 0).astype(np.uint8)

    # Convert to a PyTorch tensor and add channel dimension
    mask_tensor = torch.from_numpy(binary)
    mask_tensor = mask_tensor.unsqueeze(0)  # shape → (1, H, W)

    return mask_tensor
