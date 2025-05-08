import math
from PIL import Image
import matplotlib.pyplot as plt


def plot_images(images: list[Image.Image], size=4, cols: int = 1):
    """Plot a list of PIL images in a grid

    Args:
        images (list[Image.Image]): the list of images to show
        size (int, optional): the size in inch of the images
        col (int, optional): The number of columns of the grid. Defaults to 1.
    """
    rows = math.ceil(len(images) / cols)
    _, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size))
    if rows > 1 or cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].set_title(f"({img.size[0]}×{img.size[0]})")
        axes[i].axis("off")
    plt.show()


def compute_opacity(img: Image.Image, threshold=0) -> float:
    """Get the fraction of non-transparent pixels of an image. Can be used to determine how dense a UV map is."""
    # Ensure image is in RGBA mode
    img = img.convert("RGBA")
    pixels = img.getdata()
    non_transparent = sum(1 for px in pixels if px[3] > threshold)

    return non_transparent / len(pixels)
