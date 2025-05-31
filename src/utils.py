import math
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import bpy


def imshow(images: list[Image.Image | str | Path] | dict[Image.Image | str | Path], size=3, cols: int = None):
    """Plot a list of PIL images in a grid

    Args:
        images (list[Image.Image]): the list of images to show
        size (int, optional): the size in inch of the images
        col (int, optional): The number of columns of the grid. Defaults to 1.
    """
    images=list(images)
    if not images:
        return
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
    """Compute the fraction of non-transparent pixels of an image. Can be used to determine how dense a UV map is."""
    pixels = img.convert("RGBA").getdata()
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


def is_white(obj):
    """Has the `obj` a default white material? Used to prune irrelevant objects in shapenetcore"""
    for slot in obj.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes:
            continue

        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                color = node.inputs["Base Color"].default_value
                # Check if RGB is exactly white (ignore alpha)
                if tuple(color[:3]) == (1.0, 1.0, 1.0):
                    return True
    return False


def bpy2pil(img: bpy.types.Image, remove_black=False) -> Image.Image:
    buffer = np.empty(len(img.pixels), dtype=np.float32)
    img.pixels.foreach_get(buffer)  # MUCH faster than list(img.pixels) or np.array(img.pixels)

    pixels = (buffer * 255).astype(np.uint8)
    pixels = pixels.reshape((img.size[1], img.size[0], 4))  # PIL expects (height, width, channels)

    if remove_black:
        mask = (pixels[:, :, :3] == 0).all(axis=2)
        pixels[mask, 3] = 0

    return Image.fromarray(pixels, "RGBA")


flatten = lambda l: [x for el in list(l) for x in el]


def is_outside_uv(vector, threshold=0.15):
    for coord in [vector.x, vector.y]:
        if not -threshold < coord < 1 + threshold:
            return True
    return False
