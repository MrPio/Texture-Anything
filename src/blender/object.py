import PIL.Image
import bpy
import PIL
import numpy as np
from ..utils import plot_images
import bmesh
import bmesh
from PIL import Image, ImageDraw


def get_mesh_stats(mesh) -> dict:
    """Get the properties of a given mesh in the current scene.
    Properties are `uv_count`, `texture_count`"""
    assert mesh.type == "MESH"

    texture_count = 0
    for slot in mesh.material_slots:
        mat = slot.material
        if mat and mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE":
                    for output in node.outputs:
                        for link in output.links:
                            if link.to_socket.name == "Base Color":
                                texture_count += 1

    return {"uv_count": len(mesh.data.uv_layers), "texture_count": texture_count}


def get_diffuse_textures(mesh, plot: bool = False) -> list[PIL.Image.Image]:
    """Unpack all the diffuse texture images in the scene as PIL"""

    diffuse_images = set()
    for slot in mesh.material_slots:
        mat = slot.material
        if mat and mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image is not None:
                    for output in node.outputs:
                        for link in output.links:
                            if link.to_socket.name == "Base Color":
                                diffuse_images.add(node.image)

    embedded_images = [img for img in diffuse_images if img.packed_file is not None]
    images_pil = []

    if embedded_images:
        for img in embedded_images:
            pixels = (np.array(img.pixels) * 255).astype(np.uint8)
            pixels = pixels.reshape((*img.size, 4))
            image_pil = PIL.Image.fromarray(pixels, "RGBA")
            images_pil.append(image_pil)

    if plot and len(images_pil) > 0:
        plot_images(images_pil, cols=min(4, len(images_pil)))

    return images_pil


def draw_uv_map(mesh, size=1024, stroke=1) -> Image.Image:
    bm = bmesh.new()
    bm.from_mesh(mesh.data)
    uv_layer = bm.loops.layers.uv.active
    if not uv_layer:
        raise Exception("No UV layers found on the mesh")

    # === Create white transparent image ===
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    # === Draw UV edges ===
    for face in bm.faces:
        uv_coords = [loop[uv_layer].uv for loop in face.loops]
        if len(uv_coords) < 2:
            continue
        # Scale and convert UVs to pixel coordinates (flip V axis)
        points = [(int(uv.x * size), int(uv.y * size)) for uv in uv_coords]
        # Close the loop
        points.append(points[0])
        draw.line(points, fill=(0, 0, 0, 255), width=stroke)

    return img
