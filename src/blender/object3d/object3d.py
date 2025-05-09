import abc
from pathlib import Path
import numpy as np
from ..scene import load_model
from ...utils import plot_images
import bmesh
from PIL import Image, ImageDraw


class Object3D(abc.ABC):
    def __init__(self, uid: str, path: str):
        self.uid = uid
        self.path = Path(path)

        self.objects = load_model(path)
        meshes = [x for x in self.objects if x.type == "MESH"]
        self.has_one_mesh = len(meshes) == 1
        self.mesh = meshes[0]

    @property
    def _mesh_nodes(self):
        assert self.has_one_mesh
        nodes = []
        for slot in self.mesh.material_slots:
            mat = slot.material
            if mat and mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == "TEX_IMAGE":
                        for output in node.outputs:
                            for link in output.links:
                                if link.to_socket.name == "Base Color":
                                    nodes.append(node)
        return nodes

    @property
    def mesh_stats(self) -> dict:
        """Get the properties of a given mesh in the current scene.
        Properties are `uv_count`, `texture_count`"""

        assert self.has_one_mesh
        return {
            "uv_count": len(self.mesh.data.uv_layers),
            "texture_count": len(self._mesh_nodes),
            "face_count": len(self.mesh.data.polygons),
        }

    @property
    @abc.abstractmethod
    def textures(self) -> list[Image.Image]: ...

    def plot_diffuse(self):
        images_pil = self.textures
        plot_images(images_pil, cols=min(4, len(images_pil)))

    def draw_uv_map(self, size=1024, stroke=1) -> Image.Image:
        assert self.has_one_mesh

        bm = bmesh.new()
        bm.from_mesh(self.mesh.data)
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