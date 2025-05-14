import abc
from functools import cached_property
from pathlib import Path
import numpy as np
from ..scene import load_model
from ...utils import plot_images
import bmesh
import bpy
from PIL import Image, ImageDraw
import bmesh
import math
import numpy as np
from mathutils import Vector

DATASET_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data/dataset"


class Object3D(abc.ABC):
    """Represents a 3d object. Methods are meant to be called on objects containing 1 mesh, 1 UV and 1 diffuse texture.

    Args:
        uid: the uid provided by the dataset the model is coming from
        path: the absolute path of the file
    """

    def __init__(self, uid: str, path: str | Path):
        self.uid = uid
        self.path = Path(path) if path is str else path

        self.objects = load_model(str(self.path), reset_scene=True)
        self.meshes = [x for x in self.objects if x.type == "MESH"]
        self.has_one_mesh = len(self.meshes) == 1
        self.mesh = self.meshes[0]

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

    @cached_property
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

    @property
    @abc.abstractmethod
    def render(self) -> list[Image.Image]: ...

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

    def regenerate_uv_map(self, island_margin=0.03) -> tuple[Image.Image, Image.Image]:
        """Regenerate a new UV map and Bake the diffuse texture accordingly.

        Returns:
            tuple[Image.Image, Image.Image]: The new texture and the drawing of the new UV map.
        """
        new_uv_name = "SmartUV"
        bake_image_name = "BakedTexture"
        bake_image_size = 1024

        assert self.has_one_mesh

        # Switch to Object mode
        bpy.ops.object.mode_set(mode="OBJECT")
        mesh = self.mesh.data

        # 1. Duplicate the existing UV map
        uv_layer = mesh.uv_layers.active
        mesh.uv_layers.new(name=new_uv_name)
        mesh.uv_layers.active = mesh.uv_layers[new_uv_name]

        # 2. Smart UV unwrap
        bpy.context.view_layer.objects.active = self.mesh
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=66, island_margin=island_margin)
        bpy.ops.object.mode_set(mode="OBJECT")

        # 3. Create new image to bake into
        image = bpy.data.images.new(bake_image_name, width=bake_image_size, height=bake_image_size)

        # 4. Assign image to active UV map
        if not mesh.uv_layers.active.data:
            raise RuntimeError("UV map has no data.")

        # Create image texture node and assign image
        mat = self.mesh.active_material
        nodes = mat.node_tree.nodes
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.image = image
        mat.node_tree.nodes.active = tex_node  # Mark active for baking

        # 5. Bake the texture to the new image
        # bpy.context.view_layer.objects.active = self.mesh
        bpy.context.scene.render.engine = "CYCLES"
        bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"}, use_clear=True)

        # 6. Save baked image (optional)
        pixels = (np.array(image.pixels) * 255).astype(np.uint8)
        pixels = pixels.reshape((*image.size, 4))
        image_pil = Image.fromarray(pixels, "RGBA")
        return image_pil, self.draw_uv_map()

    @property
    def uv_score(self) -> float:
        """
        Estimate how well the active UV map of a mesh object preserves 3D face areas.

        Returns a float in [0,1], where 1 means exact area‐preservation (UV areas
        perfectly proportional to 3D areas), and 0 means no correlation.

        Method:
        1. For each polygon, get its 3D area (mesh.polygons[].area) and UV area
            (via 2D shoelace on its UV coordinates).
        2. Build two distributions: A3d_i / sum(A3d) and Auv_i / sum(Auv).
        3. The L1 distance between these two distributions is in [0,2]. We map
            that to [1,0] by doing similarity = 1 - (L1 / 2).

        Requirements:
        - The mesh must have exactly one UV map (active).
        - Faces may be n-gons; their UV area is computed in 2D by shoelace.

        Raises:
        ValueError if obj is not a mesh or has no UV map.
        """
        assert self.has_one_mesh
        mesh = self.mesh.data
        if not mesh.uv_layers:
            raise ValueError("Mesh has no UV layers")
        uv_data = mesh.uv_layers.active.data

        # Collect 3D and UV face areas
        areas_3d = []
        areas_uv = []

        for poly in mesh.polygons:
            # 3D area built in
            a3 = poly.area
            areas_3d.append(a3)

            # Gather UV coords for this face
            uv_coords = [uv_data[li].uv.copy() for li in poly.loop_indices]
            # Compute UV area by 2D shoelace formula
            area2d = 0.0
            n = len(uv_coords)
            for i in range(n):
                x0, y0 = uv_coords[i]
                x1, y1 = uv_coords[(i + 1) % n]
                area2d += (x0 * y1) - (x1 * y0)
            areas_uv.append(abs(area2d) * 0.5)

        # Normalize to distributions
        total3 = sum(areas_3d)
        totaluv = sum(areas_uv)
        if total3 == 0 or totaluv == 0:
            return 0.0

        dist3 = [a / total3 for a in areas_3d]
        distuv = [a / totaluv for a in areas_uv]

        # L1 distance between distributions
        l1 = sum(abs(d3 - du) for d3, du in zip(dist3, distuv))
        # Map [0,2] → [1,0]
        return max(0.0, 1.0 - (l1 / 2.0))
