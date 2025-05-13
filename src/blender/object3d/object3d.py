import abc
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
    def uv_score(self, weights=(0.3, 0.25, 0.25, 0.2)):
        weights = (0.3, 0.25, 0.25, 0.2)
        assert self.has_one_mesh
        """
        Scores the UV layout of a Blender mesh object for readability and cleanliness.

        Args:
            weights (tuple): Weights for (continuity, distortion, packing, fragmentation).

        Returns:
            float: The combined score S between 0 and 1.
            dict: Individual component scores {"C", "D", "P", "F"}.
        """

        mesh = self.mesh.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        # Select UV layer
        uv_layer = bm.loops.layers.uv.active
        if uv_layer is None:
            raise ValueError("No UV layer found")

        # 1. Continuity: count cut edges (edges whose two adjacent faces map to different islands)
        total_edges = len(bm.edges)
        # Identify UV islands via flood fill
        island_idxs = {}
        current_island = 0
        tagged = {}

        def flood_fill(face, idx):
            stack = [face]
            while stack:
                f = stack.pop()
                for loop in f.loops:
                    tagged[loop[uv_layer]] = True
                island_idxs[f.index] = idx
                for edge in f.edges:
                    linked = [
                        lf.face for lf in edge.link_loops if loop[uv_layer] not in tagged or not tagged[loop[uv_layer]]
                    ]
                    for lf_face in linked:
                        if lf_face.index not in island_idxs:
                            stack.append(lf_face)

        for face in bm.faces:
            if face.index not in island_idxs:
                # clear tags
                # for l in bm.loops:
                #     l[uv_layer].tag = False
                flood_fill(face, current_island)
                current_island += 1

        # Now count cut edges: edges whose adjacent faces belong to different islands
        cut_edges = 0
        for edge in bm.edges:
            faces = edge.link_faces
            if len(faces) == 2:
                if island_idxs[faces[0].index] != island_idxs[faces[1].index]:
                    cut_edges += 1
        C = 1.0 - (cut_edges / total_edges) if total_edges > 0 else 0.0

        # 2. Distortion: compare UV area vs 3D area per face
        stretches = []
        for face in bm.faces:
            # 3D area from bmesh
            area3d = face.calc_area()
            # UV area: project poly onto UV plane
            uv_coords = [loop[uv_layer].uv for loop in face.loops]
            area_uv = 0.0
            for i in range(len(uv_coords)):
                x1, y1 = uv_coords[i]
                x2, y2 = uv_coords[(i + 1) % len(uv_coords)]
                area_uv += x1 * y2 - x2 * y1
            area_uv = abs(area_uv) * 0.5
            stretches.append(area_uv / area3d if area3d > 0 else 0.0)
        sigma_s = float(np.std(stretches))
        D = 1.0 - (1.0 / (1.0 + sigma_s))

        # 3. Packing efficiency
        total_uv_area = sum((stretch * face.calc_area()) for stretch, face in zip(stretches, bm.faces))
        canvas_area = 1.0  # UV space is normalized to [0,1]
        P = total_uv_area / canvas_area

        # 4. Fragmentation: entropy of island area distribution
        island_areas = {}
        # accumulate UV area per island
        for face, stretch in zip(bm.faces, stretches):
            idx = island_idxs[face.index]
            island_areas[idx] = island_areas.get(idx, 0.0) + (stretch * face.calc_area())
        total_area = sum(island_areas.values())
        probs = [a / total_area for a in island_areas.values() if total_area > 0]
        H = -sum(p * math.log(max(1e-3, p)) for p in probs) if probs else 0.0
        F = 1.0 - (H / math.log(len(probs))) if len(probs) > 1 else 1.0

        # Combine scores
        wC, wD, wP, wF = weights
        S = (wC * C) + (wD * D) + (wP * P) + (wF * F)

        # Clean up bmesh
        bm.free()

        return S, {"C": C, "D": D, "P": P, "F": F}
