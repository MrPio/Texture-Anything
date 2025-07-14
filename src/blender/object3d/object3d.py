import abc
import contextlib
from functools import cached_property
import io
import os
from pathlib import Path
import tempfile
from typing import Literal, Optional

from tqdm import tqdm
import numpy as np
from ..scene import load_hdri, load_model
from ...utils import imshow, is_outside_uv
import bpy
from PIL import Image, ImageDraw
import bmesh
import math
import numpy as np
from mathutils import Vector


class Object3D(abc.ABC):
    """Represents a 3d object. Methods are meant to be called on objects containing 1 mesh, 1 UV and 1 diffuse texture. Since the Blender scene is singleton, you should instantiate one Object3D at the time.

    Args:
        uid: the uid provided by the dataset the model is coming from
        path: the absolute path of the file
    """

    HDRI_PATH = Path(__file__).resolve().parents[1] / "hdri" / "colorful_studio_4k.exr"
    HDRI_PATH_WHITE = Path(__file__).resolve().parents[1] / "hdri" / "white.exr"

    def __init__(self, uid: str, path: str | Path, preprocess=True):
        self.uid = uid
        self.path = Path(path)

        load_model(str(self.path), reset_scene=True)

        # Remove non mesh objects
        self.objects = []
        for obj in bpy.data.objects:
            if obj.type != "MESH":
                bpy.data.objects.remove(obj, do_unlink=True)
            else:
                self.objects.append(obj)

        # This increases the loading time almost 10 times
        if preprocess:
            # Normalize the size so that the max dimension is 1m
            self.normalize_position()
            self.normalize_scale()

    @cached_property
    def _mesh_nodes(self, object):
        nodes = []
        for slot in object.material_slots:
            mat = slot.material
            if mat and mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == "TEX_IMAGE":
                        for output in node.outputs:
                            for link in output.links:
                                if link.to_socket.name == "Base Color":
                                    nodes.append(node)
        return nodes

    def normalize_scale(self):
        sizes = []
        for obj in bpy.data.objects:
            # obj.bound_box returns a list of 8 corner axis-aligned points
            bbox = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
            max_size = max((max(c[i] for c in bbox) - min(c[i] for c in bbox)) for i in range(3))
            sizes.append(max_size)

        if max(sizes) > 0:
            for obj in bpy.data.objects:
                bpy.context.view_layer.objects.active = obj
                obj.scale /= max(sizes)
                bpy.ops.object.transform_apply(scale=True)

    def normalize_position(self):
        objs = [obj for obj in bpy.data.objects if obj.type == "MESH"]
        # Compute the scene centroid
        total_loc = Vector((0, 0, 0))
        for obj in objs:
            total_loc += obj.location
        center = total_loc / len(objs)

        # Set origin to geometry bounds for each object, then move all relative to center
        for obj in objs:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
            obj.location -= center

    @cached_property
    def mesh_stats(self, object) -> dict:
        """Get the properties of a given mesh in the current scene.
        Properties are `uv_count`, `texture_count`"""
        return {
            "uv_count": len(object.data.uv_layers),
            "texture_count": len(self._mesh_nodes(object)),
            "face_count": len(object.data.polygons),
        }

    @property
    @abc.abstractmethod
    def textures(self) -> list[Image.Image]: ...

    @property
    @abc.abstractmethod
    def renderings(self) -> list[Image.Image]: ...

    def uv_score(self, mesh, uv_layer=None) -> float | None:
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
        uv_layer = mesh.uv_layers[uv_layer] if uv_layer else mesh.uv_layers.active
        if uv_layer is None:
            return None

        # Collect 3D and UV face areas
        uv_data = uv_layer.data
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

    def plot_diffuse(self):
        images_pil = self.textures
        imshow(images_pil, cols=min(4, len(images_pil)))

    def draw_uv(self, mesh, uv_layer=None, size=512, stroke=1, fill=False, verbose=True) -> Image.Image | None:
        """Draw the UV map of the object.

        Args:
            size (int, optional): The size of the generate image. Defaults to 1024.
            stroke (int, optional): The width of the edges stroke. Defaults to 1.
            fill: Wheter to fill the non mapped zones with black. Defaults to `False`.

        Returns:
            Image.Image: The drawing of the UV map
        """
        bm = bmesh.new()
        bm.from_mesh(mesh)
        uv_layer = bm.loops.layers.uv[uv_layer] if uv_layer else bm.loops.layers.uv.active
        if not uv_layer:
            raise Exception("No UV layers found on the mesh")

        # === Create white transparent image ===
        img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        # === Draw UV edges ===
        for face in bm.faces:
            uv_coords = [loop[uv_layer].uv for loop in face.loops]
            if any(map(is_outside_uv, uv_coords)):
                if verbose:
                    print("The UV map has negative values")
                return None
            if len(uv_coords) < 2:
                continue
            # Scale and convert UVs to pixel coordinates (flip V axis)
            points = [(int(uv.x * size), int(uv.y * size)) for uv in uv_coords]
            # Close the loop
            points.append(points[0])
            if fill:
                draw.polygon(points, fill=(0, 0, 0, 255), width=stroke)
            else:
                draw.line(points, fill=(0, 0, 0, 255), width=stroke)

        return img

    def regenerate_uv_map(
        self,
        object,
        island_margin=0,
        size=512,
        samples=8,
        bake_type: Literal["DIFFUSE", "GLOSSY"] = "DIFFUSE",
        load_lights=True,
        light_strength=1,
        device="CPU",
    ) -> tuple[Image.Image, Image.Image]:
        """Regenerate a new UV map and Bake the diffuse texture accordingly.

        Returns:
            tuple[Image.Image, Image.Image]: The new texture and the drawing of the new UV map.
        """
        # Switch to Object mode
        mesh = object.data
        object.select_set(True)
        bpy.context.view_layer.objects.active = object

        # 1. Duplicate the existing UV map
        mesh.uv_layers.new(name="SmartUV")
        mesh.uv_layers.active = mesh.uv_layers["SmartUV"]

        # 2. Smart UV unwrap
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(island_margin=island_margin)
        bpy.ops.object.mode_set(mode="OBJECT")

        # 3. Create new image to bake into
        img = bpy.data.images.new("BakedTexture", size, size)
        # mat = object.active_material
        # nodes = mat.node_tree.nodes

        # # Create and activate the new texture node
        # tex_node = nodes.new("ShaderNodeTexImage")
        # tex_node.image = img
        # # Make sure it's the active one for baking
        # mat.node_tree.nodes.active = tex_node

        # 4. For every material slot on the mesh, add an Image Texture node
        #    that points to our new image and mark it active in that material.
        for slot in object.material_slots:
            mat = slot.material
            if not mat or mat.use_nodes is False:
                continue

            # make sure this material is using nodes
            nodes = mat.node_tree.nodes

            # create a new image‐texture node and point it at our bake target
            tex_node = nodes.new("ShaderNodeTexImage")
            tex_node.image = img
            tex_node.select = True

            # ensure it’s the active image node for baking
            mat.node_tree.nodes.active = tex_node

        # 5. Bake the texture to the new image
        if load_lights:
            load_hdri(Object3D.HDRI_PATH_WHITE, rotation=0, strength=light_strength)
        bpy.context.scene.render.engine = "CYCLES"
        bpy.context.scene.cycles.device = device
        bpy.context.scene.cycles.samples = samples
        bpy.context.scene.cycles.use_denoising = True
        object.select_set(True)
        bpy.ops.object.bake(type=bake_type, use_clear=True)

        # 6. Convert to PIL
        pixels = (np.array(img.pixels) * 255).astype(np.uint8)
        pixels = pixels.reshape(img.size[1], img.size[0], 4)
        image_pil = Image.fromarray(pixels, "RGBA")

        return image_pil, self.draw_uv()

    def export(self, path: Path | str = "scene.blend"):
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(path).resolve()))

    def render(
        self,
        distance=1.75,
        samples=1,
        size=(512, 512),
        views=4,
        save_scene: Optional[str | Path] = None,
        light_strength=1.75,
    ) -> list[Image.Image]:
        scene = bpy.context.scene

        # Add camera
        camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        scene.collection.objects.link(camera)
        scene.camera = camera

        # Setup light
        load_hdri(Object3D.HDRI_PATH, rotation=0, strength=light_strength)

        # Configure render
        scene.render.film_transparent = True
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        scene.render.resolution_x, scene.render.resolution_y = size

        # Launch rendering
        radius = distance
        images = []
        if views < 1:
            return
        for ang in map(math.radians, tqdm(range(45, 45 + 360, 360 // views))):
            scene.camera.location = Vector(
                (radius * math.cos(ang), radius * math.sin(ang), radius * math.cos(ang) / 2)
            )
            scene.camera.rotation_euler = (
                (Vector((0, 0, 0)) - scene.camera.location).to_track_quat("-Z", "Y").to_euler()
            )
            fd, path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            scene.render.filepath = path
            bpy.ops.render.render(write_still=True)

            img = Image.open(path).convert("RGBA")
            images.append(img)
            os.remove(path)

        if save_scene:
            print(f"Saving scene to {save_scene}")
            self.export(save_scene)

        return images

    def change_texture(self, image_path: Path | str, mat=None):
        """
        Replaces the object's main texture (Base Color) with a new image.

        This function assumes the object has a single mesh with a material that
        uses nodes and a Principled BSDF shader. It creates or updates the image
        texture node and connects it to the Base Color input of the shader.
        """
        # Flip image vertically, because the drawn UV are vertically flipped
        fd, path = tempfile.mkstemp(suffix=".png")
        Image.open(image_path).transpose(Image.FLIP_TOP_BOTTOM).save(path)
        os.close(fd)

        # Load the new image
        image = bpy.data.images.load(str(Path(path).resolve()))

        # Get the active material
        material = mat
        if not material:
            if self.objects[0].material_slots:
                material = self.objects[0].material_slots[0].material
            else:
                raise RuntimeError("Mesh has no material assigned.")
        if not material.use_nodes:
            material.use_nodes = True

        node_tree = material.node_tree
        nodes = node_tree.nodes
        links = node_tree.links

        # Find or create the image texture node
        tex_node = next((n for n in nodes if n.type == "TEX_IMAGE"), None)
        if not tex_node:
            tex_node = nodes.new("ShaderNodeTexImage")
            tex_node.location = (-300, 300)

        tex_node.image = image

        # Find the Principled BSDF node
        bsdf_node = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
        if not bsdf_node:
            raise RuntimeError("No Principled BSDF node found in material.")

        # Remove existing Base Color connections
        while bsdf_node.inputs["Base Color"].is_linked:
            link = bsdf_node.inputs["Base Color"].links[0]
            links.remove(link)

        # Connect texture node to Base Color
        links.new(tex_node.outputs["Color"], bsdf_node.inputs["Base Color"])

        # Set the texture node as active (useful for baking)
        node_tree.nodes.active = tex_node

    def join(self, *objects: bpy.types.Object):
        for obj in bpy.data.objects:
            obj.select_set(obj in objects)
        bpy.context.view_layer.objects.active = objects[0]
        for o in objects:
            self.objects.remove(o)
        bpy.ops.object.join()
