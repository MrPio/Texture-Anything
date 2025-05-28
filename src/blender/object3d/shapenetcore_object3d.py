from functools import cached_property
from pathlib import Path
from typing import Literal

import PIL
from matplotlib.pyplot import spy

from src.utils import is_textured, is_white
from .object3d import Object3D
from PIL import Image
import bpy


class ShapeNetCoreObject3D(Object3D):
    def __init__(
        self, uid: str, path: str | Path, type: Literal["obj", "glb"] = "glb", merge_vertices=True, preprocess=True
    ):
        super(ShapeNetCoreObject3D, self).__init__(uid, Path(path, "models", f"model_normalized.{type}"))
        if preprocess:
            if len(self.textures) == 1 and len(self.meshes) > 2:
                raise Exception(
                    "Tried to load a GLB-converted ShapeNetCore object with more than 1 mesh. This is not allowed because, after pruning the non textured meshes, the resulting object may represent a small part of the original one, and thus, the resulting renderings would be meaningless."
                )

            # Remove useless meshes and merge vertices by distance
            meshes = []
            for mesh in self.meshes:
                if is_textured(mesh) or not is_white(mesh):
                    meshes.append(mesh)
                    # Remove collapsed vertices
                    if merge_vertices:
                        bpy.context.view_layer.objects.active = mesh
                        bpy.ops.object.mode_set(mode="EDIT")
                        bpy.ops.mesh.select_all(action="SELECT")
                        bpy.ops.mesh.remove_doubles(threshold=0.0001)
                        bpy.ops.object.mode_set(mode="OBJECT")
                else:
                    bpy.data.objects.remove(mesh, do_unlink=True)

            if len(meshes)>1:
                for obj in meshes:
                    obj.select_set(True)
                bpy.context.view_layer.objects.active = meshes[0]
                bpy.ops.object.join()
                
            self.meshes = [meshes[0]]
            self.mesh = meshes[0]
            self.has_one_mesh = len(self.meshes) == 1

            # Recalculate Normals
            bpy.context.view_layer.objects.active = self.mesh
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.mode_set(mode="OBJECT")

            # Edge split to fix broken normals
            bpy.context.view_layer.objects.active = self.mesh
            bpy.ops.object.modifier_add(type="EDGE_SPLIT")
            bpy.context.object.modifiers["EdgeSplit"].split_angle = 1.32645
            bpy.ops.object.modifier_apply(modifier="EdgeSplit")

            # Reset normal vectors
            for mesh in self.meshes:
                bpy.context.view_layer.objects.active = mesh
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.normals_tools(mode="RESET")
                bpy.ops.object.mode_set(mode="OBJECT")

            self.normalize_scale()

    @property
    def textures(self) -> list[Image.Image]:
        """Load all the diffuse texture images in the image folder as PIL"""
        try:
            return [Image.open(x) for x in (self.path.parents[1] / "images").glob("*")]
        except:
            return None

    @cached_property
    def rendering(self) -> PIL.Image.Image | None:
        from src.dataset.shapenetcore_dataset3d import ShapeNetCoreDataset3D

        path = ShapeNetCoreDataset3D.DATASET_DIR / "render" / f"{self.uid}.jpg"
        return PIL.Image.open(path) if path.exists() else None
