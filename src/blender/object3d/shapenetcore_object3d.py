from functools import cached_property
from pathlib import Path
from typing import Literal

from PIL import Image
from matplotlib.pyplot import spy

from src.utils import is_textured, is_white
from .object3d import Object3D
from PIL import Image
import bpy


class ShapeNetCoreObject3D(Object3D):
    def __init__(
        self, uid: str, path: str | Path, type: Literal["obj", "glb"] = "glb", merge_vertices=True, preprocess=True
    ):
        super(ShapeNetCoreObject3D, self).__init__(uid, Path(path, "models", f"model_normalized.{type}"), preprocess= not preprocess)
        if preprocess:
            # Remove useless meshes and merge vertices by distance
            for obj in bpy.data.objects:
                if obj.type == "MESH" and is_textured(obj) or not is_white(obj):
                    # Remove collapsed vertices
                    if merge_vertices:
                        bpy.context.view_layer.objects.active = obj
                        bpy.ops.object.mode_set(mode="EDIT")
                        bpy.ops.mesh.select_all(action="SELECT")
                        bpy.ops.mesh.remove_doubles(threshold=0.0001)
                        bpy.ops.object.mode_set(mode="OBJECT")
                else:
                    bpy.data.objects.remove(obj, do_unlink=True)


            # Recalculate Normals
            for obj in bpy.data.objects:
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.normals_make_consistent(inside=False)
                bpy.ops.object.mode_set(mode="OBJECT")

                # Edge split to fix broken normals
                bpy.ops.object.modifier_add(type="EDGE_SPLIT")
                bpy.context.object.modifiers["EdgeSplit"].split_angle = 1.32645
                bpy.ops.object.modifier_apply(modifier="EdgeSplit")

                # Reset normal vectors
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.normals_tools(mode="RESET")
                bpy.ops.object.mode_set(mode="OBJECT")
                
            # Join the mesh objects
            for obj in bpy.data.objects:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = bpy.data.objects[0]
            bpy.ops.object.join()
            self.object = bpy.data.objects[0]

            # Normalize the size so that the max dimension is 1m
            self.normalize_scale()
            self.normalize_position()


    @property
    def textures(self) -> list[Image.Image]:
        """Load all the diffuse texture images in the image folder as PIL"""
        try:
            return [Image.open(x) for x in (self.path.parents[1] / "images").glob("*")]
        except:
            return None

    @cached_property
    def renderings(self) -> list[Image.Image]:
        from src.dataset.shapenetcore_dataset3d import ShapeNetCoreDataset3D

        path = ShapeNetCoreDataset3D.DATASET_DIR / "render"
        return [Image.open(file) for file in path.glob(f"{self.uid}*")]
