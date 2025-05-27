from functools import cached_property
from pathlib import Path

import PIL
from matplotlib.pyplot import spy

from src.utils import is_textured
from .object3d import Object3D
from PIL import Image
import bpy


class ShapeNetCoreObject3D(Object3D):
    def __init__(self, uid: str, path: str | Path, type="glb", merge_vertices=True, preprocess=True):
        super(ShapeNetCoreObject3D, self).__init__(uid, Path(path, "models", f"model_normalized.{type}"))
        if type == "glb" and preprocess:
            if len(self.textures) == 1 and len(self.meshes) > 2:
                raise Exception(
                    "Tried to load a GLB-converted ShapeNetCore object with more than 1 mesh. This is not allowed because, after pruning the non textured meshes, the resulting object may represent a small part of the original one, and thus, the resulting renderings would be meaningless."
                )

            # Remove the parent axis object
            bpy.data.objects.remove(bpy.data.objects["world"], do_unlink=True)

            # Remove useless meshes and merge vertices by distance
            meshes = []
            for mesh in self.meshes:
                if is_textured(mesh):
                    meshes.append(mesh)
                    # Remove collapsed vertices
                    if merge_vertices:
                        bpy.context.view_layer.objects.active = mesh
                        bpy.ops.object.mode_set(mode="EDIT")
                        bpy.ops.mesh.select_all(action="SELECT")
                        bpy.ops.mesh.remove_doubles(threshold=0.00004)
                        bpy.ops.object.mode_set(mode="OBJECT")
                else:
                    bpy.data.objects.remove(mesh, do_unlink=True)

            self.meshes = meshes
            self.mesh = meshes[0]
            self.has_one_mesh = len(meshes) == 1

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
