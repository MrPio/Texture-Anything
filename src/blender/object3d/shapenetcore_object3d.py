from functools import cached_property
from pathlib import Path

import PIL
from .object3d import Object3D
from PIL import Image
import os


class ShapeNetCoreObject3D(Object3D):
    def __init__(self, uid: str, path: str | Path):
        super(ShapeNetCoreObject3D, self).__init__(uid, Path(path, "models", "model_normalized.obj"))

    @property
    def textures(self) -> list[Image.Image]:
        """Load all the diffuse texture images in the image folder as PIL"""
        files = [x.image.name for x in self._mesh_nodes]
        try:
            return [Image.open(self.path.parent.parent / "images" / x) for x in files]
        except:
            return None

    @cached_property
    def render(self) -> PIL.Image.Image | None:
        from src.dataset.shapenetcore_dataset3d import ShapeNetCoreDataset3D

        path = ShapeNetCoreDataset3D.DATASET_PATH / "render" / f"{self.uid}.jpg"
        return PIL.Image.open(path) if path.exists() else None
