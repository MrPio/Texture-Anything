from functools import cached_property
from pathlib import Path

from matplotlib.pyplot import spy
import requests

from src.utils import bpy2pil, flatten
from .object3d import Object3D
import numpy as np
from PIL import Image
import bpy


class ObjaverseObject3D(Object3D):
    def __init__(self, uid: str, path: str | Path):
        from src.dataset.objaverse_dataset3d import ObjaverseDataset3D

        super(ObjaverseObject3D, self).__init__(uid, path)
        self.dataset = ObjaverseDataset3D()

    @property
    def textures(self) -> list[Image.Image]:
        """Unpack all the diffuse texture images in the scene as PIL"""

        diffuse_images = set(x.image for x in self._mesh_nodes if x.image)
        embedded_images = [img for img in diffuse_images if img.packed_file is not None]
        return list(map(bpy2pil, embedded_images))

    @cached_property
    def renderings(self) -> list[Image.Image]:
        from src.dataset.shapenetcore_dataset3d import ShapeNetCoreDataset3D

        path = ShapeNetCoreDataset3D.DATASET_DIR / "render"
        renderings = [Image.open(file) for file in path.glob(f"{self.uid}*")]
        return (
            renderings
            if len(renderings) > 0
            else [Image.open(requests.get(self.dataset.annotations.loc[self.uid]["thumbnail"], stream=True).raw)]
        )
