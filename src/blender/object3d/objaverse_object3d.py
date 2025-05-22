from functools import cached_property
from pathlib import Path

from matplotlib.pyplot import spy
import requests

import PIL
from .object3d import Object3D
import numpy as np
from PIL import Image
import bpy


class ObjaverseObject3D(Object3D):
    def __init__(self, uid: str, path: str | Path):
        from src.dataset.objaverse_dataset3d import ObjaverseDataset3D

        super(ObjaverseObject3D, self).__init__(uid, path)
        self.dataset = ObjaverseDataset3D()

        for mesh in self.objects:
            if mesh != self.mesh:
                bpy.data.objects.remove(mesh, do_unlink=True)
        self.objects = [self.mesh]
        self.meshes = [self.mesh]
        self.has_one_mesh = True

    @property
    def textures(self) -> list[Image.Image]:
        """Unpack all the diffuse texture images in the scene as PIL"""
        assert self.has_one_mesh

        diffuse_images = set(x.image for x in self._mesh_nodes if x.image)
        embedded_images = [img for img in diffuse_images if img.packed_file is not None]
        images_pil = []

        if embedded_images:
            for img in embedded_images:
                pixels = (np.array(img.pixels) * 255).astype(np.uint8)
                pixels = pixels.reshape((*img.size, 4))
                image_pil = Image.fromarray(pixels, "RGBA")
                images_pil.append(image_pil)

        return images_pil

    @cached_property
    def rendering(self) -> PIL.Image.Image | None:
        from src.dataset.objaverse_dataset3d import ObjaverseDataset3D

        path = ObjaverseDataset3D.DATASET_PATH / "render" / f"{self.uid}.jpg"
        return PIL.Image.open(
            path
            if path.exists()
            else requests.get(self.dataset.annotations.loc[self.uid]["thumbnail"], stream=True).raw
        )
