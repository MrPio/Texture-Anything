from pathlib import Path
from .object3d import Object3D
import numpy as np
from PIL import Image


class ObjaverseObject3D(Object3D):
    def __init__(self, uid: str, path: str | Path):
        super(ObjaverseObject3D, self).__init__(uid, path)

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
