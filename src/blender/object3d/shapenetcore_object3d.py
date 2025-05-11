from pathlib import Path
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

    @property
    def screenshots(self) -> list[Image.Image]:
        """Load the screenshots, if available"""
        if not os.path.exists(self.path.parent.parent / "screenshots"):
            return []
        files = os.listdir(self.path.parent.parent / "screenshots")
        return [Image.open(self.path.parent.parent / "screenshots" / x) for x in files]
