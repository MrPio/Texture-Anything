from functools import cached_property
import pandas as pd
from tqdm import tqdm
from .dataset3d import Dataset3D
from ..blender.object3d.shapenetcore_object3d import ShapeNetCoreObject3D


class ShapeNetCoreDataset3D(Dataset3D):
    DATASET_PATH = Dataset3D.DATASET_PATH / "shapenetcore"

    def __init__(self):
        super().__init__("shapenetcore")

    @cached_property
    def annotations(self) -> pd.DataFrame | None:
        return None

    @cached_property
    def paths(self) -> dict[str, str]:
        return {
            m.name: str(m)
            for p in tqdm((ShapeNetCoreDataset3D.DATASET_PATH / "objects").iterdir())
            if p.is_dir()
            for m in p.iterdir()
        }

    def __getitem__(self, key) -> ShapeNetCoreObject3D:
        try:
            return ShapeNetCoreObject3D(key, self.paths[key])
        except:
            return None
