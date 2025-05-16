from functools import cached_property
import pandas as pd
import objaverse
from .dataset3d import Dataset3D
from ..blender.object3d.objaverse_object3d import ObjaverseObject3D


class ObjaverseDataset3D(Dataset3D):
    DATASET_PATH = Dataset3D.DATASET_PATH / "objaverse"

    def __init__(self):
        objaverse._VERSIONED_PATH = str(ObjaverseDataset3D.DATASET_PATH / "objects")
        super().__init__("objaverse", ObjaverseObject3D)

    @cached_property
    def annotations(self) -> pd.DataFrame | None:
        return pd.read_parquet(ObjaverseDataset3D.DATASET_PATH / "2-annotations_filtered_by_thumbnails.parquet")

    @cached_property
    def paths(self) -> dict[str, str]:
        num_objs = sum(1 for _ in (ObjaverseDataset3D.DATASET_PATH / "objects").rglob("*.glb"))
        return objaverse.load_objects(self.annotations.index[:num_objs])

    def download(self, processes=16) -> None:
        objaverse.load_objects(self.annotations.index, download_processes=processes)
