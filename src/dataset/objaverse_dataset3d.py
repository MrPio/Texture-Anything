from functools import cached_property
import pandas as pd
import objaverse
from .dataset3d import Dataset3D
from ..blender.object3d.objaverse_object3d import ObjaverseObject3D


class ObjaverseDataset3D(Dataset3D):
    DATASET_DIR = Dataset3D.DATASET_DIR / "objaverse"

    def __init__(self):
        objaverse._VERSIONED_PATH = str(ObjaverseDataset3D.DATASET_DIR / "objects")
        super().__init__("objaverse", ObjaverseObject3D)

    @cached_property
    def annotations(self) -> pd.DataFrame | None:
        return pd.read_parquet(ObjaverseDataset3D.DATASET_DIR / "2-annotations_filtered_by_thumbnails.parquet")

    @cached_property
    def paths(self) -> tuple[dict[str, str], dict[str, int]]:
        paths = pd.read_parquet(ObjaverseDataset3D.DATASET_DIR / "objaverse_glbs.parquet")
        return paths["path"].to_dict(), paths["size"].to_dict()

    def download(self, processes=16) -> None:
        objaverse.load_objects(self.annotations.index, download_processes=processes)
