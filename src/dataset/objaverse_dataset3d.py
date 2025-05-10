from functools import cached_property
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from .dataset3d import Dataset3D
import objaverse

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
OBJAVERSE_PATH = ROOT_DIR / ".objaverse/hf-objaverse-v1"
objaverse._VERSIONED_PATH = str(OBJAVERSE_PATH)


class ObjaverseDataset3D(Dataset3D):
    @cached_property
    def annotations(self) -> pd.DataFrame | None:
        return pd.read_parquet(ROOT_DIR / "data/2-annotations_filtered_by_thumbnails.parquet")

    @cached_property
    def statistics(self) -> pd.DataFrame:
        df = pd.read_parquet(ROOT_DIR / "2-objects_filtering/statistics.parquet")
        df["valid"] = df["diffuseCount"] == 1
        return df

    @cached_property
    def paths(self) -> dict[str, str]:
        num_objs = sum(1 for _ in OBJAVERSE_PATH.rglob("*.glb")) - 50
        return objaverse.load_objects(self.annotations.index[:num_objs])

    def download(self, processes=16) -> None:
        objaverse.load_objects(self.annotations.index, download_processes=processes)
