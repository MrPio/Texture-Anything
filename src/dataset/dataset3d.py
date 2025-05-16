import abc
from functools import cached_property
from pathlib import Path
from typing import Optional
import pandas as pd

from ..blender.object3d.object3d import Object3D


class Dataset3D(abc.ABC):
    """Represents a dataset of 3D objects."""

    DATASET_PATH = Path(__file__).resolve().parents[2] / "dataset"
    DATASET_SUBFOLDERS = ["uv", "render", "diffuse", "objects"]
    IMG_EXT = [".jpg", ".png"]

    def __init__(self, dataset_folder: str, object_class: type[Object3D]):
        for folder in Dataset3D.DATASET_SUBFOLDERS:
            (Dataset3D.DATASET_PATH / dataset_folder / folder).mkdir(parents=True, exist_ok=True)
        self.dataset_folder = dataset_folder
        self.object_class = object_class

    @property
    @abc.abstractmethod
    def annotations(self) -> pd.DataFrame | None:
        """The metadata provided by the authors of the dataset, if available"""
        ...

    @property
    @abc.abstractmethod
    def statistics(self) -> pd.DataFrame:
        """The statistics generated on the downloaded models. Columns are: `meshCount`, `uvCount`, `diffuseCount`"""
        ...

    @property
    @abc.abstractmethod
    def paths(self) -> dict[str, str]:
        """A `dict` with UID as key and object path as value"""
        ...

    @cached_property
    def statistics(self) -> Optional[pd.DataFrame]:
        p = self.DATASET_PATH / "statistics.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        df["valid"] = (df["diffuseCount"] == 1) & (df["uvScore"] > 0.66)
        return df

    @cached_property
    def triplets(self) -> set[str]:
        """Load the triplets dataset as intersection of uids in `caption`, `uv` and `diffuse` folders."""
        path = Dataset3D.DATASET_PATH / self.dataset_folder
        captions = {x.stem for x in (path / "render").glob("*") if x.suffix.lower() in Dataset3D.IMG_EXT}
        uvs = {x.stem for x in (path / "uv").glob("*") if x.suffix.lower() in Dataset3D.IMG_EXT}
        diffuses = {x.stem for x in (path / "diffuse").glob("*") if x.suffix.lower() in Dataset3D.IMG_EXT}
        return captions.intersection(uvs, diffuses)

    def __getitem__(self, key) -> Object3D | None:
        try:
            return self.object_class(key, self.paths[key])
        except Exception as e:
            print(e)
            return None

    def download(self) -> None:
        raise NotImplementedError()
