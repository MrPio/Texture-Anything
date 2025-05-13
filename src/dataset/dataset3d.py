import abc
from functools import cached_property
import os
from pathlib import Path
import pandas as pd
from ..blender.object3d.object3d import Object3D

DATASET_PATH = Path(__file__).parent.parent.parent.resolve() / "data/dataset"


class Dataset3D(abc.ABC):
    """Represents a dataset of 3D objects."""

    def __init__(self, dataset_folder: str):
        self.dataset_folder = dataset_folder

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
    def triplets(self) -> set[str]:
        """Load the triplets dataset as intersection of uids in `caption`, `uv` and `diffuse` folders."""
        path = DATASET_PATH / self.dataset_folder
        captions = {x.stem for x in (path / "render").glob("*.jpg")}
        uvs = {x.stem for x in (path / "uv").glob("*.png")}
        diffuses = {x.stem for x in (path / "diffuse").glob("*.png")}
        return captions.intersection(uvs, diffuses)

    @abc.abstractmethod
    def __getitem__(self, key) -> Object3D: ...

    def download(self) -> None:
        raise NotImplementedError()
