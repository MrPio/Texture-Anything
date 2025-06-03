import abc
from functools import cached_property
import json
from pathlib import Path
from typing import Optional
import pandas as pd

from ..blender.object3d.object3d import Object3D


class Dataset3D(abc.ABC):
    """Represents a dataset of 3D objects."""

    DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"
    DATASET_SUBFOLDERS = ["uv", "mask", "render", "diffuse", "objects"]
    IMG_EXT = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]
    MIN_UV_SCORE = 0.66
    BAKE_TYPE = "DIFFUSE"

    def __init__(self, dataset_folder: str, object_class: type[Object3D]):
        for folder in Dataset3D.DATASET_SUBFOLDERS:
            (Dataset3D.DATASET_DIR / dataset_folder / folder).mkdir(parents=True, exist_ok=True)
        self.dataset_folder = dataset_folder
        self.object_class = object_class

    @property
    @abc.abstractmethod
    def annotations(self) -> pd.DataFrame | None:
        """The metadata provided by the authors of the dataset, if available"""
        ...

    @property
    @abc.abstractmethod
    def paths(self) -> dict[str, str]:
        """A `dict` with UID as key and object path as value"""
        ...

    @cached_property
    def statistics(self) -> Optional[pd.DataFrame]:
        """The statistics generated on the downloaded models. Columns are: `meshCount`, `uvCount`, `diffuseCount`, `uvScore` and `valid`. The `valid` column enstablish if a sample meets all the requirements."""
        p = self.DATASET_DIR / "statistics.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        df["valid"] = (df["diffuseCount"] == 1) & (df["uvCount"] == 1) & (df["uvScore"] > self.MIN_UV_SCORE)
        return df

    @cached_property
    def captions(self) -> dict[str, str]:
        return json.load(open(self.DATASET_DIR / "caption" / "captions.json"))

    @cached_property
    def triplets(self) -> set[str]:
        """Load the triplets dataset as intersection of uids in `caption`, `uv` and `diffuse` folders."""
        uvs = {x.stem for x in (self.DATASET_DIR / "uv").glob("*") if x.suffix in Dataset3D.IMG_EXT}
        diffuses = {
            x.stem.split("_")[0] for x in (self.DATASET_DIR / "diffuse").glob("*") if x.suffix in Dataset3D.IMG_EXT
        }
        masks = {x.stem.split("_")[0] for x in (self.DATASET_DIR / "mask").glob("*.npy")}
        captions = {
            x.split("_")[0] for x in pd.read_json(self.DATASET_DIR / "caption" / "captions.json", typ="series").index
        }
        return set(filter(lambda uv: all(uv.split("_")[0] in x for x in [captions, diffuses, masks]), uvs))

    def __getitem__(self, args: dict | str) -> Object3D | None:
        """Get a Object3D with the given UID

        Args:
            args: The uid of the object to retrieve. If a dict, must contain at least the `uid` key. Might contain additional arguments for the constructor of the instance of Object3D returned.
        """
        if isinstance(args, str):
            args = {"uid": args}
        uid = args.pop("uid")
        silent = args.pop("silent", False)
        try:
            return self.object_class(uid, self.paths[0][uid], **args)
        except Exception as e:
            if not silent:
                print(e)
            return None

    def download(self) -> None:
        raise NotImplementedError()
