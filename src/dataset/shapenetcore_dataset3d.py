from functools import cached_property
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from .dataset3d import Dataset3D

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
SHAPENET_PATH = ROOT_DIR / ".shapenet"


class ShapeNetCoreDataset3D(Dataset3D):
    def __init__(self):
        super().__init__("shapenetcore")

    @cached_property
    def annotations(self) -> pd.DataFrame | None:
        return None

    @cached_property
    def statistics(self) -> pd.DataFrame:
        return pd.read_parquet(ROOT_DIR / "3-shape_net/statistics.parquet")

    @cached_property
    def paths(self) -> dict[str, str]:
        models = {}
        for offset in tqdm(os.listdir(SHAPENET_PATH)):
            path = Path(SHAPENET_PATH, offset)
            if os.path.isdir(path):
                for model in os.listdir(path):
                    models[model] = str(SHAPENET_PATH / offset / model)
        return models
