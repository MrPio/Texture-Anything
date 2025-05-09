import os
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = str(Path(__file__).parent.parent.parent.resolve())
SHAPENET_PATH = Path(ROOT_DIR, ".shapenet")


def load_shapenetcore_objects() -> dict[str, str]:
    models = {}
    for offset in tqdm(os.listdir(SHAPENET_PATH)):
        path=Path(SHAPENET_PATH, offset)
        if os.path.isdir(path):
            for model in os.listdir(path):
                models[model] = str(Path(SHAPENET_PATH, offset, model))
    return models


def load_annotations() -> dict: ...
