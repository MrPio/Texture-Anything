import os
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = str(Path(__file__).parent.parent.parent.resolve())
SHAPENET_PATH = Path(ROOT_DIR, ".shapenet")


def load_shapenetcore_objects() -> dict[str, str]:
    """Fetch the objects in the `.shapenet/` root folder.

    Returns: A `dict` with UID as key and object path as value
    """
    models = {}
    for offset in tqdm(os.listdir(SHAPENET_PATH)):
        path = Path(SHAPENET_PATH, offset)
        if os.path.isdir(path):
            for model in os.listdir(path):
                models[model] = str(Path(SHAPENET_PATH, offset, model))
    print(f'Found {len(models)} objects')
    return models


def load_annotations() -> dict: ...
