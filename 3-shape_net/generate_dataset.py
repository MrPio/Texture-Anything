import pandas as pd
from tqdm import tqdm
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_PATH = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from src import *

"""Generate the dataset from the OBJ objects having 1 mesh, 1 uv and 1 diffuse texture. 
This script is CWD-independent"""

MIN_UV_DENSITY = 0.01
TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
NUM_TASK = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

statistics = pd.read_parquet(SCRIPT_DIR / "statistics.parquet")
selected_uids = statistics[statistics["valid"]].index
dataset = load_shapenetcore_objects()
for folder in ["render", "uv", "diffuse", "caption"]:
    os.makedirs(ROOT_PATH / f"data/dataset/shapenetcore/{folder}", exist_ok=True)

already_downloaded_uids = [os.path.splitext(x)[0] for x in os.listdir(Path(ROOT_PATH, f"data/dataset/shapenetcore/uv"))]


for uid in tqdm(selected_uids[TASK_ID::NUM_TASK]):
    if uid in already_downloaded_uids:
        continue
    
    obj = ShapeNetCoreObject3D(uid, paths[uid])

    # Extract diffuse texture
    diffuse = obj.textures
    if not diffuse:
        continue

    # Skip if texture is not square
    if diffuse[0].size[0] != diffuse[0].size[1]:
        continue

    # Bake UV map
    uv_map = obj.draw_uv_map()

    # Skip if UV is too sparse
    if compute_image_density(uv_map) < MIN_UV_DENSITY:
        continue

    # Commit
    diffuse[0].save(ROOT_PATH / f"data/dataset/shapenetcore/diffuse/{uid}.png")
    uv_map.save(ROOT_PATH / f"data/dataset/shapenetcore/uv/{uid}.png")
