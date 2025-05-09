from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
import sys

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent))
from src import *

"""Generate statistics on OBJ files. 
Please make sure that you have downloaded ShapeNetCore dataset in the `.shapenet/` root folder.
This script is CWD-independent
"""

TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
NUM_TASK = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
statistics_path = Path(SCRIPT_DIR, f"statistics{TASK_ID if NUM_TASK>1 else ''}.parquet")
if os.path.exists(statistics_path):
    statistics = pd.read_parquet(statistics_path)
else:
    statistics = pd.DataFrame(
        {
            "meshCount": pd.Series(dtype="int"),
            "uvCount": pd.Series(dtype="int"),
            "diffuseCount": pd.Series(dtype="int"),
            "faceCount": pd.Series(dtype="int"),
        }
    )
    statistics.index.name = "uid"
paths = load_shapenetcore_objects()

num_meshes = []
for uid, path in tqdm(list(paths.items())[TASK_ID::NUM_TASK]):
    if uid in statistics.index:
        continue
    obj_path = os.path.join(path, "models", "model_normalized.obj")
    if not os.path.exists(obj_path):
        continue
    objects = load_model(obj_path)
    scene_stats = get_scene_stats()
    if scene_stats["mesh_count"] == 1:
        mesh = next(x for x in objects if x.type == "MESH")
        mesh_stats = get_mesh_stats(mesh)

    statistics.loc[uid] = [
        scene_stats["mesh_count"],
        mesh_stats["uv_count"] if scene_stats["mesh_count"] == 1 else None,
        mesh_stats["texture_count"] if scene_stats["mesh_count"] == 1 else None,
        mesh_stats["face_count"] if scene_stats["mesh_count"] == 1 else None,
    ]
statistics.to_parquet(statistics_path)
