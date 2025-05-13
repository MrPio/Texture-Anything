"""
Generate statistics on OBJ files.
Please make sure that you have downloaded the first `DOWNLOADED_OBJECTS` of the annotations table before running this script.
This script is CWD-independent.

This script is meant to be used with "--array" SLURM flag and require post manual merging of the output parquet files. A wiser implementation should use MIP.
"""

from pathlib import Path
import objaverse
import pandas as pd
from tqdm import tqdm
import os
import sys

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_PATH = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from src import *
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


DOWNLOADED_OBJECTS = 45_000
TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
NUM_TASK = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

objaverse._VERSIONED_PATH = os.path.join(ROOT_PATH / ".objaverse/hf-objaverse-v1")
annotations = pd.read_parquet(ROOT_PATH / "data/2-annotations_filtered_by_thumbnails.parquet")
paths = objaverse.load_objects(annotations.index[:DOWNLOADED_OBJECTS].to_list())

statistics_path = Path(SCRIPT_DIR, f"statistics{TASK_ID if NUM_TASK>1 else ''}.parquet")
if os.path.exists(statistics_path):
    statistics = pd.read_parquet(statistics_path)
else:
    statistics = pd.DataFrame(
        {
            "meshCount": pd.Series(dtype="Int64"),
            "uvCount": pd.Series(dtype="Int64"),
            "diffuseCount": pd.Series(dtype="Int64"),
            "faceCount": pd.Series(dtype="Int64"),
        }
    )
    statistics.index.name = "uid"

num_meshes = []
for uid, path in tqdm(list(paths.items())[TASK_ID::NUM_TASK]):
    if uid in statistics.index:
        continue

    if not Path(path).exists():
        continue
    obj = ObjaverseObject3D(uid, path)
    if obj.has_one_mesh:
        mesh_stats = obj.mesh_stats

    statistics.loc[uid] = [
        len(obj.meshes),
        mesh_stats["uv_count"] if obj.has_one_mesh else None,
        mesh_stats["texture_count"] if obj.has_one_mesh else None,
    ]
statistics.to_parquet(statistics_path)
