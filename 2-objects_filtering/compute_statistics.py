import pandas as pd
from tqdm import tqdm
import os
import objaverse
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
from src import *

"""Generate statistics on GLB files. 
On a single compute node, takes 8m:30s for 26_000 objects.
Please make sure that you have downloaded the first DOWNLOADED_OBJECTS of the annotations table before running this script.
This script is CWD-dependent
"""

DOWNLOADED_OBJECTS = 26_000

objaverse._VERSIONED_PATH = os.path.join("../.objaverse", "hf-objaverse-v1")
annotations = pd.read_parquet("../data/2-annotations_filtered_by_thumbnails.parquet")
if os.path.exists("./statistics.parquet"):
    statistics = pd.read_parquet("./statistics.parquet")
else:
    statistics = pd.DataFrame(
        {
            "meshCount": pd.Series(dtype="int"),
            "uvCount": pd.Series(dtype="int"),
            "diffuseCount": pd.Series(dtype="int"),
        }
    )
    statistics.index.name = "uid"
paths = objaverse.load_objects(annotations["uid"][:DOWNLOADED_OBJECTS].to_list(), download_processes=1)

num_meshes = []
for uid, path in tqdm(paths.items()):
    if uid in statistics.index:
        continue
    objects = load_model(path)
    scene_stats = get_scene_stats()
    if scene_stats["mesh_count"] == 1:
        mesh = next(x for x in objects if x.type == "MESH")
        mesh_stats = get_mesh_stats(mesh)
    statistics.loc[uid] = [
        scene_stats["mesh_count"],
        mesh_stats["uv_count"] if scene_stats["mesh_count"] == 1 else None,
        mesh_stats["texture_count"] if scene_stats["mesh_count"] == 1 else None,
    ]
statistics.to_parquet("./statistics.parquet")
