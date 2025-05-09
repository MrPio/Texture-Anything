from io import BytesIO
import pandas as pd
from tqdm import tqdm
import os
import sys
import PIL.Image as PILImage
import requests
import objaverse
from pathlib import Path

ROOT_DIR = str(Path(__file__).parent.parent.resolve())
sys.path.insert(0, ROOT_DIR)
from src import *
import argparse

"""Generate the dataset from the GLB objects having 1 mesh, 1 uv and 1 diffuse texture. 
On a single compute node, takes 50m for 6_000 objects.
This script is CWD-independent"""

DOWNLOADED_OBJECTS = 26_000
MIN_UV_DENSITY = 0.01
MIN_RENDER_RES = 200_000
TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
NUM_TASK = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

parser = argparse.ArgumentParser(description="Download the dataset.")
parser.add_argument(
    "--computation-node",
    action="store_true",
    help="Whether to focus on thumbnail download or UV and diffusion extraction.",
)
args = parser.parse_args()

annotations = pd.read_parquet(Path(ROOT_DIR, "data/2-annotations_filtered_by_thumbnails.parquet"))
statistics = pd.read_parquet(Path(ROOT_DIR, "2-objects_filtering/statistics.parquet"))
selected_uids = statistics[statistics["diffuseCount"] == 1].index
objaverse._VERSIONED_PATH = Path(ROOT_DIR, ".objaverse/hf-objaverse-v1")
gbls = objaverse.load_objects(annotations.index[:DOWNLOADED_OBJECTS].to_list(), download_processes=256)
for folder in ["render", "uv", "diffuse", "caption"]:
    os.makedirs(Path(ROOT_DIR, f"data/dataset/objaverse/{folder}"), exist_ok=True)

already_downloaded_uids = [
    os.path.splitext(x)[0]
    for x in os.listdir(Path(ROOT_DIR, f"data/dataset/objaverse/{'uv' if args.computation_node else 'render'}"))
]


for uid in tqdm(selected_uids[TASK_ID::NUM_TASK]):
    if uid in already_downloaded_uids:
        continue

    if args.computation_node:
        obj = ObjaverseObject3D(uid, gbls[uid])

        # Extract diffuse texture
        diffuse = obj.textures[0]

        # Skip if texture is not square
        if diffuse.size[0] != diffuse.size[1]:
            continue

        # Bake UV map
        uv_map = obj.draw_uv_map()
        
        # Skip if UV is too sparse
        if compute_opacity(uv_map) < MIN_UV_DENSITY:
            continue

        # Commit
        diffuse.save(Path(ROOT_DIR, f"data/dataset/objaverse/diffuse/{uid}.png"))
        uv_map.save(Path(ROOT_DIR, f"data/dataset/objaverse/uv/{uid}.png"))
    else:
        # Download thumbnail (not possible in computation nodes)
        thumbnail = requests.get(annotations.loc[uid]["thumbnail"]).content
        render = PILImage.open(BytesIO(thumbnail))
        # Skip if the render resolution is less than 0.2MP
        if render.size[0] * render.size[1] < MIN_RENDER_RES:
            continue
        with open(Path(ROOT_DIR, f"data/dataset/objaverse/render/{uid}.jpg"), "wb") as f:
            f.write(thumbnail)
