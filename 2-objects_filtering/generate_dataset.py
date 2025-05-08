import pandas as pd
from tqdm import tqdm
import os
import sys
import PIL.Image as PILImage
import requests
import objaverse

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
from src import *

objaverse._VERSIONED_PATH = os.path.join("../.objaverse", "hf-objaverse-v1")

annotations = pd.read_parquet("../data/2-annotations_filtered_by_thumbnails.parquet")
statistics = pd.read_parquet("statistics.parquet")

uids = statistics[statistics["diffuseCount"] == 1].index
gbls = objaverse.load_objects(annotations.index[:26_000].to_list(), download_processes=256)
for folder in ["render", "uv", "diffuse", "caption"]:
    os.makedirs(f"../data/dataset/{folder}", exist_ok=True)

for uid in tqdm(uids):
    # Download thumbnail
    thumbnail = requests.get(annotations.loc[uid]["thumbnail"]).content

    # Extract diffuse texture
    mesh = next(x for x in load_glb(gbls[uid]) if x.type == "MESH")
    diffuse = get_diffuse_textures(mesh)[0]
    # Skip if texture is not square
    if diffuse.size[0]!=diffuse.size[1]:
        continue

    # Bake UV map
    uv_map=draw_uv_map(mesh, size=1024, stroke=1)
    

    # Commit
    with open(f"../data/dataset/render/{uid}.jpg", "wb") as f:
        f.write(thumbnail)
    diffuse.save(f"../data/dataset/diffuse/{uid}.png")
    uv_map.save(f"../data/dataset/uv/{uid}.png")
