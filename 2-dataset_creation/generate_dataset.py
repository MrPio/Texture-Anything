"""
Generate the dataset from the GLB objects having 1 mesh, 1 uv and 1 diffuse texture.
This script is CWD-independent

Usage:
    $ srun -n 8 --mem=30G  --time=02:00:00 python 2-dataset_creation/generate_dataset.py --dataset="shapenetcore" --regenerate-uv --render

Author:
    Valerio Morelli - 2025-05-08
"""

import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path
import bpy

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from src import *
from mpi4py import MPI
import argparse
from torch.cuda import is_available as has_cuda

MIN_UV_DENSITY = 0.0085
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
cprint("Rank:", rank, "Size:", size)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="objaverse")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--regenerate-uv", action="store_true")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


args = parse_args()

dataset = datasets[args.dataset]()
DATASET_DIR = dataset.DATASET_DIR
uids = dataset.statistics[dataset.statistics["valid"]].index[rank::size]

# Optimization --------------------------------
if args.regenerate_uv:
    prefs = bpy.context.preferences
    prefs.edit.use_global_undo = False
    load_hdri(Object3D.HDRI_PATH_WHITE, rotation=0, strength=2)
    device = "GPU" if has_cuda() else "CPU"
    log("Device=", f"red:{device}")
# ---------------------------------------------

for uid in tqdm(uids, disable=rank != 0):
    diffuse_path = DATASET_DIR / "diffuse" / f"{uid}.png"
    uv_path = DATASET_DIR / "uv" / f"{uid}.png"
    mask_path = DATASET_DIR / "mask" / f"{uid}.npy"
    if (obj := dataset[uid]) is None:
        continue
    # try:
    if args.overwrite or not diffuse_path.exists() or not uv_path.exists():
        if args.regenerate_uv:
            diffuse, uv_map = obj.regenerate_uv_map(
                samples=10,
                bake_type=dataset.BAKE_TYPE,
                device=device,
            )
        else:
            diffuse, uv_map = obj.textures[0], obj.draw_uv_map()

        if compute_image_density(uv_map) < MIN_UV_DENSITY:
            continue

        diffuse.save(diffuse_path)
        uv_map.save(uv_path)

    if args.overwrite or not mask_path.exists():
        uv_filled = obj.draw_uv_map(fill=True)
        mask = np.all(np.array(uv_filled) == [0, 0, 0, 255], axis=2)
        np.save(mask_path, np.packbits(mask))

    if args.render and (args.overwrite or not (DATASET_DIR / "render" / f"{uid}_2.png").exists()):
        renderings = obj.render(samples=1, views=3, size=(512, 512))
        for i, rendering in enumerate(renderings):
            rendering.save(DATASET_DIR / "render" / f"{uid}_{i}.png")
    # except Exception as e:
    #     print(e)
    #     continue
