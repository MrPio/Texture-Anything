"""
Generate the dataset from the GLB objects having 1 mesh, 1 uv and 1 diffuse texture.
This script is CWD-independent

Usage:
    $ srun -n 4 --mem=24G  --time=04:00:00 \
        python 2-dataset_creation/generate_dataset.py --dataset="shapenetcore" --regenerate-uv

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
already_proc = set() if args.overwrite else {x.stem for x in (DATASET_DIR / "uv").glob("*.png")}
log(f"Already processed", len(already_proc), "objects")

uids = dataset.statistics[dataset.statistics["valid"]].index[rank::size]
uids = set(uids).difference(already_proc)

# Optimization --------------------------------
if args.regenerate_uv:
    prefs = bpy.context.preferences
    prefs.edit.use_global_undo = False
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            area.spaces[0].shading.show_backface_culling = False
            area.tag_redraw()

    bpy.context.view_layer.update()
    load_hdri(Object3D.HDRI_PATH_WHITE, rotation=0, strength=1.5)
    device = "GPU" if has_cuda() else "CPU"
    log("Device=", f"red:{device}")

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    if has_cuda():
        prefs = bpy.context.preferences
        prefs.addons["cycles"].preferences.compute_device_type = "CUDA"  # or 'OPTIX' on NVIDIA
        scene.cycles.device = "GPU"
        scene.cycles.tile_x = 256
        scene.cycles.tile_y = 256
    else:
        scene.cycles.device = "CPU"

# ---------------------------------------------

for uid in tqdm(uids, disable=rank != 0):
    obj = dataset[uid]
    try:
        if args.regenerate_uv:
            diffuse, uv_map = obj.regenerate_uv_map(
                samples=8,
                bake_type=dataset.BAKE_TYPE,
                load_lights=False,
                device=device,
            )
        else:
            diffuse, uv_map = obj.textures[0], obj.draw_uv_map()
        if compute_image_density(uv_map) < MIN_UV_DENSITY:
            continue
        uv_filled = obj.draw_uv_map(fill=True)
        mask = np.all(np.array(uv_filled) == [0, 0, 0, 255], axis=2)

        # Commit
        diffuse.save(DATASET_DIR / "diffuse" / f"{uid}.png")
        uv_map.save(DATASET_DIR / "uv" / f"{uid}.png")
        np.save(DATASET_DIR / "mask" / f"{uid}.npy", np.packbits(mask))

        if args.render:
            renderings = obj.render(samples=1, views=3, size=(512, 512), distance=1.75)
            for i, rendering in enumerate(renderings):
                rendering.save(DATASET_DIR / "render" / f"{uid}_{i}.png")
    except:
        continue
