"""
Generate the dataset from the GLB objects having 1 mesh, 1 uv and 1 diffuse texture.
This script is CWD-independent

Usage:
    $ srun -n 8 --mem=30G  --time=04:00:00 python generate_dataset.py --dataset="shapenetcore" --regenerate-uv --render
    $ srun -n 8 --mem=30G  --time=04:00:00 python generate_dataset.py --dataset="objaverse" --render --split ../dataset/objaverse/missing_thumbnails.txt

Author:
    Valerio Morelli - 2025-05-08
"""

from time import time_ns
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
MIN_UV_SCORE = 0.55
MAX_SIZE = 4 * 2**20  # 4 MiB
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
cprint("Rank:", rank, "Size:", size)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="objaverse")
    parser.add_argument("--split", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--regenerate-uv", action="store_true")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


args = parse_args()

dataset = datasets[args.dataset]()
conv_filter = LaplacianFilter()
DIR = dataset.DATASET_DIR
# uids = dataset.statistics[dataset.statistics["valid"]].index[rank::size]
paths, sizes = dataset.paths
uids = set(paths.keys())
if args.split:
    split = open(args.split).read().split("\n")
    uids = uids.intersection(split)

# processed_uids = {} if args.overwrite else {file.stem for file in (DIR / "uv").glob("*")}
# uids = set(uids).difference(processed_uids)
cprint("Need to process", len(uids), "uids")
uids = list(uids)[rank::size]
cprint(f"[Rank {rank}/{size}] I have", len(uids), "uids")

# Optimization --------------------------------
if args.regenerate_uv:
    prefs = bpy.context.preferences
    prefs.edit.use_global_undo = False
    load_hdri(Object3D.HDRI_PATH_WHITE, rotation=0, strength=2)
    device = "GPU" if has_cuda() else "CPU"
    log("Device=", f"red:{device}")
# ---------------------------------------------

for uid in tqdm(uids, disable=rank != 0):
    # Load the object into the blender scene
    if sizes[uid] > MAX_SIZE or (obj := dataset[dict(uid=uid, preprocess=args.render, silent=True)]) is None:
        ObjaverseObject3D(uid, paths[uid], preprocess=True)
        print(sizes[uid], obj is None)
        continue

    print(args.render, (DIR / "render" / f"{uid}.png").exists())
    # Renderings
    if args.render and (args.overwrite or not (DIR / "render" / f"{uid}.png").exists()):
        renderings = obj.render(samples=1, views=1, size=(512, 512))
        for i, rendering in enumerate(renderings):
            rendering.save(DIR / "render" / f"{uid}_{i}.png")

    # Merge meshes with the same material
    processor = Processor(obj)
    processor.analyze_scene(verbose=False)
    processor.group_meshes(verbose=False)
    processor.analyze_scene(verbose=False)

    uvs = processor.uvs(pil=True)
    diffuses = processor.diffuses(pil=True)
    masks = processor.masks()

    for i, (uv, diff, mask) in enumerate(zip(uvs, diffuses, masks)):
        uv_path = DIR / "uv" / f"{uid}_{i}.png"
        diffuse_path = DIR / "diffuse" / f"{uid}_{i}.png"
        mask_path = DIR / "mask" / f"{uid}_{i}.npy"

        if not args.overwrite and all(p.exists() for p in [uv_path, diffuse_path, mask_path]):
            print("-", end="")
            continue

        # UVs, Diffuses and Masks
        if args.regenerate_uv:
            # TODO: GROUP MESHES WITH PROCESSOR TO ALLOW MULTI MESH OBJS
            raise NotImplementedError()
            diffuse, uv = obj.regenerate_uv_map(
                samples=10,
                bake_type=dataset.BAKE_TYPE,
                device=device,
            )

        # Quality checks
        if (
            any(_ is None for _ in [uv, diff, mask])
            or diff.size[0] != diff.size[1]
            or compute_image_density(uv) < MIN_UV_DENSITY
            or (uv_score := obj.uv_score(obj.objects[i].data)) is None
            or uv_score < MIN_UV_SCORE
            or conv_filter.is_plain(diff)
        ):
            continue

        diff.save(diffuse_path)
        uv.save(uv_path)
        np.save(mask_path, np.packbits(mask))
        if size == 1:
            cprint(f"green:EXPORTED --> {uid}_{i}.png")
