"""
Generate the dataset from the GLB objects having 1 mesh, 1 uv and 1 diffuse texture.
This script is CWD-independent

Usage:
    $ srun -n 24 --mem=10G --gpus-per-task=0 --partition=boost_usr_prod \
        python generate_dataset.py --computation-node

Author:
    Valerio Morelli - 2025-05-08
"""

from io import BytesIO
import multiprocessing
import numpy as np
from tqdm import tqdm
import sys
import PIL.Image as PILImage
import requests
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from src import *
from mpi4py import MPI
import argparse


MIN_UV_DENSITY = 0.0085
MIN_RENDER_MEGAPIXEL = 200_000
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
cprint("Rank:", rank, "Size:", size)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="objaverse")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--regenerate-uv", action="store_true")
    return parser.parse_args()


args = parse_args()

dataset = datasets[args.dataset]()
DATASET_DIR = dataset.DATASET_DIR
if not args.overwrite_existing:
    already_processed_uids = [
        x.stem for x in (DATASET_DIR / "uv" if args.computation_node else "render").glob("*") if x.is_file()
    ]
    print(f"Already processed {len(already_processed_uids)} objects")
else:
    already_processed_uids = []

uids = dataset.statistics[dataset.statistics["valid"]].index[rank::size]
uids = [x for x in uids if x not in already_processed_uids]

for uid in tqdm(uids, disable=rank != 0):
    obj = dataset[uid]
    if args.regenerate_uv:
        diffuse, uv_map = obj.regenerate_uv_map(
            samples=10,
            bake_type="GLOSSY" if args.dataset == "shapenetcore" else "DIFFUSE",
        )
    else:
        diffuse, uv_map = obj.textures[0], obj.draw_uv_map()
    uv_filled = obj.draw_uv_map(fill=True)
    mask = np.all(np.array(uv_filled) == [0, 0, 0, 255], axis=2)

    # Commit
    diffuse.save(DATASET_DIR / "diffuse" / f"{uid}.png")
    uv_map.save(DATASET_DIR / "uv" / f"{uid}.png")
    np.save(DATASET_DIR / "mask" / f"{uid}.npy", np.packbits(mask))
