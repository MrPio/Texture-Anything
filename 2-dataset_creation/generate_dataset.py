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
    parser.add_argument(
        "--computation-node",
        action="store_true",
        help="Whether to focus on thumbnail download or UV and diffusion extraction.",
    )
    parser.add_argument("--dataset", type=str, default="objaverse")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


args = parse_args()

dataset = datasets[args.dataset]()
dataset_path = dataset.DATASET_PATH
if not args.force:
    already_processed_uids = [
        x.stem for x in (dataset_path / "uv" if args.computation_node else "render").glob("*") if x.is_file()
    ]
    print(f"Already processed {len(already_processed_uids)} objects")
else:
    already_processed_uids = []

uids = dataset.statistics[dataset.statistics["valid"]].index[rank::size]
uids = [x for x in uids if x not in already_processed_uids]
if args.computation_node:
    for uid in tqdm(uids, disable=rank != 0):
        obj = dataset[uid]
        diffuse, uv_map = obj.regenerate_uv_map(
            samples=14, bake_type="GLOSSY" if args.dataset == "shapenetcore" else "DIFFUSE"
        )
        uv_filled = obj.draw_uv_map(fill=True)
        mask = np.all(np.array(uv_filled) == [0, 0, 0, 255], axis=2)

        # Commit
        diffuse.save(dataset_path / "diffuse" / f"{uid}.png")
        uv_map.save(dataset_path / "uv" / f"{uid}.png")
        np.save(dataset_path / "mask" / f"{uid}.npy", np.packbits(mask))
else:

    def download_thumbnail(uid):
        # Download thumbnail (not possible in computation nodes)
        thumbnail = requests.get(dataset.annotations.loc[uid]["thumbnail"]).content
        try:
            img = PILImage.open(BytesIO(thumbnail))
        except:
            img = None
        return uid, img

    with multiprocessing.Pool(16) as pool:
        results = pool.imap_unordered(download_thumbnail, uids)
        for uid, img in tqdm(results, total=len(uids), desc="Downloading"):
            # Skip if the render resolution is less than 0.2MP
            if img is None or img.size[0] * img.size[1] < MIN_RENDER_MEGAPIXEL:
                continue
            img.save(dataset_path / "render" / f"{uid}.png")
