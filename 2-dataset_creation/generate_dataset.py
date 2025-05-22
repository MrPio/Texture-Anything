"""
Generate the dataset from the GLB objects having 1 mesh, 1 uv and 1 diffuse texture.
This script is CWD-independent

Usage:
    $ srun -n 2 --ntasks-per-node=4 --mem=24G --gpus-per-task=0 --partition=boost_usr_prod --qos=boost_qos_dbg python generate_dataset.py --computation-node

Author:
    Valerio Morelli - 2025-05-08
"""

from io import BytesIO
import multiprocessing
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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--computation-node",
    action="store_true",
    help="Whether to focus on thumbnail download or UV and diffusion extraction.",
)
parser.add_argument("--dataset", type=str, default="objaverse")
args = parser.parse_args()

dataset = datasets[args.dataset]()
dataset_path = dataset.DATASET_PATH
already_processed_uids = [
    x.stem for x in (dataset_path / "uv" if args.computation_node else "render").glob("*") if x.is_file()
]
print(f"Already processed {len(already_processed_uids)} objects")

uids = dataset.statistics[dataset.statistics["valid"]].index[rank::size]
uids = [x for x in uids if x not in already_processed_uids]
if args.computation_node:
    for uid in tqdm(uids) if rank == 0 else uids:
        obj = dataset[uid]
        diffuse = obj.textures[0]

        # Skip if texture is not square
        if diffuse.size[0] != diffuse.size[1]:
            continue

        uv_map = obj.draw_uv_map()
        
        # Skip if UV is too sparse
        if compute_image_density(uv_map) < MIN_UV_DENSITY:
            continue

        control_uv_map = obj.draw_uv_map(fill=True)

        # Commit
        diffuse.save(dataset_path / "diffuse" / f"{uid}.png")
        uv_map.save(dataset_path / "uv" / f"{uid}.png")
        control_uv_map.save(dataset_path / "control" / f"{uid}.png")
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
