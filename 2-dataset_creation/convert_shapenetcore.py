"""
Download ShapeNetCore and convert OBJ to GLB. CWD-independent.

Usage:
    $ srun -n 8 --mem=8G --time=00:30:00 \
        python download_and_convert_shapenetcore.py

Requires:
    HF_TOKEN in the `.env` file, liked to an account whose access to the ShapeNetCore dataset was granted.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv
from mpi4py import MPI
from tqdm import tqdm
import trimesh

sys.path.append(str(Path(__file__).parents[1].resolve()))
from src import *

assert load_dotenv()

dataset = ShapeNetCoreDataset3D()
objects_dir = dataset.DATASET_PATH / "objects"

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
categories = dataset.CATEGORIES[rank::size]
cprint("Rank", rank, "is gonna process the categories", f"green:{categories}")

# dataset.download(first=-1, subset=to_download, fresh=False, silent=rank != 0)
for i, cat in enumerate(categories):
    print(i, "/", len(categories))
    for obj_path in tqdm(list((objects_dir / cat).rglob("*.obj")), desc="Converting OBJ to GLB", disable=rank != 0):
        glb_path = Path(str(obj_path).replace(".obj", ".glb"))
        if not glb_path.exists():
            trimesh.load(obj_path).export(str(glb_path), file_type="glb")
