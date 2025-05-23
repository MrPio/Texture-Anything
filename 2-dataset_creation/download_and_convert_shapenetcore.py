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

sys.path.append(str(Path("..").resolve()))
from src import *

assert load_dotenv()

dataset = ShapeNetCoreDataset3D()
cprint("You have downloaded a total of", f"blue:{len(dataset.paths)}", "objects!")

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
to_download = range(0, len(dataset.CATEGORIES))[rank::size]
cprint("Rank", rank, "is gonna download the categories", f"green:{to_download}")

dataset.download(first=-1, subset=to_download, fresh=False, silent=rank != 0)
