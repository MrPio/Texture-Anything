#!/usr/bin/env python
"""
Generate statistics on OBJ files.
Please make sure that you have downloaded the first `DOWNLOADED_OBJECTS` of the annotations table before running this script.
This script is CWD-independent.

Usage:
    $ srun -n 16 --ntasks-per-node=4 --mem=24G --gpus-per-task=0 --partition=boost_usr_prod python generate_statistics.py
    $ srun -n 2 --mem=16G --gpus-per-task=0 --partition=boost_usr_prod --qos=boost_qos_dbg --time=00:02:00 python generate_statistics.py -d

Author:
    Valerio Morelli - 2025-05-08
"""

import argparse
from pathlib import Path
import sys
import warnings
import pandas as pd
from tqdm import tqdm

ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_PATH))
from src import *
from mpi4py import MPI

warnings.simplefilter(action="ignore", category=FutureWarning)

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--demo", action="store_true")
args = parser.parse_args()

dataset = ObjaverseDataset3D()
statistics = (
    dataset.statistics.drop(columns=["valid"])
    if dataset.statistics is not None and not args.demo
    else pd.DataFrame(
        {
            "meshCount": pd.Series(dtype="Int64"),
            "uvCount": pd.Series(dtype="Int64"),
            "diffuseCount": pd.Series(dtype="Int64"),
        }
    )
)
statistics.index.name = "uid"
paths = list(dataset.paths.items())[rank::size]
if args.demo:
    paths = paths[:4]
if rank == 0:
    print("Loaded", len(statistics), "statistics of", len(dataset.paths))
    print("Each task has to process", len(paths) - len(statistics) // size, "models")

i = 0
for uid, path in tqdm(paths) if rank == 0 else paths:
    i += 1
    if uid in statistics.index or not Path(path).exists():
        continue

    print(f"{i/len(paths):.2%}")
    try:
        obj = dataset[uid]
    except:
        continue
    statistics.loc[uid] = [
        len(obj.meshes),
        obj.mesh_stats["uv_count"] if obj.has_one_mesh else None,
        obj.mesh_stats["texture_count"] if obj.has_one_mesh else None,
    ]

# Syncronize the partial results to the root task
all_statistics: list[pd.DataFrame] = comm.gather(statistics, root=0)
if rank == 0:
    concatenated = pd.concat(all_statistics)
    final_statistics = concatenated[~concatenated.index.duplicated(keep="first")].sort_index()
    if args.demo:
        log(final_statistics)
    else:
        final_statistics.to_parquet("statistics.parquet")
