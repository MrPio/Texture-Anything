#!/usr/bin/env python
"""
Generate statistics on OBJ files.
Please make sure that you have downloaded the objects relative to the UIDs in the annotations table before running this script.
This script is CWD-independent.

Usage:
    $ srun -n 4 --mem=24G  --time=04:00:00 python 2-dataset_creation/generate_statistics.py --dataset="shapenetcore"

Author:
    Valerio Morelli - 2025-05-08
"""

import argparse
from pathlib import Path
import sys
import warnings
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from src import *
from mpi4py import MPI

warnings.simplefilter(action="ignore", category=FutureWarning)

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--demo", action="store_true")
parser.add_argument("--dataset", type=str, default="objaverse")
args = parser.parse_args()

OBJECT_ARGS = {"shapenetcore": dict(type="obj")}
obj_args = OBJECT_ARGS.pop(args.dataset, {})
dataset = datasets[args.dataset]()

statistics = (
    dataset.statistics.drop(columns=["valid"])
    if dataset.statistics is not None and not args.demo
    else pd.DataFrame(columns=["meshCount", "uvCount", "diffuseCount", "uvScore"])
)
statistics.index.name = "uid"

# Loading the paths only on the root node, since it's an heavy operation
paths = comm.bcast(
    (
        [
            (k, v)
            for k, v in tqdm(dataset.paths.items(), desc="Loading paths")
            if Path(v).exists() and k not in statistics.index
        ]
        if rank == 0
        else None
    ),
    root=0,
)[rank::size]

if args.demo:
    paths = paths[:4]
if rank == 0:
    log("Loaded", len(statistics), "statistics of", len(dataset.paths))
    log("Each task has to process", len(paths) - len(statistics) // size, "objects")

for uid, path in tqdm(paths) if rank == 0 else paths:
    if (obj := dataset[dict(uid=uid, silent=True, **obj_args)]) is not None:
        statistics.loc[uid] = [
            len(obj.meshes),
            obj.mesh_stats["uv_count"] if obj.has_one_mesh else None,
            obj.mesh_stats["texture_count"] if obj.has_one_mesh else None,
            obj.uv_score if obj.has_one_mesh else None,
        ]
# Synchronize the partial results to the root task
log("Rank", rank, "has done processing statistics.")
all_statistics: list[pd.DataFrame] = comm.gather(statistics, root=0)
if rank == 0:
    concatenated = pd.concat(all_statistics)
    final_statistics = concatenated[~concatenated.index.duplicated(keep="first")].sort_index()
    final_statistics = final_statistics.astype(
        {"meshCount": int, "uvCount": "Int64", "diffuseCount": "Int64", "uvScore": "Float64"}
    )
    if args.demo:
        log(final_statistics)
    else:
        final_statistics.to_parquet(datasets[args.dataset].DATASET_DIR / "statistics.parquet")
