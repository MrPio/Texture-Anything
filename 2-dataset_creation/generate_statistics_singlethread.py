#!/usr/bin/env python
"""
Generate statistics on OBJ files.
Please make sure that you have downloaded the objects relative to the UIDs in the annotations table before running this script.
This script is CWD-independent.

Usage:
    $ python generate_statistics_singlethread.py

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
sys.path.append(str(ROOT_DIR))
from src import *

warnings.simplefilter(action="ignore", category=FutureWarning)
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
paths = [
    (k, v)
    for k, v in tqdm(dataset.paths.items(), desc="Loading paths")
    if Path(v).exists() and k not in statistics.index
]

if args.demo:
    paths = paths[:4]
log("Loaded", len(statistics), "statistics of", len(dataset.paths))
log("Each task has to process", len(paths) - len(statistics), "objects")

for uid, path in tqdm(paths):
    if (obj := dataset[dict(uid=uid, silent=True, **obj_args)]) is not None:
        statistics.loc[uid] = [
            len(obj.meshes),
            obj.mesh_stats["uv_count"] if obj.has_one_mesh else None,
            obj.mesh_stats["texture_count"] if obj.has_one_mesh else None,
            obj.uv_score if obj.has_one_mesh else None,
        ]
final_statistics = statistics[~statistics.index.duplicated(keep="first")].sort_index()
final_statistics = final_statistics.astype(
    {
        "meshCount": int,
        "uvCount": "Int64",
        "diffuseCount": "Int64",
        "uvScore": "Float64",
    }
)
if args.demo:
    log(final_statistics)
else:
    final_statistics.to_parquet(datasets[args.dataset].DATASET_DIR / "statistics.parquet")
