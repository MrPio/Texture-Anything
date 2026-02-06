#!/usr/bin/env python
"""
Generate statistics on OBJ files.
Please make sure that you have downloaded the objects relative to the UIDs in the annotations table before running this script.
This script is CWD-independent.

Usage:
    $ python generate_statistics.py

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
parser.add_argument("--dataset", type=str, default="objaverse")
args = parser.parse_args()

OBJECT_ARGS = {"shapenetcore": dict(type="obj")}
obj_args = OBJECT_ARGS.pop(args.dataset, {})
dataset = datasets[args.dataset]()

statistics = (
    dataset.statistics.drop(columns=["valid"])
    if dataset.statistics is not None
    else pd.DataFrame(columns=["meshCount", "uvCount", "diffuseCount", "uvScore"])
)
statistics.index.name = "uid"
paths = [
    (k, v)
    for k, v in tqdm(dataset.paths.items(), desc="Loading paths")
    if Path(v).exists() and k not in statistics.index
]

processed_uids = set(statistics.index).intersection(dataset.paths.keys())
missing_uids = set(dataset.paths.keys()).difference(processed_uids)
log("Loaded", len(processed_uids), "statistics of", len(dataset.paths))
log("Each task has to process", len(missing_uids), "objects")


def save(stats):
    stats = stats[~stats.index.duplicated(keep="first")].sort_index()
    stats = stats.astype(
        {
            "meshCount": int,
            "uvCount": "Int64",
            "diffuseCount": "Int64",
            "uvScore": "Float64",
        }
    ).to_parquet(datasets[args.dataset].DATASET_DIR / "statistics.parquet")


for i, uid in tqdm(enumerate(missing_uids)):
    if (obj := dataset[dict(uid=uid, silent=True, **obj_args)]) is not None:
        has_one_mesh = len(obj.objects) == 1
        mesh = obj.objects[0].to_mesh() if has_one_mesh else None
        statistics.loc[uid] = [
            len(obj.objects),
            obj.mesh_stats(obj.objects[0])["uv_count"] if has_one_mesh else None,
            obj.mesh_stats(obj.objects[0])["texture_count"] if has_one_mesh else None,
            obj.uv_score(mesh) if has_one_mesh else None,
        ]
    if i % 100 == 0:
        save(statistics)
