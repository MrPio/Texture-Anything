import sys
from pathlib import Path
from tqdm import tqdm
from shutil import copy, rmtree
import pandas as pd
from datasets import load_dataset
from numpy.random import choice
import random
from PIL import Image
import numpy as np

FILE_DIR = Path(__file__).resolve().parent
sys.path.append(str(FILE_DIR.parent))
from src import *

MAX_DATASET_SIZE = None  # 1_000
TEST_SET_RATIO = 0.01  # diffusers train script doesn't use a test set
OUTPUT_PATH = Path(FILE_DIR / "dataset").resolve()
VALIDATION_UIDS = [
    "0adf456c59094a3da23329a6d27cb239",
    "3b15c410f87f42daa7e8cb5b5f74e3f1",
]

datasets: list[Dataset3D] = [ObjaverseDataset3D()]


def mask_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    image = image.convert("RGB")
    mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_pil = mask_pil.resize(image.size[:2], resample=Image.NEAREST)
    mask = np.array(mask_pil)
    mask = np.stack([mask] * 3, axis=-1)
    return Image.fromarray(np.where(mask, image, 0))


def is_black(img: Image.Image, threshold=0.9):
    arr = np.array(img.convert("RGB"))
    black_pixels = np.all(arr == 0, axis=-1)
    ratio = np.mean(black_pixels)
    return ratio > threshold


train_dir, test_dir, valid_dir = "train", "test", "validation"
for dir in [train_dir, test_dir, valid_dir]:
    rmtree(OUTPUT_PATH / dir, ignore_errors=True)
    for folder in ["diffuse", "uv"]:
        (OUTPUT_PATH / dir / folder).mkdir(parents=True, exist_ok=True)

metadata = pd.DataFrame(columns=["uv_file_name", "diffuse_file_name", "caption", "split"])
for dataset in datasets:
    valid_uids = dataset.statistics["valid"].index
    avail_uids = dataset.triplets
    uids = list(avail_uids.intersection(valid_uids))
    if MAX_DATASET_SIZE:
        uids = list(set(uids[:MAX_DATASET_SIZE] + VALIDATION_UIDS))
    test_uids = choice(uids, size=int(len(uids) * TEST_SET_RATIO), replace=False)
    cprint(f"yellow:{dataset.__class__.__name__}", "has", len(avail_uids), "uids,", len(uids), "of them are valid.")

    uv_paths = {x.stem: x for x in (dataset.DATASET_DIR / "uv").glob("*") if x.suffix in dataset.IMG_EXT}
    diffuse_paths = {x.stem: x for x in (dataset.DATASET_DIR / "diffuse").glob("*") if x.suffix in dataset.IMG_EXT}
    mask_paths = {x.stem: x for x in (dataset.DATASET_DIR / "mask").glob("*.npy")}
    captions = dataset.captions

    for uid in tqdm(uids):
        split = test_dir if uid in test_uids else valid_dir if uid in VALIDATION_UIDS else train_dir
        copy(uv_paths[uid], OUTPUT_PATH / split / "uv")
        diffuse = Image.open(diffuse_paths[uid])
        if diffuse.size[0] != diffuse.size[1]:
            continue
        mask = np.unpackbits(np.load(mask_paths[uid])).reshape(1024, 1024)
        masked_diffuse = mask_image(diffuse, mask)
        if is_black(masked_diffuse, threshold=0.66):
            continue
        masked_diffuse.save(OUTPUT_PATH / split / "diffuse" / f"{uid}.png")
        metadata.loc[-1] = [
            "uv/" + uv_paths[uid].name,
            "diffuse/" + diffuse_paths[uid].name,
            captions[uid],
            split,
        ]
        metadata.index += 1

for dir in [train_dir, test_dir, valid_dir]:
    metadata[metadata["split"] == dir].drop(columns=["split"]).to_csv(OUTPUT_PATH / dir / "metadata.csv", index=False)

size = sum(f.stat().st_size for f in (OUTPUT_PATH).glob("**/*") if f.is_file())
cprint("Dataset size:", size // 1024**2, "green:MiB")
