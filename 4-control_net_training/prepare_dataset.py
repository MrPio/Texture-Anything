import base64
import sys
from pathlib import Path
import zlib
from tqdm import tqdm
from shutil import rmtree
import pandas as pd
from numpy.random import choice
import random
from PIL import Image
import numpy as np

FILE_DIR = Path(__file__).resolve().parent
sys.path.append(str(FILE_DIR.parent))
from src import datasets, Dataset3D, LaplacianFilter, cprint

MAX_DATASET_SIZE = None  # 1_000
TARGET_SIZE = (512, 512)
DATASETS: list[Dataset3D] = [_() for _ in datasets.values()]
BLACK_TRESHOLD = 0.66
OUTPUT_PATH = Path(FILE_DIR / "dataset").resolve()
VALIDATION_UIDS = [
    # "1305b9266d38eb4d9f818dd0aa1a251",  # "0adf456c59094a3da23329a6d27cb239",
    # "1de679dd26d8c69cae44c65a6d0f0732",  # "3b15c410f87f42daa7e8cb5b5f74e3f1",
]
TESTSET_SIZE = 50  # diffusers train script doesn't use a test set
TEST_UIDS = (FILE_DIR / "testset_uids.txt").read_text().split("\n")


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


conv_filter = LaplacianFilter()
train_dir, test_dir, valid_dir = "train", "test", "validation"
for dir in [train_dir, test_dir, valid_dir]:
    rmtree(OUTPUT_PATH / dir, ignore_errors=True)
    for folder in ["diffuse", "uv"]:
        (OUTPUT_PATH / dir / folder).mkdir(parents=True, exist_ok=True)

metadata = pd.DataFrame(columns=["uv_file_name", "diffuse_file_name", "caption", "mask", "split"])
for dataset in DATASETS:
    uids = list(dataset.triplets)
    if MAX_DATASET_SIZE:
        uids = list(set(uids[:MAX_DATASET_SIZE] + VALIDATION_UIDS))
    test_uids = TEST_UIDS if TEST_UIDS else choice(uids, size=TESTSET_SIZE, replace=False)
    cprint(f"yellow:{dataset.__class__.__name__}", "has", len(uids))

    uv_paths = {x.stem: x for x in (dataset.DATASET_DIR / "uv").glob("*") if x.suffix in dataset.IMG_EXT}
    diffuse_paths = {x.stem: x for x in (dataset.DATASET_DIR / "diffuse").glob("*") if x.suffix in dataset.IMG_EXT}
    mask_paths = {x.stem: x for x in (dataset.DATASET_DIR / "mask").glob("*.npy")}
    captions = dataset.captions

    for uid in tqdm(uids):
        try:
            split = test_dir if uid in test_uids else valid_dir if uid in VALIDATION_UIDS else train_dir
            diffuse = Image.open(diffuse_paths[uid])
            if diffuse.size[0] != diffuse.size[1]:
                continue

            diffuse = diffuse.resize(TARGET_SIZE, Image.NEAREST)
            uv = Image.open(uv_paths[uid]).resize(TARGET_SIZE, Image.NEAREST)

            mask_packed = np.load(mask_paths[uid])
            mask = np.unpackbits(mask_packed)
            mask = mask.reshape(int(len(mask) ** 0.5), -1)  # Assuming it's square
            mask_compressed = zlib.compress(mask_packed.tobytes())
            mask_base64 = base64.b64encode(mask_compressed).decode("utf-8")
            # masked_diffuse = mask_image(diffuse, mask)

            # Quality checks
            if (
                np.mean(mask == 0) > BLACK_TRESHOLD
                or conv_filter.is_jagged(diffuse, dataset.__class__)
                or conv_filter.is_plain(diffuse)
            ):
                continue
        except Exception as e:
            cprint("Skipping UID=", uid, "because of:")
            print(e)
            continue

        diffuse.save(OUTPUT_PATH / split / "diffuse" / f"{uid}.png")
        uv.save(OUTPUT_PATH / split / "uv" / f"{uid}.png")

        metadata.loc[-1] = [
            "uv/" + uv_paths[uid].name,
            "diffuse/" + diffuse_paths[uid].name,
            captions[uid.split("_")[0]],
            mask_base64,
            split,
        ]
        metadata.index += 1

for dir in [train_dir, test_dir, valid_dir]:
    metadata[metadata["split"] == dir].drop(columns=["split"]).to_csv(OUTPUT_PATH / dir / "metadata.csv", index=False)

size = sum(f.stat().st_size for f in (OUTPUT_PATH).glob("**/*") if f.is_file())
cprint("Dataset size:", size // 1024**2, "green:MiB")
