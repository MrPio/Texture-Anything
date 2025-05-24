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

MIN_RENDER_MEGAPIXEL = 200_000
dataset = ObjaverseDataset3D()
uids = [x.stem for x in (dataset.DATASET_DIR / "uv").glob("*")]


def download_thumbnail(uid):
    if (dataset.DATASET_DIR / "render" / f"{uid}.png").exists():
        return None
    thumbnail = requests.get(dataset.annotations.loc[uid]["thumbnail"]).content
    try:
        img = PILImage.open(BytesIO(thumbnail))
    except:
        img = None
    return uid, img


with multiprocessing.Pool(16) as pool:
    results = pool.imap_unordered(download_thumbnail, uids)
    for uid, img in tqdm(results, total=len(uids), desc="Downloading"):
        # Skip if the render resolution is less than MIN_RENDER_MEGAPIXEL
        if img is None or img.size[0] * img.size[1] < MIN_RENDER_MEGAPIXEL:
            continue
        img.save(dataset.DATASET_DIR / "render" / f"{uid}.png")
