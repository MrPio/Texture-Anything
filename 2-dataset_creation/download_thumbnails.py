from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import multiprocessing
from tqdm import tqdm
import sys
import PIL.Image as PILImage
import requests
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from src import *

MIN_RENDER_MEGAPIXEL = 100_000
dataset = ObjaverseDataset3D()
uids = {x.stem.split("_")[0] for x in (dataset.DATASET_DIR / "uv").glob("*")}
processed = {x.stem.split("_")[0] for x in (dataset.DATASET_DIR / "render").glob("*")}

log("You have already downloaded", len(processed), "thumbnails of", len(uids), "samples.")
uids = uids.difference(processed)
log("I will now download the missing", len(uids), "thumbnails")

thumbnails = pd.read_parquet(dataset.DATASET_DIR / "thumbnails_checked.parquet")["thumbnail"]
missing = uids.difference(thumbnails)
log("Unfortunately,", len(missing), "thumbnails are missing from thumbnails_checked.parquet")
open("missing_thumbnails.txt", "w").write("\n".join(missing))
uids = uids.difference(missing)
OUTPUT_DIR = dataset.DATASET_DIR / "render"


def download_thumbnail(uid):
    url = thumbnails.get(uid)
    if (OUTPUT_DIR / f"{uid}.png").exists() or url is None:
        return uid, None
    thumbnail = requests.get(url).content
    try:
        img = PILImage.open(BytesIO(thumbnail))
    except:
        img = None
    return uid, img


with multiprocessing.Pool(2) as pool:
    results = pool.imap_unordered(download_thumbnail, uids)
    for uid, img in tqdm(results, total=len(uids), desc="Downloading"):
        # Skip if the render resolution is less than MIN_RENDER_MEGAPIXEL
        if img is None or img.size[0] * img.size[1] < MIN_RENDER_MEGAPIXEL:
            continue
        img.save(OUTPUT_DIR / f"{uid}.png")

# with ThreadPoolExecutor(max_workers=2) as executor:
#     futures = {executor.submit(download_thumbnail, uid): uid for uid in uids}

#     for future in tqdm(as_completed(futures), total=len(uids), desc="Downloading"):
#         uid = futures[future]
#         img = future.result()
#         if img is None or img.size[0] * img.size[1] < MIN_RENDER_MEGAPIXEL:
#             continue
#         img.save(OUTPUT_DIR / f"{uid}.png")
