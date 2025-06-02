import pandas as pd
import seaborn as sns
import requests
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sns.set_style("darkgrid")
sns.set_context("notebook", font_scale=1.15)
tqdm.pandas()

MIN_SIZE = 20_000  # bytes
annotations = pd.read_parquet("../dataset/objaverse/thumbnails.parquet")


def check_thumbnails(row) -> str | None:
    def get_size(url) -> int:
        try:
            response = requests.head(url, timeout=5)
            if response.status_code == 200 and "Content-Length" in response.headers:
                return int(response.headers["Content-Length"])
            return 0
        except requests.RequestException:
            return None

    if not isinstance(row["thumbnails"], np.ndarray):
        return None

    for thumbnail in row["thumbnails"][::-1] if "x" in row["thumbnails"][0][-8] else row["thumbnails"]:
        if (size := get_size(thumbnail)) is not None and size > MIN_SIZE:
            return thumbnail
    return None


def parallel_apply(df, func, max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        rows = [row for _, row in df.iterrows()]
        results = list(tqdm(executor.map(func, rows), total=len(rows), desc="Processing"))
    return results


thumbnails = parallel_apply(annotations, check_thumbnails, max_workers=48)

annotations["thumbnail"] = thumbnails
annotations.drop(columns=["thumbnails"], inplace=True)

annotations_filtered = annotations[annotations["thumbnail"].notna()]
annotations_filtered.to_parquet("../dataset/objaverse/thumbnails_checked.parquet", index=False)
