from functools import cached_property
import os
from zipfile import ZipFile
import pandas as pd
import requests
from tqdm import tqdm
from .dataset3d import Dataset3D
from ..blender.object3d.shapenetcore_object3d import ShapeNetCoreObject3D


class ShapeNetCoreDataset3D(Dataset3D):
    DATASET_PATH = Dataset3D.DATASET_PATH / "shapenetcore"
    # Sorted by file size, largest to smallest
    CATEGORIES = [
        "02958343",
        "02691156",
        "03001627",
        "04379243",
        "04530566",
        "04256520",
        "04090263",
        "03636649",
        "02924116",
        "03467517",
        "03790512",
        "03691459",
        "02933112",
        "02828884",
        "04468005",
        "03991062",
        "04401088",
        "03211117",
        "02992529",
        "03642806",
        "02808440",
        "03046257",
        "03593526",
        "03325088",
        "02876657",
        "03928116",
        "03948459",
        "04225987",
        "04330267",
        "02871439",
        "03337140",
        "02818832",
        "03624134",
        "02747177",
        "04460130",
        "02954340",
        "03513137",
        "03761084",
        "02942699",
        "04004475",
        "02801938",
        "02773838",
        "03938244",
        "04554684",
        "03797390",
        "03261776",
        "04099429",
        "02880940",
        "02946921",
        "03085013",
        "03759954",
        "03710193",
        "03207941",
        "04074963",
        "02843684",
    ]

    def __init__(self):
        super().__init__("shapenetcore", ShapeNetCoreObject3D)

    @cached_property
    def annotations(self) -> pd.DataFrame | None:
        return None

    @cached_property
    def paths(self) -> dict[str, str]:
        return {
            m.name: str(m)
            for p in tqdm((ShapeNetCoreDataset3D.DATASET_PATH / "objects").iterdir())
            if p.is_dir()
            for m in p.iterdir()
        }

    def download(self, first=-1) -> None:
        """Download the ShapeNetCore dataset

        Args:
            first (int): Number of categories to process (use -1 for all).
        """
        chunk_size = 16_384
        base_url = "https://huggingface.co/datasets/ShapeNet/ShapeNetCore/resolve/main/"
        headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

        selected_cats = self.CATEGORIES[::-1]
        if first > 0:
            selected_cats = selected_cats[:first]

        for i, cat in enumerate(selected_cats):
            print(i, "/", len(self.CATEGORIES))
            response = requests.get(f"{base_url}{cat}.zip", headers=headers, stream=True)
            zip_path = self.DATASET_PATH / "objects" / f"{cat}.zip"
            total_size = int(response.headers.get("content-length", 0))

            with open(zip_path, "wb") as f, tqdm(
                desc=f"Downloading {cat}.zip",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            with ZipFile(zip_path, "r") as zf:
                zf.extractall(path=self.DATASET_PATH / "objects")
            zip_path.unlink()
