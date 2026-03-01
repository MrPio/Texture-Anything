import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append("..")
from src import *

dataset = ObjaverseDataset3D()

TESTSET_DIR = Path("dataset/test")
TEST_DIR = Path("tests")
dirs = {
    "gt": TESTSET_DIR / "diffuse",
    "sd15_mlsd": TEST_DIR / "sd15_mlsd",
    "sd15_ours": TEST_DIR / "sd15_ours",
    "sdxl_ours": TEST_DIR / "sdxl_ours",
    "sdxl_mlsd_llite": TEST_DIR / "sdxl_mlsd_llite",
    "sdxl_ours_llite": TEST_DIR / "sdxl_ours_llite",
}
texture_dir = "sdxl_ours_llite"

out_dir = mkdir(f"renderings/{texture_dir}")
dir = dirs[texture_dir]

for file in tqdm(list(dir.glob("*.png"))):
    uid = file.stem.split("_")[0]
    obj = dataset[dict(uid=uid, preprocess=True)]
    if obj and not (out_dir / f"{uid}_0.png").exists():
        obj.change_texture(file)  # To remove any image other than diffuse
        for i, view in enumerate(obj.render(views=3)):
            view.save(out_dir / f"{uid}_{i}.png")
