import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path("..").resolve()))
from src import *

dataset = ObjaverseDataset3D()

TESTSET_DIR = Path("./dataset/test")
gt_dir = TESTSET_DIR / "diffuse"
vanilla_dir = Path("tests/lllyasviel-sd-controlnet-mlsd")
trained_dir = Path("tests/SD1.5_CNmlsd_64bs_1e-5lr_8k_combined-loss_7000s")
trained_dir2 = Path("tests/SDxl_CN_32bs_1e-5lr_8k_latent-loss_7000s")

(Path("renderings") / "gt").mkdir(exist_ok=True)
(Path("renderings") / vanilla_dir.parts[-1]).mkdir(exist_ok=True)
(Path("renderings") / trained_dir.parts[-1]).mkdir(exist_ok=True)
(Path("renderings") / trained_dir2.parts[-1]).mkdir(exist_ok=True)

files = list(trained_dir.glob("*.png"))
sample_files = files[:]

for i, file in enumerate(sample_files):
    uid = Path(file).stem
    try:
        obj = dataset[dict(uid=uid, preprocess=True)]
        
        if not (Path("renderings") / "gt" / f"{uid}.png").exists():
            obj.change_texture(str(gt_dir / f"{uid}.png"))  # To remove any image other than diffuse
            obj.render(views=1)[0].save(Path("renderings") / "gt" / f"{uid}.png")

        if not (vanilla_dir / f"{uid}.png").exists() or not (trained_dir / f"{uid}.png").exists() or not (trained_dir2 / f"{uid}.png").exists():
            continue
        if not (Path("renderings") / vanilla_dir.parts[-1] / f"{uid}.png").exists():
            obj.change_texture(str(vanilla_dir / f"{uid}.png"))
            obj.render(views=1)[0].save(Path("renderings") / vanilla_dir.parts[-1] / f"{uid}.png")

        if not (Path("renderings") / trained_dir.parts[-1] / f"{uid}.png").exists():
            obj.change_texture(str(trained_dir / f"{uid}.png"))
            obj.render(views=1)[0].save(Path("renderings") / trained_dir.parts[-1] / f"{uid}.png")

        if not (Path("renderings") / trained_dir2.parts[-1] / f"{uid}.png").exists():
            obj.change_texture(str(trained_dir2 / f"{uid}.png"))
            obj.render(views=1)[0].save(Path("renderings") / trained_dir2.parts[-1] / f"{uid}.png")
    except:
        continue
