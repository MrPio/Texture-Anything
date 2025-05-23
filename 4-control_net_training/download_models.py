import sys
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline
from diffusers import ControlNetModel

ROOT_DIR = Path(__file__).parents[1]
DOWNLOAD_PATH = ROOT_DIR / ".huggingface"
sys.path.insert(0, str(ROOT_DIR))
from src import log

# SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
# SD_MODEL = "stabilityai/stable-diffusion-2-1"
SD_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
CNET_MODEL = "lllyasviel/sd-controlnet-mlsd"

log("Downloading", f"blue:{SD_MODEL}", "to", f"red:{DOWNLOAD_PATH}")
StableDiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path=SD_MODEL,
    torch_dtype=torch.float16,
    cache_dir=DOWNLOAD_PATH,
)

log("Downloading", f"blue:{CNET_MODEL}", "to", f"red:{DOWNLOAD_PATH}")
ControlNetModel.from_pretrained(
    pretrained_model_name_or_path=CNET_MODEL,
    cache_dir=DOWNLOAD_PATH,
)
