# %%
from utils import masked_mse_loss

import sys
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline, ControlNetModel, AutoencoderKL

ROOT_DIR = Path(__file__).parents[1]
CACHE_DIR = ROOT_DIR / ".huggingface"
sys.path.insert(0, str(ROOT_DIR))
from src import log

# %%
SD_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
# SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
# SD_MODEL = "stabilityai/stable-diffusion-2-1"
log("Downloading", f"blue:{SD_MODEL}", "to", f"red:{CACHE_DIR}")
StableDiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path=SD_MODEL,
    torch_dtype=torch.float16,
    cache_dir=CACHE_DIR,
)

# %%
CNET_MODEL = "lllyasviel/sd-controlnet-mlsd"
log("Downloading", f"blue:{CNET_MODEL}", "to", f"red:{CACHE_DIR}")
ControlNetModel.from_pretrained(
    pretrained_model_name_or_path=CNET_MODEL,
    cache_dir=CACHE_DIR,
)

# %%
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
log("Downloading", f"blue:{VAE_MODEL}", "to", f"red:{CACHE_DIR}")
AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path=VAE_MODEL,
    cache_dir=CACHE_DIR,
)
