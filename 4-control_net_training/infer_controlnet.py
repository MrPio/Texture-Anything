"""
Predict 20 samples from Objaverse dataset. CWD-independent.

Based on: https://github.com/huggingface/diffusers/tree/main/examples/controlnet
"""

from pathlib import Path
import sys

ROOT_PATH = Path(__file__).parents[1]
CACHE_DIR = ROOT_PATH / ".huggingface"

SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CNET_MODEL = (
    ROOT_PATH / "4-control_net_training/output/SD1.5_CNmlsd_128bs_5e-6lr_13k/checkpoint-500/controlnet"
)  # lllyasviel/sd-controlnet-mlsd
OUTPUT_FOLDER = "SD1.5_CNmlsd_128bs_5e-6lr_13k_500s"
INVERT_UV = True
UIDS = [
    "3603cf85c49e4323a93b62db0258b36f",
    "2947627cf0ba404e89e7d16d172d4c0e",
    "2b0e069216f046a7a93583660b913644",
    "07158256ea464c8f9654c1af3a30b7de",
    "d4a2a7a15c0c453ea60d7787fed42ff7",
    "8e40f131cd6e4020b20c44083675f358",
    "96d6e60d8f26409bbfddb0fced84a374",
    "574405e702814309bae45ca661b66718",
    "3ff4c879b6ca411dae8ddfa2bff2a223",
    "4ec49279363b407089d0d9e0edea9f4a",
    "9d1343495429463d9bc324ae1ed1e0c1",
    "bdf7f484270744a5802cf64cb8635936",
    "e340722a9f8043f7b45f1b5c8ebf3181",
    "51936b099dab41dabfecb1fc283695f9",
    "68e89a16259742788160a2b73f982fd8",
    "ae0965c4a43f4063b80889465081e4f2",
    "8767a5bd7dda4f2d803a796321e5484d",
    "92e0e8a10c234221a6e3fbdb1d25e99e",
    "16cd7b9ba74f44239d1987cb1546396e",
    "e1c5e66f5b50407f8d453813a6b81220",
]
output_path = ROOT_PATH / "4-control_net_training" / "infer" / OUTPUT_FOLDER
if output_path.exists():
    raise "The output path already exists."

import PIL
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
import torch
from tqdm import tqdm

sys.path.insert(0, str(ROOT_PATH))
from src import *

controlnet = ControlNetModel.from_pretrained(
    pretrained_model_name_or_path=str(CNET_MODEL),
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,
    local_files_only=True,
)
pipe = (
    StableDiffusionXLControlNetPipeline if "xl" in SD_MODEL else StableDiffusionControlNetPipeline
).from_pretrained(
    pretrained_model_name_or_path=SD_MODEL,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    cache_dir=CACHE_DIR,
    local_files_only=True,
    safety_checker=None,
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()

dataset = ObjaverseDataset3D()
uv_paths = {x.stem: x for x in (dataset.DATASET_PATH / "uv").glob("*") if x.suffix in dataset.IMG_EXT}
captions = dataset.captions
generator = torch.manual_seed(0)
output_path.mkdir(parents=True, exist_ok=True)

for uid in tqdm(UIDS):
    control_image = PIL.Image.open(uv_paths[uid]).convert("RGB")
    if INVERT_UV:
        control_image = PIL.ImageOps.invert(control_image)
    prompt = captions[uid]
    image = pipe(prompt, num_inference_steps=20, generator=generator, image=control_image).images[0]
    image.save(output_path / f"{uid}.png")
