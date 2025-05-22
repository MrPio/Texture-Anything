from pathlib import Path

ROOT_DIR = Path(__file__).parents[1]
CACHE_DIR = ROOT_DIR / ".huggingface"

# SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DREAM_BOOTH = None  # "texture_hell.safetensors" # Place https://civitai.com/models/43468/texture-hell in CACHE_DIR
OUTPUT_FOLDER = "SDxl_vanilla"
PROMPTS = [
    "Texture of a rusty barrel",
    "Texture of a brand new dark coocking pan",
    "Texture of a combat asian knife",
    "A texture of a smartphone, ios, round corners.",
    "A texture of a birch wooden crate.",
    "texture, wood, wall, man made, old, worn, uneven, painted, planks",
    "texture, rock, cave, uneven, weathered, rough, moss",
    "texture, cobblestone, mud, old, uneven, ground, outdoors",
]
output_path = ROOT_DIR / "4-control_net_training" / "tests" / OUTPUT_FOLDER
if output_path.exists():
    raise "The output path already exists."

import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DPMSolverMultistepScheduler, UniPCMultistepScheduler
from tqdm import tqdm

if DREAM_BOOTH:
    pipe = StableDiffusionPipeline.from_single_file(
        pretrained_model_link_or_path=str(CACHE_DIR / DREAM_BOOTH),
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )
else:
    # use DiffusionPipeline if SDxl
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path=SD_MODEL,
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    )
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

generator = torch.manual_seed(0)
output_path.mkdir(parents=True)

for prompt in tqdm(PROMPTS):
    image = pipe(prompt, num_inference_steps=20, generator=generator).images[0]
    image.save(output_path / f"{prompt}.png")
