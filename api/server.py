# uvicorn server:app --host 0.0.0.0 --port 7860

import hashlib
import io
import random
import torch
from pathlib import Path
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image, ImageOps
import gradio as gr
from huggingface_hub import login
import os
import dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Union
from PIL import Image, ImageDraw
import base64
import io
dotenv.load_dotenv()

login(os.environ["HF_TOKEN"])

# ---- Model loading ----
CACHE_DIR = Path(__file__).parent/".cache"
CNET_MODEL = "MrPio/Texture-Anything_CNet-SD15"
SD_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"

controlnet = ControlNetModel.from_pretrained(
    CNET_MODEL, cache_dir=CACHE_DIR, torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    SD_MODEL,
    controlnet=controlnet,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,
    safety_checker=None,
)

# speed & memory optimizations
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()  # if xformers installed
pipe.enable_model_cpu_offload()


def pil2hash(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    return hashlib.sha256(image_bytes).hexdigest()


def caption2hash(caption: str) -> str:
    return hashlib.sha256(caption.encode()).hexdigest()


# ---- Inference function ----
def infer(caption: str, condition_image: Image.Image, steps: int = 20, seed: int = -1, invert: bool = True):
    print("Loading condition image")
    img = condition_image.convert("RGB")
    if seed==-1:
        seed=random.randint(0,1e6)
    if invert:
        img = ImageOps.invert(img)
        print("Condition image inverted")
    cache_file = Path(f"inferences/{pil2hash(img)}_{caption2hash(caption)}_{steps}_{seed}.png").resolve()
    cache_file.parent.mkdir(exist_ok=True, parents=True)
    if cache_file.exists():
        return Image.open(cache_file)

    print("Starting generation...")
    generator = torch.manual_seed(seed)
    output = pipe(prompt=caption, image=img, num_inference_steps=steps, generator=generator).images[0]
    print("Caching result...")
    output.save(cache_file)
    return output

app = FastAPI()

class ImageData(BaseModel):
    name: str
    data: str  # base64 string

class InputPayload(BaseModel):
    data: List[Union[str, ImageData, int, float, bool]]

@app.post("/predict")
async def run_predict(payload: InputPayload):
    # Parse inputs
    txt = payload.data[0]
    cond = payload.data[1]
    steps = int(payload.data[2])
    seed = int(payload.data[3])
    inv = bool(payload.data[4])

    # Decode base64 image
    image_bytes = base64.b64decode(cond.data)
    image = Image.open(io.BytesIO(image_bytes))

    output=infer(txt, image, steps, seed, inv)

    # Encode output image to base64
    buffer = io.BytesIO()
    output.save(buffer, format="PNG")
    output_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {"data": [output_base64]}