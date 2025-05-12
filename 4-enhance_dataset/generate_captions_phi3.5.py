#!/usr/bin/env python
"""
Distributed 3D Render Captioning Script using Phi 3.5 Vision Instruct and MPI.

Usage:
    srun -n 32 --ntasks-per-node=4 --mem=32G --gpus-per-task=1 python generate_captions_phi3.5.py
    (Takes ~2s/it on A100)

Arguments:
    --input   (str):  Path to directory containing input images.
    --output  (str):  Path to the JSON file.

Environment:
    Requires access to CUDA-compatible GPUs and pre-cached models via Hugging Face.

Based on:
    https://huggingface.co/microsoft/Phi-3.5-vision-instruct

Author:
    Valerio Morelli - 2025-05-12
"""
import argparse
import json
from pathlib import Path

from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from mpi4py import MPI


ROOT_PATH = Path(__file__).parent.parent.resolve()
CACHE_PATH = ROOT_PATH / ".huggingface"

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/dataset/objaverse/render/")
    parser.add_argument("--output", type=str, default="data/dataset/objaverse/caption/captions_phi35.json")
    return parser.parse_args()


args = parse_args()
paths = sorted(p for p in (ROOT_PATH / args.input).glob("*") if p.suffix in {".jpg", ".png"})[rank::size]

model_id = "microsoft/Phi-3.5-vision-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation="eager",  # 'flash_attention_2'
    cache_dir=CACHE_PATH,
)
model.eval()
processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    num_crops=16,
    cache_dir=CACHE_PATH,
)

captions = {}
placeholder = f"<|image_{1}|>\n"
messages = [
    {
        "role": "user",
        "content": placeholder
        + "Generate a Stable Diffusion caption of about 30 words for the object in this 3D render. Discuss the shape of the object, but also the materials, the textures and the colors. Ignore the background. Describe the texture properties and colors in details.",
    },
]
generation_args = {
    "max_new_tokens": 64,
    "temperature": 0.2,
    "do_sample": False,
}
prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

for p in tqdm(paths) if rank == 0 else paths:
    image = Image.open(p)
    inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")
    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]  # remove input tokens
    caption = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    captions[Path(p).stem] = caption.lower().strip()

# Syncronize the partial results to the root task
all_captions = comm.gather(captions, root=0)
if rank == 0:
    all_captions = {k: v for d in all_captions for k, v in d.items()}
    with open(ROOT_PATH / args.output, "w") as f:
        json.dump(all_captions, f, indent=4)
