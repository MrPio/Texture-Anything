#!/usr/bin/env python
"""
Distributed 3D Render Captioning Script using Phi 3.5 Vision Instruct and MPI.

Usage:
    $ srun -n 32 --ntasks-per-node=4 --mem=32G --gpus-per-task=1 --partition=boost_usr_prod python generate_captions_phi3.5.py
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
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from src import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="dataset/objaverse/render/")
    parser.add_argument("--output", type=str, default="dataset/objaverse/caption/captions_phi35.json")
    parser.add_argument("-d", "--demo", action="store_true")
    return parser.parse_args()


args = parse_args()

CACHE_PATH = ROOT_DIR / ".huggingface"
OUTPUT_PATH = ROOT_DIR / args.output

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

captions = json.load(open(OUTPUT_PATH)) if OUTPUT_PATH.exists() else {}
paths = sorted(
    p for p in (ROOT_DIR / args.input).glob("*") if p.suffix in {".jpg", ".png"} and p.stem not in captions
)[rank::size]
if args.demo:
    paths = paths[:4]
if rank == 0:
    log("Loaded", f"blue:{len(captions):,}", "captions of", f"{len(paths):,}")
    log("Each task has to generate", f"red:{len(paths) // size:,}", "captions")

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
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_captions, f, indent=4)
    if args.demo:
        log("Now you have", f"blue:{len(captions):,}", "captions")
