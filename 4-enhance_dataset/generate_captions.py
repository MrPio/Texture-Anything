#!/usr/bin/env python
"""
Distributed 3D Render Captioning Script using BLIP-2 and MPI.

Usage:
    srun -n 32 --ntasks-per-node=4 --mem=32G --gpus-per-task=1 python generate_captions.py
    (Took ~5min on 32 A100)

Arguments:
    --input   (str):  Path to directory containing input JPEG images.
    --output  (str):  Path to save the generated captions JSON.
    --demo          : Optional flag to run in demo mode (processes only a few images).

Environment:
    Requires access to CUDA-compatible GPUs and pre-cached models via Hugging Face.

Author:
    Valerio Morelli - 2025-05-11
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
from mpi4py import MPI
from transformers import Blip2Processor, Blip2ForConditionalGeneration

ROOT_PATH = Path(__file__).parent.parent.resolve()
CACHE_PATH = ROOT_PATH / ".huggingface"

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/dataset/objaverse/render/")
    parser.add_argument("--output", type=str, default="data/dataset/objaverse/caption/")
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()


args = parse_args()
paths = list((ROOT_PATH / args.input).glob("*.jpg"))[rank::size]
if args.demo:
    paths = paths[:6]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Blip2Processor.from_pretrained(
    "Salesforce/blip2-flan-t5-xxl",
    cache_dir=CACHE_PATH,
    use_fast=True,
)
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xxl",
    cache_dir=CACHE_PATH,
).to(device)
model.eval()

captions = {}
# instruction = "Describe this rendered 3D model in rich detail, focusing on its shape, materials, textures, and colors."
instruction = (
    "Describe the object in this 3D render in rich detail.  "
    "Describe how it is made and all of its properties.  "
    "Focus on the object itself and not on the background.  "
    "First identify its overall shape and proportions.  "
    "Then discuss the materials and the textures.  "
    "Finally, mention any color nuances or surface details you notice."
)
gen_kwargs = {
    # "max_new_tokens": 128,
    "num_beams": 5,
    "length_penalty": 0.8,
    "early_stopping": True,
    # "do_sample": True,
    # "top_p": 0.9,
    # "temperature": 1.0,
}

for p in tqdm(paths) if rank == 0 else paths:
    raw_image = Image.open(p).convert("RGB")
    inputs = processor(
        images=raw_image,
        text=instruction,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )
    outputs = model.generate(
        **inputs.to(device),
        **gen_kwargs,
    )
    caption = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

    captions[Path(p).stem] = caption.lower().strip()

all_captions = comm.gather(captions, root=0)
if rank == 0:
    all_captions = {k: v for d in all_captions for k, v in d.items()}
    with open(ROOT_PATH / args.output / "captions.json", "w") as f:
        json.dump(all_captions, f, indent=4)
