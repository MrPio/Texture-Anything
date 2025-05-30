#!/usr/bin/env python
"""
Distributed 3D Render Captioning Script using Phi 3.5 Vision Instruct and MPI.

Usage:
    $ srun -n 8 --ntasks-per-node=4 --mem=32G --gpus-per-task=1 --partition=boost_usr_prod python generate_captions_phi3.5.py
    (Takes ~2s/it on A100)

Arguments:
    --input   (str):  Path to directory containing input images.
    --output  (str):  Path to the JSON file.

Based on
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
from torch.cuda import is_available as is_cuda_available
from random import randint
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from src import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="dataset/shapenetcore/render/")
    parser.add_argument("--output", type=str, default="dataset/shapenetcore/caption/captions_phi35.json")
    parser.add_argument("--samples", type=int, default=3, help="The number of renderings for each sample.")
    parser.add_argument("-d", "--demo", action="store_true")
    return parser.parse_args()


args = parse_args()
CACHE_PATH = ROOT_DIR / ".huggingface"
OUTPUT_DIR = ROOT_DIR / args.output
OUTPUT_DIR.parent.mkdir(exist_ok=True, parents=True)

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

captions = {}
if OUTPUT_DIR.parent.exists():
    for partial in OUTPUT_DIR.parent.glob("*.json"):
        captions |= json.load(open(partial))

processed_uids = list(captions.keys())
input_dir = ROOT_DIR / args.input
paths = sorted(p for p in input_dir.glob("*") if not any(uid in p.stem for uid in processed_uids))

# [0_0.png, 0_1.png, 1_0.png, 1_1.png] --> [[0_0.png, 0_1.png], [1_0.png, 1_1.png]]
paths = [paths[i : i + args.samples] for i in range(0, len(paths), args.samples)]

if args.demo:
    paths = paths[: args.samples * 4]
if rank == 0:
    log("Already processed", f"blue:{len(captions):,}", "captions of a total of", f"{len(paths)+len(captions):,}")
    log("Each task has", f"red:{len(paths) // size:,}", "captions, each observing", args.samples, "renderings")
paths = paths[rank::size]

device = "cuda" if is_cuda_available() else "cpu"
model_id = "microsoft/Phi-3.5-vision-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device,
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
placeholder = "".join(f"<|image_{i+1}|>\n" for i in range(args.samples))
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
cache_suffix = randint(1e6, 1e7 - 1)

for i, renderings in enumerate(tqdm(paths, disable=rank != 0)):
    uid = renderings[0].stem.split("_")[0]
    images = [Image.open(r) for r in renderings]
    inputs = processor(prompt, images, return_tensors="pt").to(device)
    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]  # remove input tokens
    caption = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    captions[uid] = caption.lower().strip()
    if not args.demo and i % 10 == 0:
        with open(str(OUTPUT_DIR).replace(".json", f"_{rank}_{cache_suffix}.json"), "w") as f:
            json.dump(captions, f, indent=4)

# Syncronize the partial results to the root task
all_captions = comm.gather(captions, root=0)
if rank == 0:
    all_captions = {k: v for d in all_captions for k, v in d.items()}
    with open(OUTPUT_DIR, "w") as f:
        json.dump(all_captions, f, indent=4)
    if args.demo:
        log("Now you have", f"blue:{len(captions):,}", "captions")
