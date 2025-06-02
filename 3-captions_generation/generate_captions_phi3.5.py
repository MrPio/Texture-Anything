#!/usr/bin/env python
"""
Distributed 3D Render Captioning Script using Phi 3.5 Vision Instruct and MPI.

Usage:
    $ srun -n 8 --ntasks-per-node=4 --mem=32G --gpus-per-task=1 --partition=boost_usr_prod python generate_captions_phi3.5.py
    (Takes ~0.5s/it on A100)

Arguments:
    --input   (str): Path to directory containing input images.
    --output  (str): Path to the JSON file.
    --samples (int): Number of renderings for each sample.
    --demo         : Wether to process just a small sample.

Based on
    https://huggingface.co/microsoft/Phi-3.5-vision-instruct

Author:
    Valerio Morelli - 2025-05-12
"""
import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from mpi4py import MPI
from torch.cuda import is_available as is_cuda_available
from random import randint
from transformers.image_processing_utils import BatchFeature
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from src import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="dataset/objaverse/render/")
    parser.add_argument("--output", type=str, default="dataset/objaverse/caption/captions_phi35.json")
    parser.add_argument("--samples", type=int, default=1, help="The number of renderings for each sample.")
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()


args = parse_args()
CACHE_PATH = ROOT_DIR / ".huggingface"
OUTPUT_DIR = ROOT_DIR / args.output
OUTPUT_DIR.parent.mkdir(exist_ok=True, parents=True)
BATCH_SIZE = 16
NUM_CROPS = 8
DTYPE = torch.bfloat16  # use bf16 (preferred on A100) or fp16

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

captions = {}
if OUTPUT_DIR.parent.exists():
    for partial in OUTPUT_DIR.parent.glob("*.json"):
        captions |= json.load(open(partial))

processed_uids = list(captions.keys())
input_dir = ROOT_DIR / args.input
paths = sorted(p for p in tqdm(input_dir.glob("*")) if p.stem.split("_")[0] not in processed_uids)

# [0_0.png, 0_1.png, 1_0.png, 1_1.png] --> [[0_0.png, 0_1.png], [1_0.png, 1_1.png]]
paths = [paths[i : i + args.samples] for i in range(0, len(paths), args.samples)]

if args.demo:
    paths = paths[: args.samples * BATCH_SIZE * 4]
if rank == 0:
    log("Already processed", f"blue:{len(captions):,}", "captions of a total of", f"{len(paths)+len(captions):,}")
    log("Each task has", f"red:{len(paths) // size:,}", "captions, each observing", args.samples, "renderings")
paths = paths[rank::size]

device = "cuda" if is_cuda_available() else "cpu"
model_id = "microsoft/Phi-3.5-vision-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=DTYPE,
    _attn_implementation="eager",  # flash_attention_2
    cache_dir=CACHE_PATH,
)
model.eval()

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    num_crops=NUM_CROPS,
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
    "do_sample": False,
}
prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
cache_suffix = randint(1e6, 1e7 - 1)

# Based on: https://huggingface.co/microsoft/Phi-3.5-vision-instruct/discussions/5#66c6fa5859ff3e4811b4349b
def stack_and_pad_inputs(inputs: list[BatchFeature], pad_token_id: int) -> BatchFeature:
    listof_input_ids = [i.input_ids[0] for i in inputs]
    new_input_ids = pad_left(listof_input_ids, pad_token_id=pad_token_id)
    data = dict(
        pixel_values=torch.cat([i.pixel_values for i in inputs], dim=0),
        image_sizes=torch.cat([i.image_sizes for i in inputs], dim=0),
        input_ids=new_input_ids,
        attention_mask=(new_input_ids != pad_token_id).long(),
    )
    new_inputs = BatchFeature(data).to("cuda")
    return new_inputs


def pad_left(seqs: list[torch.Tensor], pad_token_id: int) -> torch.Tensor:
    """Example: pad_left([[1, 2], [3, 4, 5]], pad_token_id=0) -> [[0, 1, 2], [3, 4, 5]]"""
    max_len = max(len(seq) for seq in seqs)
    padded = torch.full((len(seqs), max_len), pad_token_id)
    for i, seq in enumerate(seqs):
        padded[i, -len(seq) :] = seq
    return padded


batches = [paths[i : i + BATCH_SIZE] for i in range(0, len(paths), BATCH_SIZE)]
invalid_images_uids = []
with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=DTYPE):
        for i, batch in enumerate(tqdm(batches, disable=rank != 0)):
            uids = [sample[0].stem.split("_")[0] for sample in batch]
            assert args.samples == 1
            listof_inputs: list[BatchFeature] = []
            for j, sample in enumerate(batch):
                try:
                    image = Image.open(sample[0])
                    inputs = processor(prompt, [image], return_tensors="pt").to(device)
                    listof_inputs.append(inputs)
                except:
                    invalid_images_uids.append(uids[j])
                    print("invalid image detected, ", sample[0])

            inputs = stack_and_pad_inputs(listof_inputs, pad_token_id=processor.tokenizer.pad_token_id)
            generate_ids = model.generate(
                **inputs,
                eos_token_id=processor.tokenizer.eos_token_id,
                **generation_args,
            )
            generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]  # remove input tokens
            outputs = processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for j, uid in enumerate(filter(lambda uid: uid not in invalid_images_uids, uids)):
                caption = outputs[j].lower().strip()
                captions[uid] = caption
                if args.demo:
                    log(f"[{j}] ({uid})", f"green:{caption}")

            # Save partial results, just in case
            if not args.demo and i % 10 == 0:
                log(f"[Rank {rank}/{size}] [{i}] EXPORTING...")
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
