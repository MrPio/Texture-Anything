import argparse
import json
import os
from glob import glob
from pathlib import Path
from tqdm import tqdm
import torch
from lavis.models import load_model_and_preprocess, load_model
from PIL import Image
from mpi4py import MPI
import sys

ROOT_PATH = Path(__file__).parent.parent.resolve()

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()


import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# choose your cache directory
cache_dir = "../.huggingface/"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl", cache_dir=cache_dir)
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xxl", cache_dir=cache_dir)

# TODO

model.eval()

# You can now mimic lavis’s vis_processors / txt_processors:
#   - vis_processors["eval"] takes a PIL image or tensor and returns pixel_values
#   - txt_processors["eval"] takes a string (or list of strings) and returns input_ids / attention_mask

def vis_eval(image):
    """
    image: PIL.Image.Image or torch.Tensor
    returns: pixel_values tensor ready for model
    """
    # If you have a PIL image:
    #   inputs = processor(images=image, return_tensors="pt")
    # If you already have a tensor HxWxC in [0,1]:
    #   inputs = processor(images=image, return_tensors="pt")
    inputs = processor(images=image, return_tensors="pt")
    return inputs.pixel_values  # shape (1, 3, H, W)

def txt_eval(text: str):
    """
    text: prompt string
    returns: dict(input_ids=..., attention_mask=...)
    """
    return processor.tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        truncation=True
    )

# Example usage:
# pixel_values = vis_eval(pil_image).to(device)
# text_inputs  = txt_eval("Describe this image").to(device)
# outputs = model.generate(pixel_values=pixel_values, **text_inputs)


sys.exit(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/dataset/objaverse/render/")
    parser.add_argument("--output", type=str, default="data/dataset/objaverse/caption/")
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "blip_caption:base_coco",
            "blip2_opt:caption_coco_opt2.7b",
            "blip2_opt:caption_coco_opt6.7b",
            "blip2_t5:pretrain_flant5xxl",
        ],
        default="blip2_t5:pretrain_flant5xxl",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
    )
    return parser.parse_args()


args = parse_args()
paths = list((ROOT_PATH / args.input).glob("*.jpg"))[rank::size]
if args.demo:
    paths = paths[:2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name, model_type = args.model.split(":")
load_model(model_name, model_type, True)
model, vis_processors, _ = load_model_and_preprocess(
    name=model_name, model_type=model_type, is_eval=True, device=device
)

captions = {}
instruction = "Describe this rendered 3D model in rich detail, focusing on its shape, materials, textures, and colors."

paths = tqdm(paths) if rank == 0 else paths
for p in paths:
    raw_image = Image.open(p).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    caption = model.generate(
        samples={"image": image, "prompt": instruction},
        use_nucleus_sampling=True,
        temperature=0.7,
        top_p=0.9,
        max_length=64,
        repetition_penalty=1.2,
    )[0]
    captions[os.path.basename(p)] = caption.lower().strip()

all_captions = comm.gather(captions, root=0)
if rank == 0:
    all_captions = {k: v for d in all_captions for k, v in d.items()}
    with open(ROOT_PATH / args.output / "captions.json", "w") as f:
        json.dump(all_captions, f, indent=4)
