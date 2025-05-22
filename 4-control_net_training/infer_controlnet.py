"""
Generate predictions over the testset. CWD-independent.

Usage:
    $ srun --mem=16G --gres=gpu:1 --time=00:08:00 --partition=boost_usr_prod --qos=boost_qos_dbg \
        python infer_controlnet.py \
            --cnet="SD1.5_CNmlsd_128bs_5e-6lr_13k" \
            --checkpoint=500
            

Based on: https://github.com/huggingface/diffusers/tree/main/examples/controlnet
"""

import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT_DIR = Path(__file__).parents[1]
CACHE_DIR = ROOT_DIR / ".huggingface"
TESTSET_DIR = ROOT_DIR / "4-control_net_training" / "dataset" / "test"
output_dir = None


# Args =====================================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sd",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="The SD model with which the CNet was trained.",
    )
    parser.add_argument(
        "--cnet",
        type=str,
        default="lllyasviel/sd-controlnet-mlsd",
        help="The subfolder of trainings/ where to load the checkpoints from, or a model in the hub.",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="The steps of the checkpoint to load. Required if --cnet indicates a local training.",
    )
    parser.add_argument(
        "--invert-uv",
        type=bool,
        default=True,
        help="Whether to invert the color of the UV images.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=30,
        help="The number of samples to consider in the test set.",
    )
    parser.add_argument(
        "--generator-seed",
        type=int,
        default=0,
        help="The seed for the diffusion process.",
    )
    args = parser.parse_args()

    # Check that this cnet model has not already been tested
    global output_dir
    output_dir = (
        Path(__file__).parent
        / "tests"
        / (args.cnet.replace("/", "-") + (f"_{args.checkpoint}s" if args.checkpoint else ""))
    )
    if output_dir.exists():
        raise ValueError("This model already has prediction in", output_dir)

    return args


args = parse_args()
CNET_MODEL = (
    (Path(__file__).parent / "trainings" / args.cnet / f"checkpoint-{args.checkpoint}" / "controlnet")
    if args.checkpoint
    else args.cnet
)

# Load dataset =============================================================
testset = pd.read_csv(TESTSET_DIR / "metadata.csv").iloc[: args.samples]

# Loading pipeline =========================================================
import PIL
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
import torch
from tqdm import tqdm

sys.path.insert(0, str(ROOT_DIR))
from src import *

controlnet = ControlNetModel.from_pretrained(
    pretrained_model_name_or_path=str(CNET_MODEL),
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,
    local_files_only=True,
)
pipe = (StableDiffusionXLControlNetPipeline if "xl" in args.sd else StableDiffusionControlNetPipeline).from_pretrained(
    pretrained_model_name_or_path=args.sd,
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

generator = torch.manual_seed(args.generator_seed)
output_dir.mkdir(parents=True, exist_ok=True)

for _, row in tqdm(testset.iterrows()):
    control_image = PIL.Image.open(TESTSET_DIR / row.uv_file_name).convert("RGB")
    if args.invert_uv:
        control_image = PIL.ImageOps.invert(control_image)
    image = pipe(row.caption, num_inference_steps=20, generator=generator, image=control_image).images[0]
    image.save(output_dir / f"{Path(row.uv_file_name).stem}.png")
