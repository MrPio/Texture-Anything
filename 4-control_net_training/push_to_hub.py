from diffusers import ControlNetModel
from huggingface_hub import login
import os
import dotenv
import torch

dotenv.load_dotenv()
login(os.environ['HF_TOKEN'])

CNET_MODEL="trainings/SD1.5_CNmlsd_64bs_1e-5lr_8k_combined-loss/checkpoint-7000/controlnet"
model = ControlNetModel.from_pretrained(
    CNET_MODEL, torch_dtype=torch.float16, local_files_only=True
)

model.push_to_hub("MrPio/Texture-Anything_CNet-SD15")
