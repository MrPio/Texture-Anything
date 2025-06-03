#!/bin/bash

# To run a new training, please make sure to set:
# - Accelerate's --num_processes=<NUM_GPUS>
# - The name of the train in $OUTPUT_DIR
# - A number of --epochs compatible with the logic batch size

#SBATCH --job-name=controlnet_sdxl
#SBATCH --output=controlnet_sdxl.log
#SBATCH --error=controlnet_sdxl.log
#SBATCH --time=24:00:00
#SBATCH --partition=boost_usr_prod
##SBATCH --qos=boost_qos_dbg
#SBATCH --gres=gpu:2
#SBATCH --mem=42G

export SD_MODEL="stabilityai/stable-diffusion-xl-base-1.0"
export CACHE_DIR="/leonardo_scratch/fast/IscrC_MACRO/Texture-Anything/.huggingface"
export DATASET_DIR="/leonardo_scratch/fast/IscrC_MACRO/Texture-Anything/4-control_net_training/dataset"
export VAE_DIR="madebyollin/sdxl-vae-fp16-fix"

export OUTPUT_DIR="/leonardo_scratch/fast/IscrC_MACRO/Texture-Anything/4-control_net_training/trainings/SDxl_CN_64bs_1e-5lr_80k_masked-loss"

cd /leonardo_scratch/fast/IscrC_MACRO/Texture-Anything/4-control_net_training

# accelerate launch train_controlnet_sdxl.py \
accelerate launch --mixed_precision="fp16" --num_processes=2 train_controlnet_sdxl.py \
    --pretrained_model_name_or_path=$SD_MODEL \
    --output_dir=$OUTPUT_DIR \
    --cache_dir=$CACHE_DIR \
    --train_data_dir=$DATASET_DIR \
    --pretrained_vae_model_name_or_path=$VAE_DIR \
    --image_column="diffuse" \
    --mask_column="mask" \
    --conditioning_image_column="uv" \
    --caption_column="caption" \
    \
    --resolution=512 \
    --num_train_epochs=50 \
    --learning_rate=1e-5 \
    --train_batch_size=8 \
    --gradient_accumulation_steps=4 \
    --mixed_precision="fp16" \
    --checkpointing_steps=1000 \
    --validation_steps=100 \
    --seed=42 \
    \
    --validation_image \
    "dataset/validation/uv/0adf456c59094a3da23329a6d27cb239.png" \
    "dataset/validation/uv/3b15c410f87f42daa7e8cb5b5f74e3f1.png" \
    --validation_prompt \
    "a dark brown, metallic gear with a rough, industrial texture, featuring sharp teeth and a central hole, suggesting a robust and durable design." \
    "a delicate, ornate silver plate with intricate floral patterns and a glossy finish, reflecting light subtly."
