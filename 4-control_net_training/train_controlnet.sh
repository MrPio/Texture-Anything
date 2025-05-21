#!/bin/bash
#SBATCH --job-name=controlnet_sd15
#SBATCH --output=controlnet_sd15.log
#SBATCH --error=controlnet_sd15.log
#SBATCH --time=16:00:00
#SBATCH --partition=boost_usr_prod
##SBATCH --qos=boost_qos_dbg                  # Refer to https://wiki.u-gov.it/confluence/display/SCAIUS/Booster+Section
#SBATCH --gres=gpu:4
#SBATCH --mem=64G

export SD_MODEL="stable-diffusion-v1-5/stable-diffusion-v1-5"
export CNET_MODEL="lllyasviel/sd-controlnet-mlsd"
export OUTPUT_DIR="/leonardo_scratch/fast/IscrC_MACRO/Texture-Anything/4-control_net_training/output/SD1.5_CNmlsd_128bs_5e-6lr_13k"
export CACHE_DIR="/leonardo_scratch/fast/IscrC_MACRO/Texture-Anything/.huggingface"
export DATASET_DIR="/leonardo_scratch/fast/IscrC_MACRO/Texture-Anything/4-control_net_training/dataset"

cd /leonardo_scratch/fast/IscrC_MACRO/Texture-Anything/4-control_net_training

# accelerate launch train_controlnet.py \
accelerate launch --mixed_precision="fp16" --num_processes=4 train_controlnet.py \
    --pretrained_model_name_or_path=$SD_MODEL \
    --controlnet_model_name_or_path=$CNET_MODEL \
    --output_dir=$OUTPUT_DIR \
    --cache_dir=$CACHE_DIR \
    --train_data_dir=$DATASET_DIR \
    --image_column="diffuse" \
    --conditioning_image_column="uv" \
    --caption_column="caption" \
    --invert_conditioning_image \
    \
    --resolution=512 \
    --num_train_epochs=50 \
    --learning_rate=5e-6 \
    --train_batch_size=8 \
    --gradient_accumulation_steps=2 \
    --mixed_precision="fp16" \
    \
    --validation_image \
    "dataset/validation/uv/0a2ad73285bc44d7a4362f54f5de3cc4.png" \
    "dataset/validation/uv/0adf456c59094a3da23329a6d27cb239.png" \
    "dataset/validation/uv/0cde463ba5d64acd979ab2d4e31fded0.png" \
    "dataset/validation/uv/ffebd77d0373462eb7c6d2d4941eb39c.png" \
    "dataset/validation/uv/3b15c410f87f42daa7e8cb5b5f74e3f1.png" \
    --validation_prompt \
    "a rustic wooden house with a dark brown shingled roof and a white stucco exterior with red cross patterns." \
    "a dark brown, metallic gear with a rough, industrial texture, featuring sharp teeth and a central hole, suggesting a robust and durable design." \
    "a wooden block with a smooth, polished surface and a light brown color, showcasing a natural wood grain pattern." \
    "a black non-stick frying pan with a matte finish and a handle made of a different material, possibly plastic or rubber, is placed on a neutral background." \
    "a delicate, ornate silver plate with intricate floral patterns and a glossy finish, reflecting light subtly."
