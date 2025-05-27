#!/bin/bash

# To run a new training, please make sure to set:
# - Accelerate's --num_processes=<NUM_GPUS>
# - The name of the train in $OUTPUT_DIR
# - A number of --epochs compatible with the logic batch size
# - Wheter to use --use_pixel_space_loss

#SBATCH --job-name=controlnet_sd15
#SBATCH --output=controlnet_sd15.log
#SBATCH --error=controlnet_sd15.log
#SBATCH --time=10:00:00
#SBATCH --partition=boost_usr_prod
##SBATCH --qos=boost_qos_dbg                  # Refer to https://wiki.u-gov.it/confluence/display/SCAIUS/Booster+Section
#SBATCH --gres=gpu:2
#SBATCH --mem=48G

export SD_MODEL="stable-diffusion-v1-5/stable-diffusion-v1-5"
export CNET_MODEL="lllyasviel/sd-controlnet-mlsd"
export CACHE_DIR="/leonardo_scratch/fast/IscrC_MACRO/Texture-Anything/.huggingface"
export DATASET_DIR="/leonardo_scratch/fast/IscrC_MACRO/Texture-Anything/4-control_net_training/dataset"

export OUTPUT_DIR="/leonardo_scratch/fast/IscrC_MACRO/Texture-Anything/4-control_net_training/trainings/SD1.5_CNmlsd_64bs_1e-5lr_8k_pixel-loss"

cd /leonardo_scratch/fast/IscrC_MACRO/Texture-Anything/4-control_net_training

# accelerate launch train_controlnet.py \
accelerate launch --mixed_precision="fp16" --num_processes=2 train_controlnet.py \
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
    --use_pixel_space_loss \
    \
    --resolution=512 \
    --num_train_epochs=60 \
    --learning_rate=1e-5 \
    --train_batch_size=8 \
    --gradient_accumulation_steps=4 \
    --mixed_precision="fp16" \
    --checkpointing_steps=500 \
    \
    --validation_image \
    "dataset/validation/uv/0adf456c59094a3da23329a6d27cb239.png" \
    "dataset/validation/uv/3b15c410f87f42daa7e8cb5b5f74e3f1.png" \
    --validation_prompt \
    "a dark brown, metallic gear with a rough, industrial texture, featuring sharp teeth and a central hole, suggesting a robust and durable design." \
    "a delicate, ornate silver plate with intricate floral patterns and a glossy finish, reflecting light subtly."
