export SD_MODEL="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_DIR="madebyollin/sdxl-vae-fp16-fix"
export CACHE_DIR="../.huggingface"
export DATASET_DIR="dataset"
export OUTPUT_DIR="trainings/SDxl_CN_64bs_1e-5lr_80k_masked-loss"

# accelerate launch train_controlnet_sdxl.py \
python train_controlnet_sdxl.py \
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
    --learning_rate=1.5e-5 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --mixed_precision="fp16" \
    --checkpointing_steps=1000 \
    --validation_steps=200 \
    --seed=42 \
    \
    --validation_image \
    "validation/uv/8699d1508975469fbfb70d8b96d937e4_0.png" \
    "validation/uv/ea7fc3f240694f82adb2a38e7946c792_0.png" \
    --validation_prompt \
    "a polished, reflective pyramid with a glossy finish, showcasing a gradient of colors from white to red, with a shadow indicating a light source from the upper left." \
    "a 3d rendered image of a gray, matte, and textured object with a rough surface and visible cracks."
