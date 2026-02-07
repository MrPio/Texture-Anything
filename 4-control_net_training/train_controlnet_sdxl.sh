export SD_MODEL="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_DIR="madebyollin/sdxl-vae-fp16-fix"
export CACHE_DIR="../.huggingface"
export DATASET_DIR="dataset"
export OUTPUT_DIR="trainings/SDxl_CN_8bs_165e-5lr_2k_masked-loss"

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
    --num_train_epochs=100 \
    --learning_rate=1.65e-5 \
    --train_batch_size=8 \
    --gradient_accumulation_steps=2 \
    --mixed_precision="fp16" \
    --checkpointing_steps=500 \
    --validation_steps=125 \
    --seed=42 \
    \
    --validation_image \
    "dataset/validation/uv/6a9d81b18a844a33b56339046df45035_0.png" \
    "dataset/validation/uv/53f4f0376556450abf37190f9a462b1d_0.png" \
    "dataset/validation/uv/1090ceab53994fd289f09fcecbce8e6d_0.png" \
    "dataset/validation/uv/e47c4080012b48b9b973782505985629_0.png" \
    --validation_prompt \
    "a rectangular box with a rough, weathered surface, showing signs of age and exposure to the elements. the color is a muted, earthy tone with a mix of grays and browns, suggesting a metallic or stone material." \
    "a rustic wooden box with a weathered finish, featuring a dark brown hue and natural wood grain patterns. the box has a rectangular shape with a flat top and a small, round, black handle on the front. a red sticker with the text \"cute burros\" is affixed to" \
    "a vibrant, teal-colored box with a floral pattern on the front and a smaller image on the side, showcasing a variety of flowers with detailed petals and centers." \
    "a rectangular metal plate with a rusted surface, featuring vertical stripes and a series of rivets along the edges." 
