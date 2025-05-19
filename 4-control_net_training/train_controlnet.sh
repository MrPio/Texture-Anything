export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="/leonardo_scratch/fast/IscrC_MACRO/Texture-Anything/4-control_net_training/output"
export CACHE_DIR="/leonardo_scratch/fast/IscrC_MACRO/Texture-Anything/.huggingface"
export DATASET_DIR="/leonardo_scratch/fast/IscrC_MACRO/Texture-Anything/4-control_net_training/dataset"

accelerate launch --mixed_precision="fp16" --multi_gpu train_controlnet.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --cache_dir=$CACHE_DIR \
    --train_data_dir=$MODEL_DIR \
    \
    --resolution=512 \
    --learning_rate=1e-5 \
    --train_batch_size=4 \
    --mixed_precision="fp16" \
    \
    --validation_image $(printf "\"%s\" " dataset/validation/uv/*.png) \
    --validation_prompt python3 -c 'import csv; print(" ".join(f"\"{row[2]}\"" for row in csv.reader(open("dataset/validation/metadata.csv")) if row != [] and row[0] != "uv_file_name"))'
