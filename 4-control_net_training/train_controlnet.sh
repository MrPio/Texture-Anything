export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="/leonardo_scratch/fast/IscrC_MACRO/Texture-Anything/4-control_net_training/output"
export CACHE_DIR="/leonardo_scratch/fast/IscrC_MACRO/Texture-Anything/.huggingface"

accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --cache_dir=$CACHE_DIR \
 --dataset_name=fusing/fill50k \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --train_batch_size=4