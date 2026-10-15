#!/bin/bash

MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_ROOT="/work/nvme/bgxm/wliu12/repos/SpinQuant/outputs/llama3_8b_instruct_w16a4kv4"

mkdir -p "${OUTPUT_ROOT}/rotation"
mkdir -p "${OUTPUT_ROOT}/trainer"
mkdir -p "${OUTPUT_ROOT}/logs"

# one gpu version
torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  optimize_rotation.py \
  --input_model "${MODEL_NAME}" \
  --output_rotation_path "${OUTPUT_ROOT}/rotation" \
  --output_dir "${OUTPUT_ROOT}/trainer" \
  --logging_dir "${OUTPUT_ROOT}/logs" \
  --model_max_length 512 \
  --fp16 False \
  --bf16 True \
  --log_on_each_node False \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --logging_steps 1 \
  --learning_rate 1.5 \
  --weight_decay 0.0 \
  --lr_scheduler_type cosine \
  --gradient_checkpointing True \
  --save_safetensors False \
  --max_steps 100 \
  --w_bits 16 \
  --a_bits 4 \
  --k_bits 4 \
  --v_bits 4 \
  --w_clip \
  --a_asym \
  --k_asym \
  --v_asym \
  --k_groupsize 128 \
  --v_groupsize 128 \