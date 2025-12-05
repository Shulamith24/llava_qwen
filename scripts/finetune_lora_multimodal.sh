#!/bin/bash

# LoRA微调阶段：微调Qwen3+投影层
# 使用方法: bash scripts/finetune_lora_multimodal.sh

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 训练参数
MODEL_PATH="Qwen/Qwen3-8B"
DATA_PATH="data/finetune.jsonl"
OUTPUT_DIR="outputs/finetune_lora"
PATCHTST_CKPT="PatchTST_supervised/checkpoints/checkpoint.pth"

# 加载预训练的投影层
PRETRAIN_PROJECTOR="outputs/pretrain_projector/mm_projector.bin"

# PatchTST配置（应与预训练保持一致）
CONTEXT_WINDOW=256
PATCH_LEN=16
STRIDE=8
TS_D_MODEL=128

# LoRA配置
LORA_R=128
LORA_ALPHA=256
LORA_DROPOUT=0.05

# 训练配置
NUM_TRAIN_EPOCHS=10
PER_DEVICE_TRAIN_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=2e-4
WARMUP_RATIO=0.03

deepspeed --num_gpus=4 src/train_multimodal.py \
    --deepspeed configs/zero3.json \
    --model_name_or_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --mm_ts_tower patchtst \
    --patchtst_checkpoint ${PATCHTST_CKPT} \
    --freeze_patchtst True \
    --context_window ${CONTEXT_WINDOW} \
    --patch_len ${PATCH_LEN} \
    --stride ${STRIDE} \
    --ts_d_model ${TS_D_MODEL} \
    --ts_n_layers 3 \
    --ts_n_heads 16 \
    --ts_d_ff 256 \
    --ts_dropout 0.1 \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter ${PRETRAIN_PROJECTOR} \
    --lora_enable True \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --lora_bias "none" \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0. \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name finetune_lora_$(date +%Y%m%d_%H%M%S) \
    --bf16 True \
    --tf32 True
