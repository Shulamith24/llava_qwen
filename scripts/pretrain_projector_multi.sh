#!/bin/bash

# 预训练阶段：只训练投影层 (双卡版本)
# 使用方法: bash scripts/pretrain_projector_multi.sh

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1

# 训练参数
MODEL_PATH="Qwen/Qwen3-4B"
DATA_PATH="/root/data1/datasets/ChatTS/align_256/train.jsonl"
OUTPUT_DIR="outputs/pretrain_projector_multi"
PATCHTST_CKPT=None
CHECKPOINT_PATH="/mnt/data/qyh/codes/llava_qwen/outputs/pretrain_projector_multi/mm_projector.bin"

# PatchTST配置
CONTEXT_WINDOW=256
PATCH_LEN=16
STRIDE=8
TS_D_MODEL=128

# 训练配置 (双卡优化)
NUM_TRAIN_EPOCHS=3
PER_DEVICE_TRAIN_BATCH_SIZE=4
# Global Batch Size = 4 * 16 * 2 = 128 (与单卡 4 * 32 * 1 = 128 保持一致)
GRADIENT_ACCUMULATION_STEPS=16
LEARNING_RATE=1e-3
WARMUP_RATIO=0.03

# 双卡训练，使用 deepspeed
deepspeed src/train_multimodal.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --mm_ts_tower patchtst \
    --patchtst_checkpoint ${PATCHTST_CKPT} \
    --freeze_patchtst False \
    --context_window ${CONTEXT_WINDOW} \
    --patch_len ${PATCH_LEN} \
    --stride ${STRIDE} \
    --ts_d_model ${TS_D_MODEL} \
    --ts_n_layers 3 \
    --ts_n_heads 16 \
    --ts_d_ff 256 \
    --ts_dropout 0.1 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
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
    --save_total_limit 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name pretrain_projector_multi_$(date +%Y%m%d_%H%M%S) \
    --bf16 True \
    --tf32 True
