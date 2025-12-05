#!/bin/bash

# 预训练阶段：只训练投影层
# 使用方法: bash scripts/pretrain_projector.sh

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 根据GPU数量调整

# 训练参数
MODEL_PATH="Qwen/Qwen3-8B"  # Qwen3模型路径
DATA_PATH="data/pretrain.jsonl"  # 预训练数据路径
OUTPUT_DIR="outputs/pretrain_projector"  # 输出目录
PATCHTST_CKPT="PatchTST_supervised/checkpoints/checkpoint.pth"  # PatchTST权重

# PatchTST配置
CONTEXT_WINDOW=256
PATCH_LEN=16
STRIDE=8
TS_D_MODEL=128

# 训练配置
NUM_TRAIN_EPOCHS=3
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=1e-3
WARMUP_RATIO=0.03

deepspeed --num_gpus=4 src/train_multimodal.py \
    --deepspeed configs/zero2.json \
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
    --run_name pretrain_projector_$(date +%Y%m%d_%H%M%S) \
    --bf16 True \
    --tf32 True
