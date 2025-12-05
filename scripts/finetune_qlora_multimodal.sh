#!/bin/bash

# QLoRA微调阶段（4-bit量化）
# 使用方法: bash scripts/finetune_qlora_multimodal.sh

export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="Qwen/Qwen3-8B"
DATA_PATH="data/finetune.jsonl"
OUTPUT_DIR="outputs/finetune_qlora"
PATCHTST_CKPT="PatchTST_supervised/checkpoints/checkpoint.pth"
PRETRAIN_PROJECTOR="outputs/pretrain_projector/mm_projector.bin"

# PatchTST配置
CONTEXT_WINDOW=256
PATCH_LEN=16
STRIDE=8
TS_D_MODEL=128

# QLoRA配置
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.05
BITS=4  # 4-bit量化

NUM_TRAIN_EPOCHS=10
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=2e-4

python src/train_multimodal.py \
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
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter ${PRETRAIN_PROJECTOR} \
    --bits ${BITS} \
    --lora_enable True \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --run_name finetune_qlora_$(date +%Y%m%d_%H%M%S) \
    --fp16 True
