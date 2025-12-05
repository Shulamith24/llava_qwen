#!/bin/bash

# Qwen3 全参数微调脚本（不使用 LoRA）
# 注意：全参数微调需要更多显存

# ========== 配置区域 ==========
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
DATA_PATH="./data/finetune.jsonl"
OUTPUT_DIR="./outputs/qwen3-full-finetune"

# DeepSpeed 配置
DEEPSPEED_CONFIG="./configs/zero3.json"

# 训练超参数
LR=2e-5  # 全参数训练通常用更小的学习率
EPOCHS=3
BATCH_SIZE=2  # 全参数需要更小的 batch
GRAD_ACCUM=8
MAX_LENGTH=2048

# ========== 开始训练 ==========
deepspeed ../src/train.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --lora_enable False \
    --bf16 True \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate $LR \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length $MAX_LENGTH \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to "tensorboard"
