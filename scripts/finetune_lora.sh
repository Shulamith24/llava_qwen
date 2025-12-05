#!/bin/bash

# Qwen3 LoRA 微调脚本（对齐 LLaVA 参数设计）
# 使用前请修改以下变量：
# - MODEL_PATH: Qwen3 模型路径
# - DATA_PATH: 训练数据路径

# ========== 路径配置 ==========
MODEL_PATH="Qwen/Qwen3-8B"
DATA_PATH="/root/data1/datasets/ChatTS/align_256/train.jsonl"
OUTPUT_DIR="/root/data1/qwen3_finetune/outputs/qwen3-lora-finetune"

# DeepSpeed 配置（可选：zero2.json, zero3.json, zero3_offload.json）
DEEPSPEED_CONFIG="/root/data1/qwen3_finetune/configs/zero3.json"

# LoRA 参数（对齐 LLaVA v1.5）
LORA_R=128
LORA_ALPHA=256

# 训练超参数
LR=2e-4
EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM=4
MAX_LENGTH=2048

# 梯度检查点 False
#


# ========== 开始训练 ==========
deepspeed /root/data1/qwen3_finetune/src/train.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --bits 4 \
    --lora_enable True \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.05 \
    --lora_bias "none" \
    --bf16 True \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 3 \
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
