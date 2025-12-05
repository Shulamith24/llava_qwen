#!/bin/bash

# Qwen3 QLoRA (4-bit) 微调脚本（对齐 LLaVA 参数设计）
# QLoRA = 4bit 量化 + LoRA，显存占用更低

# ========== 配置区域 ==========
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
DATA_PATH="./data/finetune.jsonl"
OUTPUT_DIR="./outputs/qwen3-qlora-finetune"

# DeepSpeed 配置（QLoRA 建议用 zero2）
DEEPSPEED_CONFIG="./configs/zero2.json"

# LoRA 参数
LORA_R=128
LORA_ALPHA=256

# 训练超参数
LR=2e-4
EPOCHS=3
BATCH_SIZE=8  # QLoRA 可以用更大的 batch
GRAD_ACCUM=2
MAX_LENGTH=2048

# ========== 开始训练 ==========
deepspeed ../src/train.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --bits 4 \
    --double_quant True \
    --quant_type "nf4" \
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
    --evaluation_strategy "no" \
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
