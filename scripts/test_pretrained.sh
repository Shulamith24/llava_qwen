#!/bin/bash

# 预训练模型测试脚本
# 在相同数据集上进行前向验证
# 用法: bash scripts/test_pretrained.sh

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 路径配置
CHECKPOINT_PATH="/mnt/data/qyh/codes/llava_qwen/outputs/pretrain_projector_multi/checkpoint-2463"
DATA_PATH="/mnt/data/qyh/codes/dataset/ChatTS/align_256/train.jsonl"

# 测试参数
BATCH_SIZE=2
CONTEXT_WINDOW=256
MODEL_MAX_LENGTH=2048

# 可选：限制测试样本数（加快测试速度）
# NUM_SAMPLES=100

# 可选：生成对比测试
GENERATE_SAMPLES=3
MAX_NEW_TOKENS=256

# 运行测试
python src/test_pretrained_model.py \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --data_path ${DATA_PATH} \
    --batch_size ${BATCH_SIZE} \
    --context_window ${CONTEXT_WINDOW} \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --generate_samples ${GENERATE_SAMPLES} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --bf16
    # --num_samples ${NUM_SAMPLES}  # 取消注释以限制样本数
