# 多模态时序Qwen3模型

基于LLaVA范式的时间序列+文本多模态模型，结合PatchTST时序编码器和Qwen3语言模型。

## 📋 目录

- [架构概述](#架构概述)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [训练流程](#训练流程)
- [推理使用](#推理使用)
- [项目结构](#项目结构)

## 🏗️ 架构概述

```
时间序列 [n_vars × seq_len]
    ↓
PatchTST编码器 [预训练/冻结]
    ↓
Patch特征 [n_vars × n_patches × d_model]
    ↓
MLP投影层 [可训练]
    ↓
投影特征 [n_vars × n_patches × hidden_size]
    ↓
<ts> token替换 → 融合Embedding序列
    ↓
Qwen3模型 [LoRA微调]
    ↓
文本生成
```

### 核心特性

- ✅ **两阶段训练**：预训练投影层 + LoRA微调
- ✅ **灵活的时序编码器**：支持PatchTST权重加载和冻结/解冻
- ✅ **特殊token设计**：使用`<ts>`作为时序占位符（类似LLaVA的`<image>`）
- ✅ **批量处理**：支持不同长度的时间序列（逐样本编码）
- ✅ **QLoRA支持**：4-bit量化，降低显存需求

## 🛠️ 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 检查环境

```bash
python scripts/check_env.py
```

### 3. PatchTST权重

确保PatchTST预训练权重存在：
```bash
ls PatchTST_supervised/checkpoints/*/checkpoint.pth
```

## 📊 数据准备

### 数据格式（JSONL）

每行一个JSON对象：

```json
{
  "input": "There are 2 time series. <ts></ts> and <ts></ts>. What patterns do you see?",
  "time_series": [
    [1.0, 2.0, 3.0, ..., 256个值],
    [4.0, 5.0, 6.0, ..., 256个值]
  ],
  "output": "Both time series show upward trends with seasonal patterns..."
}
```

### 重要规则

1. ⚠️ **`<ts></ts>`成对出现**：在input文本中，每个变量对应一对`<ts></ts>`标签
2. ⚠️ **数量匹配**：`<ts></ts>`的数量必须等于`time_series`数组长度
3. ⚠️ **长度一致**：同一样本内所有变量的长度相同（默认256，可配置）
4. ⚠️ **跨样本可变**：不同样本的变量数和长度可以不同

### 示例数据准备

```bash
# 假设你的数据在data/目录下
# data/pretrain.jsonl - 预训练数据（大规模）
# data/finetune.jsonl - 微调数据（任务相关）
```

## 🚀 训练流程

### 阶段1：预训练投影层

只训练MLP投影层，Qwen3和PatchTST都冻结。

```bash
# 修改scripts/pretrain_projector.sh中的数据路径
bash scripts/pretrain_projector.sh
```

**关键参数**：
- `--tune_mm_mlp_adapter True`：只训练投影层
- `--freeze_patchtst True`：冻结PatchTST
- `--learning_rate 1e-3`：较大的学习率

**输出**：
- `outputs/pretrain_projector/mm_projector.bin`：投影层权重

### 阶段2：LoRA微调

微调Qwen3（使用LoRA）+ 投影层，PatchTST仍冻结。

```bash
# 修改scripts/finetune_lora_multimodal.sh中的数据路径
bash scripts/finetune_lora_multimodal.sh
```

**关键参数**：
- `--pretrain_mm_mlp_adapter outputs/pretrain_projector/mm_projector.bin`：加载预训练的投影层
- `--lora_enable True`：启用LoRA
- `--lora_r 128`：LoRA秩
- `--learning_rate 2e-4`：较小的学习率

**输出**：
- `outputs/finetune_lora/adapter_model.bin`：LoRA权重
- `outputs/finetune_lora/non_lora_trainables.bin`：投影层权重

### 阶段2（可选）：QLoRA微调（4-bit量化，单GPU）

```bash
bash scripts/finetune_qlora_multimodal.sh
```

**优势**：
- 显存需求更低（约10GB可训练8B模型）
- 适合单GPU环境

## 🔍 测试验证

### 1. 测试数据加载

```bash
python tools/test_multimodal_data.py \
    --data_path data/finetune.jsonl \
    --model_path Qwen/Qwen3-8B \
    --context_window 256 \
    --test all
```

**测试内容**：
- ✅ 数据集加载（JSONL解析，`<ts>`验证）
- ✅ DataCollator（批处理，padding）
- ✅ 模型forward（loss计算）

### 2. 推理示例

```bash
# 使用示例数据
python tools/inference_demo.py \
    --model_path Qwen/Qwen3-8B \
    --checkpoint outputs/finetune_lora \
    --max_new_tokens 512

# 使用自定义数据
python tools/inference_demo.py \
    --model_path Qwen/Qwen3-8B \
    --checkpoint outputs/finetune_lora \
    --input_file data/test_sample.jsonl
```

## 📁 项目结构

```
qwen3_finetune/
├── src/
│   ├── constants.py              # 全局常量（IGNORE_INDEX, TS_TOKEN_INDEX等）
│   ├── dataset.py                # 纯文本数据集（原有）
│   ├── dataset_multimodal.py     # 多模态数据集（新增）
│   ├── train.py                  # 纯文本训练（原有）
│   ├── train_multimodal.py       # 多模态训练（新增）
│   └── model/
│       ├── projector.py          # MLP投影层
│       ├── ts_encoder.py         # PatchTST编码器包装
│       └── qwen3_ts.py           # 多模态Qwen3模型
├── scripts/
│   ├── pretrain_projector.sh     # 预训练脚本
│   ├── finetune_lora_multimodal.sh  # LoRA微调脚本
│   └── finetune_qlora_multimodal.sh # QLoRA微调脚本
├── tools/
│   ├── test_multimodal_data.py   # 数据加载测试
│   └── inference_demo.py         # 推理示例
├── llava_model/                  # LLaVA参考代码（不修改）
├── PatchTST_supervised/          # PatchTST源码和权重
├── configs/                      # DeepSpeed配置
└── data/                         # 数据目录
```

## 🎯 使用技巧

### 1. PatchTST配置

关键超参数（应与预训练PatchTST保持一致）：
- `--context_window 256`：时序窗口长度
- `--patch_len 16`：patch长度
- `--stride 8`：patch步长
- `--ts_d_model 128`：PatchTST输出维度

### 2. 内存优化

如果显存不足：
1. 使用QLoRA（4-bit量化）
2. 减小batch size，增大gradient accumulation
3. 使用DeepSpeed Zero3
4. 减小`--model_max_length`

### 3. 训练监控

使用wandb监控训练：
```bash
# 设置wandb
export WANDB_PROJECT="qwen3_ts"
export WANDB_API_KEY="your_key"
```

### 4. 调试模式

快速验证流程（使用小数据集）：
```bash
python src/train_multimodal.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --data_path data/debug_10samples.jsonl \
    --output_dir outputs/debug \
    --num_train_epochs 1 \
    --save_steps 10 \
    --logging_steps 1
```

## 🐛 常见问题

### 1. `<ts>` token数量不匹配

**错误**：`ValueError: 文本中<ts>数量(3)与时间序列变量数(2)不匹配`

**解决**：检查JSONL数据，确保每个样本的`<ts></ts>`数量等于`time_series`数组长度

### 2. PatchTST权重加载失败

**错误**：`FileNotFoundError: checkpoint.pth不存在`

**解决**：检查`PatchTST_supervised/checkpoints/*/*/checkpoint.pth`路径

### 3. CUDA out of memory

**解决方案**：
- 使用QLoRA（`--bits 4`）
- 减小batch size（`--per_device_train_batch_size 1`）
- 增大gradient accumulation（`--gradient_accumulation_steps 16`）
- 使用DeepSpeed Zero3（`--deepspeed configs/zero3.json`）

### 4. 不同长度的时间序列

**说明**：
- ✅ 同一样本内所有变量长度必须相同
- ✅ 不同样本的长度可以不同（会自动padding/截断到`context_window`）
- ✅ 不同样本的变量数可以不同（逐样本编码）

## 📝 引用

如果使用本项目，请引用：

```bibtex
@software{qwen3_timeseries_multimodal,
  title = {Multimodal Time Series Qwen3 Model},
  author = {Your Name},
  year = {2025},
  note = {Building on LLaVA and PatchTST architectures}
}
```

## 📄 许可证

本项目遵循原Qwen3和LLaVA的许可证。

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

**最后更新**：2025-12-05
