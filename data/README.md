# 数据格式说明

## JSONL 格式

每个训练样本为一行 JSON，包含 `conversations` 字段：

```jsonl
{"conversations": [{"from": "human", "value": "你好"}, {"from": "gpt", "value": "你好！有什么我可以帮助你的吗？"}]}
{"conversations": [{"from": "human", "value": "什么是深度学习？"}, {"from": "gpt", "value": "深度学习是机器学习的一个分支..."}]}
```

## 示例数据

### pretrain.jsonl（预训练/对齐阶段）
用于初步对齐训练，通常是简单的问答对。

### finetune.jsonl（指令微调阶段）
用于指令微调，可以包含多轮对话：

```jsonl
{"conversations": [{"from": "human", "value": "解释量子计算的基本原理"}, {"from": "gpt", "value": "量子计算利用量子力学的叠加态和纠缠态..."}, {"from": "human", "value": "它和传统计算机有什么区别？"}, {"from": "gpt", "value": "主要区别在于..."}]}
```

## 数据准备

请将你的数据按照上述格式准备好，放入此目录：
- `pretrain.jsonl`：预训练数据
- `finetune.jsonl`：微调数据

## 字段说明

- `from`: 发言者角色
  - `"human"`: 用户
  - `"gpt"`: 助手
  
- `value`: 对话内容（字符串）

## 注意事项

1. 每行必须是合法的 JSON
2. `conversations` 必须是列表
3. 建议 human 和 gpt 交替出现
4. 每个样本至少包含一轮对话
