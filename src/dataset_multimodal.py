"""
多模态数据集类
支持时间序列+文本的JSONL格式数据加载
"""

import re
import json
import torch
from typing import Dict, List, Optional
from torch.utils.data import Dataset
import transformers

# 兼容两种导入方式：作为包导入 vs 直接运行脚本
try:
    from .constants import IGNORE_INDEX, DEFAULT_TS_TOKEN
except ImportError:
    from constants import IGNORE_INDEX, DEFAULT_TS_TOKEN


def normalize_timeseries(ts_tensor: torch.Tensor, eps: float = 1e-5):
    """
    对时间序列进行Instance Normalization，并提取尺度信息
    
    Args:
        ts_tensor: [n_vars, seq_len] 时间序列张量
        eps: 防止除零的小常数
        
    Returns:
        normalized: [n_vars, seq_len] 归一化后的时间序列
        scale_stats: [n_vars, 2] 每个变量的(mean, std)
    """
    # 计算每个变量的统计量
    means = ts_tensor.mean(dim=-1)  # [n_vars]
    stds = ts_tensor.std(dim=-1).clamp(min=eps)  # [n_vars]
    
    # 归一化: (x - mean) / std
    normalized = (ts_tensor - means.unsqueeze(-1)) / stds.unsqueeze(-1)
    
    # 组合尺度信息: [n_vars, 2]
    scale_stats = torch.stack([means, stds], dim=-1)
    
    return normalized, scale_stats



# 正则表达式：匹配<ts></ts>成对标签

# 用于删除空的<think>块（继承自原dataset.py）
EMPTY_THINK_PATTERN = re.compile(r'<think>\s*</think>\s*', flags=re.S)


def preprocess_multimodal_qwen(
    input_text: str,
    output_text: str,
    timeseries: List[List[float]],
    tokenizer: transformers.PreTrainedTokenizer,
    context_window: int,
    remove_thinking: bool = True
) -> Dict:
    """
    预处理多模态样本（时间序列+文本）
    
    Args:
        input_text: 包含<ts></ts>占位符的用户问题
        output_text: 模型回答
        timeseries: 时间序列数据 [[var1], [var2], ...]
        tokenizer: Qwen3 tokenizer
        context_window: 时序序列长度（用于验证）
        remove_thinking: 是否移除空的<think>块
        
    Returns:
        dict: {
            "input_ids": tensor,
            "labels": tensor,
            "timeseries": tensor [n_vars, seq_len]
        }
    """
    # 1. 处理文本：将<ts><ts/>替换为<ts>
    input_text = input_text.replace("<ts><ts/>", DEFAULT_TS_TOKEN)
    
    # 2. 验证<ts>数量与timeseries变量数一致
    ts_count = input_text.count(DEFAULT_TS_TOKEN)
    n_vars = len(timeseries)
    if ts_count != n_vars:
        raise ValueError(
            f"文本中<ts>数量({ts_count})与时间序列变量数({n_vars})不匹配\n"
            f"Input text: {input_text[:100]}..."
        )
    
    # 3. 构建messages
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text}
    ]
    
    # 4. 使用官方chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )
    
    # 5. 移除空think块
    if remove_thinking:
        text = EMPTY_THINK_PATTERN.sub('', text)
    
    # 6. Tokenize
    input_ids = tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids[0]
    
    # 7. 构建labels（mask掉user部分）
    labels = input_ids.clone()
    
    # 计算user部分长度
    user_only_messages = [{"role": "user", "content": input_text}]
    user_text = tokenizer.apply_chat_template(
        user_only_messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )
    if remove_thinking:
        user_text = EMPTY_THINK_PATTERN.sub('', user_text)
    
    user_ids = tokenizer(user_text, add_special_tokens=True).input_ids
    user_len = len(user_ids)
    
    # assistant前缀
    assistant_prefix = "<|im_start|>assistant\n"
    assistant_prefix_ids = tokenizer(assistant_prefix, add_special_tokens=False).input_ids
    prefix_len = user_len + len(assistant_prefix_ids)
    
    # Mask user部分
    labels[:prefix_len] = IGNORE_INDEX
    
    # 8. 转换时间序列为tensor并进行归一化
    # timeseries: [[var1], [var2], ...] -> [n_vars, seq_len]
    ts_tensor = torch.tensor(timeseries, dtype=torch.float32)
    
    # 验证长度
    if ts_tensor.shape[1] != context_window:
        # Padding或截断到context_window
        if ts_tensor.shape[1] < context_window:
            # Padding（右侧补0）
            pad_len = context_window - ts_tensor.shape[1]
            ts_tensor = torch.nn.functional.pad(ts_tensor, (0, pad_len), value=0.0)
        else:
            # 截断
            ts_tensor = ts_tensor[:, :context_window]
    
    # 9. 归一化时间序列并提取尺度信息
    ts_normalized, scale_stats = normalize_timeseries(ts_tensor)
    
    return dict(
        input_ids=input_ids,
        labels=labels,
        timeseries=ts_normalized,  # [n_vars, seq_len] 归一化后的时间序列
        scale_stats=scale_stats     # [n_vars, 2] 每个变量的(mean, std)
    )


class MultimodalDataset(Dataset):
    """
    多模态监督学习数据集（时间序列+文本）
    
    数据格式（JSONL）：
    {
        "input": "There are 2 time series. <ts></ts> and <ts></ts>. What patterns?",
        "timeseries": [[1.0, 2.0, ...], [3.0, 4.0, ...]],
        "output": "Both time series show upward trends..."
    }
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        model_max_length: int,
        context_window: int = 256,
        remove_thinking: bool = True
    ):
        super(MultimodalDataset, self).__init__()
        
        # 读取JSONL数据
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if line.strip():
                    try:
                        item = json.loads(line)
                        # 验证必需字段
                        if 'input' not in item or 'output' not in item or 'timeseries' not in item:
                            print(f"警告：第{line_idx+1}行缺少必需字段，跳过")
                            continue
                        self.data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"警告：第{line_idx+1}行JSON解析失败，跳过: {e}")
                        continue
        
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.context_window = context_window        #TODO：作用
        self.remove_thinking = remove_thinking
        
        print(f"✓ 加载 {len(self.data)} 条多模态样本（来自 {data_path}）")
        print(f"✓ 时序窗口长度: {context_window}")
        if remove_thinking:
            print(f"✓ 已启用<think>块移除")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = self.data[i]   #一个dict,dict['timeseries']是list[list[float]]
        
        try:
            # 预处理
            processed = preprocess_multimodal_qwen(
                input_text=sample["input"],
                output_text=sample["output"],
                timeseries=sample["timeseries"],
                tokenizer=self.tokenizer,
                context_window=self.context_window,
                remove_thinking=self.remove_thinking
            )
            
            return dict(
                input_ids=processed["input_ids"],
                labels=processed["labels"],
                timeseries=processed["timeseries"],
                scale_stats=processed["scale_stats"]
            )
        
        except Exception as e:
            print(f"\n错误：处理样本{i}时出错: {e}")
            print(f"样本内容: {sample}")
            raise


class DataCollatorForMultimodalDataset:
    """
    多模态数据整理器
    
    处理batch内的padding，特别处理timeseries（因为长度可能不同）
    """
    
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        整理batch
        
        Args:
            instances: List of dicts, 每个dict包含 {input_ids, labels, timeseries, scale_stats}
            
        Returns:
            batch: {
                input_ids: [batch, seq_len],
                labels: [batch, seq_len],
                attention_mask: [batch, seq_len],
                timeseries: List of [n_vars, seq_len],  # 注意：这里是list而非tensor
                scale_stats: List of [n_vars, 2]        # 每个变量的(mean, std)
            }
        """
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        timeseries = [instance["timeseries"] for instance in instances]
        scale_stats = [instance["scale_stats"] for instance in instances]
        
        # Padding文本部分
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )
        
        # 截断到model_max_length
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        # TODO: attention_mask的构造，作用？
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        # timeseries和scale_stats保持为list（因为可能变量数不同）
        # 模型forward时会逐个处理
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            timeseries=timeseries,  # List of [n_vars, seq_len]
            scale_stats=scale_stats  # List of [n_vars, 2]
        )
        
        return batch


def make_multimodal_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    model_max_length: int,
    context_window: int = 256
) -> Dict:
    """
    创建多模态数据模块
    
    Args:
        tokenizer: Qwen3 tokenizer
        data_args: 数据参数对象（包含data_path属性）
        model_max_length: 最大序列长度
        context_window: 时序窗口长度
        
    Returns:
        dict: {
            train_dataset: Dataset,
            eval_dataset: Optional[Dataset],
            data_collator: DataCollator
        }
    """
    train_dataset = MultimodalDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        model_max_length=model_max_length,
        context_window=context_window,
        remove_thinking=True
    )
    
    data_collator = DataCollatorForMultimodalDataset(tokenizer=tokenizer)
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )
