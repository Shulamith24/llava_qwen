import re
import json
import torch
from typing import Dict, List
from torch.utils.data import Dataset
import transformers


# 常量
IGNORE_INDEX = -100

# 用于删除空的 <think>...</think> 块的正则
EMPTY_THINK_PATTERN = re.compile(r'<think>\s*</think>\s*', flags=re.S)


def preprocess_qwen(
    input_text: str, 
    output_text: str, 
    tokenizer: transformers.PreTrainedTokenizer,
    remove_thinking: bool = True
) -> Dict:
    """
    预处理单个样本（input/output 格式）
    
    Args:
        input_text: 用户问题
        output_text: 模型回答
        tokenizer: Qwen3 tokenizer
        remove_thinking: 是否移除空的 <think> 块（默认 True）
        
    Returns:
        dict: {"input_ids": tensor, "labels": tensor}
    """
    # 构建 Qwen3 标准 messages 格式
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text}
    ]
    
    # 使用官方 chat template 生成完整序列
    # 注意：即使 enable_thinking=False，对于历史 assistant 消息，
    # 模板仍会插入 <think>\n\n</think>\n\n
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )
    
    # 移除模板自动添加的空 <think> 块
    # 这样训练时模型不会学到"每次回答前先输出 <think></think>"
    if remove_thinking:
        text = EMPTY_THINK_PATTERN.sub('', text)
    
    # Tokenize 完整序列, 返回一个类似字典的batchEncoding对象
    input_ids = tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids[0]
    
    # 构建 labels：只对 assistant 回复部分计算 loss
    labels = input_ids.clone()
    
    # 计算 user 部分的长度（用于 mask）
    # 关键：这次模板调用的参数要和上面完全一致
    user_only_messages = [{"role": "user", "content": input_text}]
    user_text = tokenizer.apply_chat_template(
        user_only_messages,
        tokenize=False,
        add_generation_prompt=False,  # 改为 False，与上面对齐
        enable_thinking=False
    )
    
    # 同样移除空 think 块（虽然 user 消息不会有，但保持一致）
    if remove_thinking:
        user_text = EMPTY_THINK_PATTERN.sub('', user_text)
    
    # 计算需要 mask 的前缀长度
    # 注意：这里不能用 add_special_tokens=False，因为上面 tokenize text 时用的是默认（True）
    user_ids = tokenizer(user_text, add_special_tokens=True).input_ids
    user_len = len(user_ids)
    
    # 在 user_text 和完整 text 之间，还有 "<|im_start|>assistant\n" 的部分
    # 我们需要把这部分也加到 mask 里
    assistant_prefix = "<|im_start|>assistant\n"
    assistant_prefix_ids = tokenizer(assistant_prefix, add_special_tokens=False).input_ids
    prefix_len = user_len + len(assistant_prefix_ids)
    
    # 将 user 部分 + assistant 前缀 mask 掉
    labels[:prefix_len] = IGNORE_INDEX
    
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """监督学习数据集（input/output 格式）"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        model_max_length: int,
        remove_thinking: bool = True
    ):
        super(SupervisedDataset, self).__init__()
        
        # 读取 jsonl 数据
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # 只保留 input 和 output，忽略 timeseries
                    self.data.append({
                        "input": item["input"],
                        "output": item["output"]
                    })
        
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.remove_thinking = remove_thinking
        
        print(f"✓ 加载 {len(self.data)} 条样本（来自 {data_path}）")
        if remove_thinking:
            print(f"✓ 已启用 <think> 块移除")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = self.data[i]
        
        # 预处理,返回input_ids和labels的字典，labels中"assistant"之前的token都被mask掉
        processed = preprocess_qwen(
            input_text=sample["input"],
            output_text=sample["output"],
            tokenizer=self.tokenizer,
            remove_thinking=self.remove_thinking
        )
        
        return dict(
            input_ids=processed["input_ids"],
            labels=processed["labels"]
        )


class DataCollatorForSupervisedDataset:
    """数据整理器（批处理 + padding）"""
    
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        
        # Padding
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
        
        # 截断到 model_max_length
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        
        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    model_max_length: int
) -> Dict:
    """创建数据模块"""
    
    train_dataset = SupervisedDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        model_max_length=model_max_length,
        remove_thinking=True  # 默认移除空 think 块
    )
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )
