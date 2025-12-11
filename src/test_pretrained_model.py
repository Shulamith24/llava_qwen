"""
预训练模型测试脚本
在相同数据集上进行前向验证，计算loss并可选生成对比

用法:
    python src/test_pretrained_model.py \
        --checkpoint_path outputs/pretrain_projector_multi/checkpoint-2463 \
        --data_path /path/to/train.jsonl \
        --batch_size 4
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer

# 导入多模态组件
from model.qwen3_ts import Qwen3TSConfig, Qwen3TSForCausalLM
from dataset_multimodal import (
    MultimodalDataset, 
    DataCollatorForMultimodalDataset,
    preprocess_multimodal_qwen,
    normalize_timeseries
)
from constants import DEFAULT_TS_TOKEN, IGNORE_INDEX
import constants as GLOBAL_CONSTANTS


@dataclass
class TestArguments:
    """测试参数"""
    checkpoint_path: str = field(
        metadata={"help": "Checkpoint路径，包含完整模型权重"}
    )
    data_path: str = field(
        metadata={"help": "测试数据路径（JSONL格式）"}
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "测试batch大小"}
    )
    num_samples: Optional[int] = field(
        default=None,
        metadata={"help": "测试样本数量，None表示全部"}
    )
    context_window: int = field(
        default=256,
        metadata={"help": "时序窗口长度"}
    )
    model_max_length: int = field(
        default=2048,
        metadata={"help": "最大序列长度"}
    )
    generate_samples: int = field(
        default=0,
        metadata={"help": "生成对比的样本数量，0表示不生成"}
    )
    max_new_tokens: int = field(
        default=256,
        metadata={"help": "生成时最大新token数"}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "是否使用bf16精度"}
    )


def load_model_and_tokenizer(args: TestArguments, device: str):
    """
    从checkpoint加载模型和tokenizer
    
    Args:
        args: 测试参数
        device: 设备
        
    Returns:
        model, tokenizer
    """
    checkpoint_path = args.checkpoint_path
    
    print(f"\n{'='*60}")
    print(f"从checkpoint加载模型: {checkpoint_path}")
    print(f"{'='*60}")
    
    # 检查checkpoint路径
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint路径不存在: {checkpoint_path}")
    
    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        raise ValueError(f"Checkpoint缺少config.json: {checkpoint_path}")
    
    # 1. 加载配置
    print("加载配置...")
    config = Qwen3TSConfig.from_pretrained(checkpoint_path)
    print(f"  - 时序编码器: {config.mm_ts_tower}")
    print(f"  - 投影层类型: {config.mm_projector_type}")
    print(f"  - ts_d_model: {config.ts_d_model}")
    print(f"  - context_window: {config.context_window}")
    
    # 2. 加载模型
    print("加载模型权重...")
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float32
    
    model = Qwen3TSForCausalLM.from_pretrained(
        checkpoint_path,
        config=config,
        torch_dtype=compute_dtype,
        device_map=device
    )
    model.eval()
    print(f"  ✓ 模型加载完成，dtype={compute_dtype}")
    
    # 3. 加载tokenizer
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False
    )
    
    # 检查是否已包含<ts> token
    if DEFAULT_TS_TOKEN not in tokenizer.get_vocab():
        print(f"  - 添加 {DEFAULT_TS_TOKEN} token")
        tokenizer.add_tokens([DEFAULT_TS_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
    else:
        print(f"  ✓ {DEFAULT_TS_TOKEN} token已存在于tokenizer中")
    
    # 更新全局常量
    GLOBAL_CONSTANTS.TS_TOKEN_INDEX = tokenizer.convert_tokens_to_ids(DEFAULT_TS_TOKEN)
    print(f"  - TS_TOKEN_INDEX = {GLOBAL_CONSTANTS.TS_TOKEN_INDEX}")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"  ✓ Tokenizer加载完成，词表大小: {len(tokenizer)}")
    
    return model, tokenizer


def load_dataset(args: TestArguments, tokenizer):
    """
    加载测试数据集，复用训练数据处理逻辑
    
    Args:
        args: 测试参数
        tokenizer: tokenizer
        
    Returns:
        dataset, dataloader, data_collator
    """
    print(f"\n加载数据集: {args.data_path}")
    
    # 创建数据集
    dataset = MultimodalDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        model_max_length=args.model_max_length,
        context_window=args.context_window,
        remove_thinking=True
    )
    
    # 限制样本数量
    if args.num_samples is not None and args.num_samples < len(dataset):
        print(f"  - 限制测试样本数: {args.num_samples}/{len(dataset)}")
        # 使用Subset来限制数据集大小
        from torch.utils.data import Subset
        indices = list(range(args.num_samples))
        dataset = Subset(dataset, indices)
    
    # 创建data collator
    data_collator = DataCollatorForMultimodalDataset(tokenizer=tokenizer)
    
    # 创建dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0  # 单进程避免问题
    )
    
    print(f"  ✓ 数据集大小: {len(dataset)}")
    print(f"  ✓ Batch数量: {len(dataloader)}")
    
    return dataset, dataloader, data_collator


def run_forward_test(model, dataloader, device):
    """
    运行前向测试，计算平均loss
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        
    Returns:
        avg_loss: 平均loss
        all_losses: 每个batch的loss列表
    """
    print(f"\n{'='*60}")
    print("开始前向测试")
    print(f"{'='*60}")
    
    model.eval()
    total_loss = 0.0
    total_batches = 0
    all_losses = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Testing")
        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # timeseries和scale_stats是list，需要移动每个tensor并转换dtype
            model_dtype = next(model.parameters()).dtype
            timeseries = [ts.to(device=device, dtype=model_dtype) for ts in batch["timeseries"]]
            scale_stats = [ss.to(device=device, dtype=model_dtype) for ss in batch["scale_stats"]]
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                timeseries=timeseries,
                scale_stats=scale_stats,
                return_dict=True
            )
            
            loss = outputs.loss.item()
            all_losses.append(loss)
            total_loss += loss
            total_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "avg_loss": f"{total_loss/total_batches:.4f}"
            })
    
    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    
    print(f"\n{'='*60}")
    print(f"测试结果")
    print(f"{'='*60}")
    print(f"  总Batch数: {total_batches}")
    print(f"  平均Loss: {avg_loss:.6f}")
    print(f"  最小Loss: {min(all_losses):.6f}")
    print(f"  最大Loss: {max(all_losses):.6f}")
    
    return avg_loss, all_losses


def run_generation_comparison(
    model, 
    tokenizer, 
    data_path: str,
    context_window: int,
    num_samples: int, 
    max_new_tokens: int,
    device: str
):
    """
    对若干样本进行生成并与reference对比
    
    Args:
        model: 模型
        tokenizer: tokenizer
        data_path: 数据路径
        context_window: 时序窗口
        num_samples: 生成样本数
        max_new_tokens: 最大新token数
        device: 设备
    """
    if num_samples <= 0:
        return
    
    print(f"\n{'='*60}")
    print(f"生成对比测试 (前{num_samples}个样本)")
    print(f"{'='*60}")
    
    # 读取原始数据
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            if line.strip():
                samples.append(json.loads(line))
    
    model.eval()
    
    for idx, sample in enumerate(samples):
        print(f"\n{'─'*40}")
        print(f"样本 {idx + 1}/{len(samples)}")
        print(f"{'─'*40}")
        
        input_text = sample["input"]
        output_text = sample["output"]
        timeseries_data = sample["timeseries"]
        
        # 处理时间序列
        input_text_processed = input_text.replace("<ts></ts>", DEFAULT_TS_TOKEN)
        ts_tensor = torch.tensor(timeseries_data, dtype=torch.float32)
        
        # Padding/截断
        if ts_tensor.shape[1] < context_window:
            pad_len = context_window - ts_tensor.shape[1]
            ts_tensor = torch.nn.functional.pad(ts_tensor, (0, pad_len), value=0.0)
        else:
            ts_tensor = ts_tensor[:, :context_window]
        
        # 归一化
        ts_normalized, scale_stats = normalize_timeseries(ts_tensor)
        
        # 构建只包含user消息的输入（用于生成）
        messages = [{"role": "user", "content": input_text_processed}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        # Tokenize
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        # 生成
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                timeseries=[ts_normalized.to(device)],
                scale_stats=[scale_stats.to(device)],
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 使用greedy decoding以便对比
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码生成结果
        generated_text = tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        print(f"\n【输入】:\n{input_text[:200]}...")
        print(f"\n【参考输出】:\n{output_text[:300]}...")
        print(f"\n【生成输出】:\n{generated_text[:300]}...")


def main():
    # 解析参数
    parser = argparse.ArgumentParser(description="预训练模型测试脚本")
    parser.add_argument("--checkpoint_path", type=str, default="/mnt/data/qyh/codes/llava_qwen/outputs/pretrain_projector_multi/checkpoint-2463",
                       help="Checkpoint路径")
    parser.add_argument("--data_path", type=str, default="/mnt/data/qyh/dataset/ChatTS/align_256/train.jsonl",
                       help="测试数据路径")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch大小")
    parser.add_argument("--num_samples", type=int, default=4,
                       help="测试样本数量")
    parser.add_argument("--context_window", type=int, default=256,
                       help="时序窗口长度")
    parser.add_argument("--model_max_length", type=int, default=2048,
                       help="最大序列长度")
    parser.add_argument("--generate_samples", type=int, default=0,
                       help="生成对比样本数")
    parser.add_argument("--max_new_tokens", type=int, default=2000,
                       help="生成最大新token数")
    parser.add_argument("--bf16", action="store_true", default=True,
                       help="使用bf16精度")
    parser.add_argument("--no_bf16", action="store_false", dest="bf16",
                       help="不使用bf16精度")
    
    args = parser.parse_args()
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 转换为dataclass
    test_args = TestArguments(
        checkpoint_path=args.checkpoint_path,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        context_window=args.context_window,
        model_max_length=args.model_max_length,
        generate_samples=args.generate_samples,
        max_new_tokens=args.max_new_tokens,
        bf16=args.bf16
    )
    
    # 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer(test_args, device)
    
    # 加载数据集
    dataset, dataloader, data_collator = load_dataset(test_args, tokenizer)
    
    # 运行前向测试
    avg_loss, all_losses = run_forward_test(model, dataloader, device)
    
    # 可选：生成对比
    if test_args.generate_samples > 0:
        run_generation_comparison(
            model=model,
            tokenizer=tokenizer,
            data_path=test_args.data_path,
            context_window=test_args.context_window,
            num_samples=test_args.generate_samples,
            max_new_tokens=test_args.max_new_tokens,
            device=device
        )
    
    print(f"\n{'='*60}")
    print("测试完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
