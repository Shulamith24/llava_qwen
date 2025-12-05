"""
测试多模态数据加载
验证数据集、DataCollator和模型forward是否正常工作
"""

import sys
import os
import torch
from transformers import AutoTokenizer

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset_multimodal import MultimodalDataset, DataCollatorForMultimodalDataset
from model.qwen3_ts import Qwen3TSConfig, Qwen3TSForCausalLM
from constants import DEFAULT_TS_TOKEN, TS_TOKEN_INDEX
import constants as GLOBAL_CONSTANTS


def test_dataset_loading(data_path, tokenizer, context_window=256):
    """测试数据集加载"""
    print("\n" + "="*60)
    print("测试1：数据集加载")
    print("="*60)
    
    try:
        dataset = MultimodalDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            model_max_length=2048,
            context_window=context_window
        )
        
        print(f"✓ 成功加载数据集，共 {len(dataset)} 条样本")
        
        # 测试第一个样本
        sample = dataset[0]
        print(f"\n样本 0:")
        print(f"  - input_ids shape: {sample['input_ids'].shape}")
        print(f"  - labels shape: {sample['labels'].shape}")
        print(f"  - time_series shape: {sample['time_series'].shape}")
        print(f"  - time_series: [n_vars={sample['time_series'].shape[0]}, seq_len={sample['time_series'].shape[1]}]")
        
        # 解码文本
        text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
        print(f"\n  文本内容（前200字符）:\n  {text[:200]}...")
        
        # 统计<ts> token数量
        ts_count = (sample['input_ids'] == TS_TOKEN_INDEX).sum().item()
        print(f"\n  <ts> token数量: {ts_count}")
        print(f"  时序变量数: {sample['time_series'].shape[0]}")
        
        if ts_count == sample['time_series'].shape[0]:
            print(f"  ✓ <ts>数量与变量数一致")
        else:
            print(f"  ✗ 警告：<ts>数量与变量数不一致！")
        
        return True
    
    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_collator(data_path, tokenizer, context_window=256):
    """测试DataCollator"""
    print("\n" + "="*60)
    print("测试2：DataCollator（批处理）")
    print("="*60)
    
    try:
        dataset = MultimodalDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            model_max_length=2048,
            context_window=context_window
        )
        
        collator = DataCollatorForMultimodalDataset(tokenizer=tokenizer)
        
        # 创建一个batch（2个样本）
        batch_size = min(2, len(dataset))
        instances = [dataset[i] for i in range(batch_size)]
        
        batch = collator(instances)
        
        print(f"✓ 成功创建batch，batch_size={batch_size}")
        print(f"\nBatch内容:")
        print(f"  - input_ids: {batch['input_ids'].shape}")
        print(f"  - labels: {batch['labels'].shape}")
        print(f"  - attention_mask: {batch['attention_mask'].shape}")
        print(f"  - time_series: List[Tensor], 长度={len(batch['time_series'])}")
        
        for i, ts in enumerate(batch['time_series']):
            print(f"    - 样本{i}: {ts.shape}")
        
        return True
    
    except Exception as e:
        print(f"✗ DataCollator测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward(data_path, model_path, context_window=256):
    """测试模型forward"""
    print("\n" + "="*60)
    print("测试3：模型forward")
    print("="*60)
    
    try:
        # 加载tokenizer
        print(f"加载tokenizer: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=2048,
            padding_side="right",
            use_fast=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 添加<ts> token
        print(f"添加特殊token: {DEFAULT_TS_TOKEN}")
        num_new_tokens = tokenizer.add_tokens([DEFAULT_TS_TOKEN], special_tokens=True)
        GLOBAL_CONSTANTS.TS_TOKEN_INDEX = tokenizer.convert_tokens_to_ids(DEFAULT_TS_TOKEN)
        print(f"  TS_TOKEN_INDEX = {GLOBAL_CONSTANTS.TS_TOKEN_INDEX}")
        
        # 创建配置
        print(f"\n创建模型配置...")
        config = Qwen3TSConfig.from_pretrained(
            model_path,
            mm_ts_tower="patchtst",
            patchtst_checkpoint=None,  # 测试不加载权重
            freeze_patchtst=True,
            context_window=context_window,
            patch_len=16,
            stride=8,
            ts_d_model=128,
            ts_n_layers=3,
            ts_n_heads=16,
            ts_d_ff=256,
            mm_projector_type="mlp2x_gelu",
        )
        
        # 加载模型
        print(f"加载模型: {model_path}")
        model = Qwen3TSForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16,
        )
        
        # Resize embedding
        if num_new_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))
        
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        print(f"✓ 模型加载成功，设备: {device}")
        
        # 准备数据
        dataset = MultimodalDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            model_max_length=2048,
            context_window=context_window
        )
        
        collator = DataCollatorForMultimodalDataset(tokenizer=tokenizer)
        batch = collator([dataset[0]])
        
        # 移动到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        time_series = [ts.to(device) for ts in batch['time_series']]
        
        print(f"\n执行forward...")
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                time_series=time_series
            )
        
        print(f"✓ Forward成功！")
        print(f"  - loss: {outputs.loss.item():.4f}")
        print(f"  - logits shape: {outputs.logits.shape}")
        
        return True
    
    except Exception as e:
        print(f"✗ 模型forward失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="测试多模态数据加载和模型")
    parser.add_argument("--data_path", type=str, required=True, help="JSONL数据文件路径")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-8B", help="Qwen3模型路径")
    parser.add_argument("--context_window", type=int, default=256, help="时序窗口长度")
    parser.add_argument("--test", type=str, default="all", choices=["all", "dataset", "collator", "model"],
                       help="测试类型")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("多模态数据加载测试")
    print("="*60)
    print(f"数据路径: {args.data_path}")
    print(f"模型路径: {args.model_path}")
    print(f"时序窗口: {args.context_window}")
    
    # 加载tokenizer（用于测试1和2）
    if args.test in ["all", "dataset", "collator"]:
        print(f"\n加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            model_max_length=2048,
            padding_side="right",
            use_fast=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 添加<ts> token
        num_new_tokens = tokenizer.add_tokens([DEFAULT_TS_TOKEN], special_tokens=True)
        GLOBAL_CONSTANTS.TS_TOKEN_INDEX = tokenizer.convert_tokens_to_ids(DEFAULT_TS_TOKEN)
        print(f"✓ Tokenizer加载完成")
        print(f"  添加 {num_new_tokens} 个新token")
        print(f"  TS_TOKEN_INDEX = {GLOBAL_CONSTANTS.TS_TOKEN_INDEX}")
    
    # 运行测试
    results = {}
    
    if args.test in ["all", "dataset"]:
        results["dataset"] = test_dataset_loading(args.data_path, tokenizer, args.context_window)
    
    if args.test in ["all", "collator"]:
        results["collator"] = test_data_collator(args.data_path, tokenizer, args.context_window)
    
    if args.test in ["all", "model"]:
        results["model"] = test_model_forward(args.data_path, args.model_path, args.context_window)
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print("\n❌ 部分测试失败")
        return 1


if __name__ == "__main__":
    exit(main())
