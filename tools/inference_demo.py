"""
推理示例脚本
演示如何使用训练好的多模态模型进行推理
"""

import sys
import os
import torch
import json
from transformers import AutoTokenizer

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.qwen3_ts import Qwen3TSConfig, Qwen3TSForCausalLM
from constants import DEFAULT_TS_TOKEN
import constants as GLOBAL_CONSTANTS


def load_model(model_path, checkpoint_path=None):
    """
    加载多模态模型
    
    Args:
        model_path: Qwen3模型路径
        checkpoint_path: 微调后的checkpoint路径（可选）
    """
    print(f"加载模型: {model_path}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 添加<ts> token
    num_new_tokens = tokenizer.add_tokens([DEFAULT_TS_TOKEN], special_tokens=True)
    GLOBAL_CONSTANTS.TS_TOKEN_INDEX = tokenizer.convert_tokens_to_ids(DEFAULT_TS_TOKEN)
    
    # 创建配置
    config = Qwen3TSConfig.from_pretrained(
        checkpoint_path if checkpoint_path else model_path,
        mm_ts_tower="patchtst",
        patchtst_checkpoint="PatchTST_supervised/checkpoints/checkpoint.pth",
        freeze_patchtst=True,
        context_window=256,
        patch_len=16,
        stride=8,
        ts_d_model=128,
        mm_projector_type="mlp2x_gelu",
    )
    
    # 加载模型
    model = Qwen3TSForCausalLM.from_pretrained(
        checkpoint_path if checkpoint_path else model_path,
        config=config,
        torch_dtype=torch.float16,
    )
    
    # Resize embedding
    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"✓ 模型加载完成，设备: {device}")
    
    return model, tokenizer, device


def inference(
    model,
    tokenizer,
    device,
    input_text: str,
    time_series: list,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9
):
    """
    执行推理
    
    Args:
        model: 多模态模型
        tokenizer: tokenizer
        device: 设备
        input_text: 输入文本（包含<ts>占位符）
        time_series: 时间序列数据 [[var1], [var2], ...]
        max_new_tokens: 最大生成token数
        temperature: 温度参数
        top_p: top-p采样参数
    
    Returns:
        generated_text: 生成的文本
    """
    # 验证<ts>数量
    ts_count = input_text.count(DEFAULT_TS_TOKEN)
    n_vars = len(time_series)
    
    if ts_count != n_vars:
        raise ValueError(
            f"输入文本中<ts>数量({ts_count})与时间序列变量数({n_vars})不匹配"
        )
    
    # 构建messages
    messages = [{"role": "user", "content": input_text}]
    
    # 使用chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    # 准备时序数据
    ts_tensor = torch.tensor(time_series, dtype=torch.float32).to(device)
    
    # 生成
    print("\n生成中...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            time_series=[ts_tensor],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取assistant回复（去掉prompt）
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    if generated_text.startswith(prompt_text):
        generated_text = generated_text[len(prompt_text):].strip()
    
    return generated_text


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="多模态模型推理示例")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-8B", help="Qwen3模型路径")
    parser.add_argument("--checkpoint", type=str, default=None, help="微调后的checkpoint路径")
    parser.add_argument("--input_file", type=str, default=None, help="输入JSONL文件（可选）")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度参数")
    parser.add_argument("--top_p", type=float, default=0.9, help="top-p采样")
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer, device = load_model(args.model_path, args.checkpoint)
    
    if args.input_file:
        # 从文件读取
        print(f"\n从文件读取输入: {args.input_file}")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.loads(f.readline())
            input_text = data["input"]
            time_series = data["timeseries"]
    else:
        # 使用示例数据
        print("\n使用示例数据")
        input_text = "There are 2 time series. <ts> and <ts>. Can you describe the trend of these time series?"
        
        # 生成示例时序数据（2个变量，每个256个点）
        import numpy as np
        t = np.linspace(0, 10, 256)
        var1 = np.sin(t) + np.random.normal(0, 0.1, 256)
        var2 = np.cos(t) + np.random.normal(0, 0.1, 256)
        time_series = [var1.tolist(), var2.tolist()]
    
    print("\n" + "="*60)
    print("推理示例")
    print("="*60)
    print(f"输入文本: {input_text}")
    print(f"时序变量数: {len(time_series)}")
    print(f"时序长度: {len(time_series[0])}")
    
    # 执行推理
    generated_text = inference(
        model=model,
        tokenizer=tokenizer,
        device=device,
        input_text=input_text,
        time_series=time_series,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    print("\n" + "="*60)
    print("生成结果")
    print("="*60)
    print(generated_text)
    print("\n")


if __name__ == "__main__":
    main()
