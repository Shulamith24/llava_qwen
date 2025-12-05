"""
合并 LoRA 权重到基座模型（对齐 LLaVA）
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_weights(
    model_path: str,
    lora_path: str,
    output_path: str,
    device: str = "cpu"
):
    """
    合并 LoRA 权重到基座模型
    
    Args:
        model_path: 基座模型路径
        lora_path: LoRA 权重路径
        output_path: 输出路径
        device: 设备（cpu/cuda）
    """
    print(f"Loading base model from {model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
    )
    
    print(f"Loading LoRA weights from {lora_path}...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_path}...")
    model.save_pretrained(output_path)
    
    # 同时保存 tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_path)
    
    print("Done! Merged model saved to:", output_path)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights to base model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to base model"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        required=True,
        help="Path to LoRA weights"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save merged model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for merging"
    )
    
    args = parser.parse_args()
    
    merge_lora_weights(
        model_path=args.model_path,
        lora_path=args.lora_path,
        output_path=args.output_path,
        device=args.device
    )


if __name__ == "__main__":
    main()
