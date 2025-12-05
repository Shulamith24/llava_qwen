"""
测试推理脚本（支持 LoRA 模型）
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(model_path: str, lora_path: str = None, device: str = "cuda"):
    """加载模型和 tokenizer"""
    print(f"Loading model from {model_path}...")
    
    if lora_path:
        # 加载 LoRA 模型
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
        )
        print(f"Loading LoRA weights from {lora_path}...")
        model = PeftModel.from_pretrained(base_model, lora_path)
    else:
        # 加载完整模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """生成回复"""
    # 格式化 prompt（Qwen 格式）
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 提取 assistant 的回复
    if "<|im_start|>assistant\n" in response:
        response = response.split("<|im_start|>assistant\n")[-1]
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
    
    return response.strip()


def interactive_chat(model, tokenizer):
    """交互式对话"""
    print("\n" + "="*50)
    print("Interactive Chat (type 'exit' to quit)")
    print("="*50 + "\n")
    
    while True:
        prompt = input("User: ")
        if prompt.lower() in ["exit", "quit", "q"]:
            break
        
        response = generate_response(model, tokenizer, prompt)
        print(f"Assistant: {response}\n")


def main():
    parser = argparse.ArgumentParser(description="Test inference with Qwen3 model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to base model"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA weights (optional)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt for testing (if not provided, enters interactive mode)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max generation length"
    )
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer = load_model(
        args.model_path,
        args.lora_path,
        args.device
    )
    
    if args.prompt:
        # 单次推理
        response = generate_response(
            model, 
            tokenizer, 
            args.prompt,
            max_length=args.max_length
        )
        print(f"User: {args.prompt}")
        print(f"Assistant: {response}")
    else:
        # 交互式对话
        interactive_chat(model, tokenizer)


if __name__ == "__main__":
    main()
