"""
环境检查脚本
"""

import sys


def check_environment():
    """检查环境配置"""
    print("="*60)
    print("Environment Check")
    print("="*60 + "\n")
    
    # Python 版本
    print(f"✓ Python version: {sys.version.split()[0]}")
    
    # PyTorch
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
                mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"      Memory: {mem:.1f} GB")
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    # Transformers
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not installed")
        return False
    
    # PEFT
    try:
        import peft
        print(f"✓ PEFT version: {peft.__version__}")
    except ImportError:
        print("✗ PEFT not installed")
        return False
    
    # bitsandbytes
    try:
        import bitsandbytes as bnb
        print(f"✓ bitsandbytes version: {bnb.__version__}")
    except ImportError:
        print("⚠ bitsandbytes not installed (QLoRA will not work)")
    
    # DeepSpeed
    try:
        import deepspeed
        print(f"✓ DeepSpeed version: {deepspeed.__version__}")
    except ImportError:
        print("⚠ DeepSpeed not installed (multi-GPU training will not work)")
    
    # 其他依赖
    optional_packages = [
        "accelerate",
        "tensorboard",
        "wandb",
        "datasets",
    ]
    
    for package in optional_packages:
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            print(f"✓ {package} version: {version}")
        except ImportError:
            print(f"⚠ {package} not installed (optional)")
    
    print("\n" + "="*60)
    print("Environment check completed!")
    print("="*60 + "\n")
    
    return True


def test_data_loading():
    """测试数据加载"""
    import json
    import os
    
    print("="*60)
    print("Data Loading Test")
    print("="*60 + "\n")
    
    # 检查示例数据
    sample_file = "../data/finetune_sample.jsonl"
    if os.path.exists(sample_file):
        print(f"✓ Found sample data: {sample_file}")
        
        # 读取并验证格式
        with open(sample_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"  - Total samples: {len(lines)}")
            
            # 检查第一条数据
            if lines:
                sample = json.loads(lines[0])
                print(f"  - First sample keys: {list(sample.keys())}")
                if "conversations" in sample:
                    print(f"  - Conversations length: {len(sample['conversations'])}")
                    print(f"  - First turn: {sample['conversations'][0]}")
    else:
        print(f"✗ Sample data not found: {sample_file}")
    
    print("\n" + "="*60)
    print("Data loading test completed!")
    print("="*60 + "\n")


def print_next_steps():
    """打印下一步操作"""
    print("="*60)
    print("Next Steps")
    print("="*60 + "\n")
    
    print("1. Prepare your data:")
    print("   - Put training data in ./data/finetune.jsonl")
    print("   - Format: JSONL with 'conversations' field")
    print("")
    
    print("2. Configure training script:")
    print("   - Edit ./scripts/finetune_lora.sh")
    print("   - Set MODEL_PATH, DATA_PATH, OUTPUT_DIR")
    print("")
    
    print("3. Start training:")
    print("   cd scripts")
    print("   bash finetune_lora.sh      # LoRA training")
    print("   # or")
    print("   bash finetune_qlora.sh     # QLoRA training (less memory)")
    print("")
    
    print("4. Test inference:")
    print("   python scripts/test_inference.py \\")
    print("       --model-path Qwen/Qwen2.5-7B-Instruct \\")
    print("       --lora-path ./outputs/qwen3-lora-finetune")
    print("")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    success = check_environment()
    
    if success:
        test_data_loading()
        print_next_steps()
    else:
        print("\n⚠ Please install missing packages:")
        print("pip install -r requirements.txt")
