"""
测试时序编码器的修复
验证encoder能够处理不同变量数的输入
"""

import sys
import torch
from pathlib import Path

# 添加src路径
current_file = Path(__file__).resolve()
parent_dir = current_file.parents[1]
sys.path.insert(0, str(parent_dir))

from src.model.ts_encoder import PatchTSTEncoderWrapper


def test_encoder_with_different_nvars():
    """测试encoder处理不同变量数的能力"""
    print("\n" + "="*60)
    print("测试：Encoder处理不同变量数")
    print("="*60)
    
    # 创建encoder
    context_window = 256
    patch_len = 16
    stride = 8
    d_model = 128
    
    encoder = PatchTSTEncoderWrapper(
        context_window=context_window,
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        n_layers=3,
        n_heads=16,
        d_ff=256,
        dropout=0.1
    )
    
    print(f"✓ Encoder创建成功")
    print(f"  - context_window: {context_window}")
    print(f"  - patch_len: {patch_len}")
    print(f"  - stride: {stride}")
    print(f"  - d_model: {d_model}")
    print(f"  - expected patch_num: {encoder.patch_num}")
    
    # 测试不同的变量数
    test_cases = [
        (2, "2个变量"),
        (5, "5个变量"),
        (10, "10个变量"),
    ]
    
    for n_vars, desc in test_cases:
        print(f"\n测试: {desc}")
        
        # 创建随机输入
        x = torch.randn(n_vars, context_window)
        print(f"  输入shape: {x.shape}")
        
        # Forward
        with torch.no_grad():
            features = encoder(x)
        
        print(f"  输出shape: {features.shape}")
        
        # 验证输出形状
        expected_shape = (n_vars, encoder.patch_num, d_model)
        assert features.shape == expected_shape, \
            f"输出shape不匹配！期望{expected_shape}, 实际{features.shape}"
        
        print(f"  ✓ 输出shape正确: {features.shape}")
    
    print("\n" + "="*60)
    print("✅ 所有测试通过！")
    print("="*60)


def test_encoder_shared_parameters():
    """验证不同nvars的样本共享相同的encoder参数"""
    print("\n" + "="*60)
    print("测试：验证参数共享")
    print("="*60)
    
    encoder = PatchTSTEncoderWrapper(
        context_window=256,
        patch_len=16,
        stride=8,
        d_model=128,
    )
    
    # 记录初始参数
    initial_params = {name: param.clone() for name, param in encoder.named_parameters()}
    
    # 处理第一个样本（3个变量）
    x1 = torch.randn(3, 256)
    with torch.no_grad():
        _ = encoder(x1)
    
    # 处理第二个样本（7个变量）
    x2 = torch.randn(7, 256)
    with torch.no_grad():
        _ = encoder(x2)
    
    # 验证参数没有变化（因为是推理模式）
    for name, param in encoder.named_parameters():
        assert torch.all(param == initial_params[name]), \
            f"参数{name}发生了变化！"
    
    print("✓ 验证通过：不同nvars的样本共享相同的encoder")
    print("✓ 参数总数:", sum(p.numel() for p in encoder.parameters()))
    

if __name__ == "__main__":
    try:
        test_encoder_with_different_nvars()
        test_encoder_shared_parameters()
        print("\n🎉 所有测试通过！Encoder修复成功。")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
