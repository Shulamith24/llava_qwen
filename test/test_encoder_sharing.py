"""
测试PatchTST编码器在不同n_vars样本间共享
"""

import sys
import torch
from pathlib import Path

# 添加src路径
current_file = Path(__file__).resolve()
parent_dir = current_file.parents[1]
sys.path.insert(0, str(parent_dir))

from src.model.ts_encoder import PatchTSTEncoderWrapper


def test_encoder_sharing():
    """测试编码器在不同变量数的样本间是否共享"""
    print("=" * 60)
    print("测试PatchTST编码器共享")
    print("=" * 60)
    
    # 创建编码器
    encoder = PatchTSTEncoderWrapper(
        context_window=256,
        patch_len=16,
        stride=8,
        d_model=128,
        n_layers=3,
        n_heads=16,
        d_ff=256,
    )
    
    print(f"\n编码器已创建")
    print(f"  - context_window: {encoder.context_window}")
    print(f"  - patch_len: {encoder.patch_len}")
    print(f"  - stride: {encoder.stride}")
    print(f"  - patch_num: {encoder.patch_num}")
    print(f"  - d_model: {encoder.d_model}")
    
    # 记录初始backbone id
    initial_backbone_id = id(encoder.backbone)
    print(f"\n初始backbone ID: {initial_backbone_id}")
    
    # 测试不同n_vars的样本
    test_cases = [
        (2, 256),   # 2个变量
        (5, 256),   # 5个变量
        (10, 256),  # 10个变量
        (2, 256),   # 再次2个变量
    ]
    
    print("\n" + "=" * 60)
    print("测试不同n_vars样本的编码")
    print("=" * 60)
    
    for i, (n_vars, seq_len) in enumerate(test_cases):
        print(f"\n测试样本 {i+1}: n_vars={n_vars}, seq_len={seq_len}")
        
        # 创建随机输入
        x = torch.randn(n_vars, seq_len)
        
        # 前向传播
        features = encoder(x)
        
        # 检查形状
        expected_shape = (n_vars, encoder.patch_num, encoder.d_model)
        actual_shape = tuple(features.shape)
        
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {actual_shape}")
        print(f"  期望形状: {expected_shape}")
        
        if actual_shape == expected_shape:
            print(f"  ✓ 形状正确")
        else:
            print(f"  ✗ 形状错误！")
            return False
        
        # 检查backbone是否共享
        current_backbone_id = id(encoder.backbone)
        if current_backbone_id == initial_backbone_id:
            print(f"  ✓ Backbone共享（ID: {current_backbone_id}）")
        else:
            print(f"  ✗ Backbone被重新创建！（ID: {current_backbone_id}）")
            return False
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
    print("\n✓ 编码器在所有样本间共享")
    print("✓ 输出形状正确")
    
    return True


if __name__ == "__main__":
    success = test_encoder_sharing()
    exit(0 if success else 1)
