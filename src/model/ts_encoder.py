"""
时序编码器包装类
包装PatchTST编码器，提供统一的接口和权重加载功能
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import sys

# 添加PatchTST路径
patchtst_path = os.path.join(os.path.dirname(__file__), '..', '..', 'PatchTST_supervised')
if patchtst_path not in sys.path:
    sys.path.insert(0, patchtst_path)

from layers.PatchTST_backbone import PatchTST_backbone


class PatchTSTEncoderWrapper(nn.Module):
    """
    PatchTST编码器包装类
    
    提供统一的接口，支持：
    - 预训练权重加载
    - 灵活的冻结/解冻控制
    - 兼容LLaVA架构的属性
    
    Args:
        context_window: 输入序列长度
        patch_len: patch长度
        stride: patch步长
        d_model: 模型维度
        n_layers: Transformer层数
        n_heads: 注意力头数
        d_ff: 前馈网络维度
        dropout: dropout率
        **kwargs: 其他PatchTST参数
    """
    
    def __init__(
        self,
        context_window: int = 256,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 16,
        d_ff: int = 256,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        # 存储配置参数
        self.context_window = context_window
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        
        # 计算patch数量
        self.patch_num = int((context_window - patch_len) / stride + 1)
        
        # 关键修复：PatchTST可以处理任意的nvars，因为它会将batch和nvars合并处理
        # 所以我们只需要创建一个encoder，使用c_in=1作为占位
        # 在forward时，PatchTST内部会自动处理不同的nvars
        # 
        # 重要：禁用RevIN，因为：
        # 1. 用户要求不进行时间序列分解
        # 2. RevIN的affine参数维度依赖于c_in，会导致不同nvars的样本无法共享参数
        # 3. 归一化应该在数据预处理阶段完成
        self.encoder = PatchTST_backbone(
            c_in=1,  # 占位值，实际处理时nvars会被合并到batch维度
            context_window=context_window,
            target_window=0,  # 我们只需要特征，不需要预测
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            # 关键配置
            pretrain_head=False,  # 不使用预测头
            head_type='flatten',
            individual=False,
            revin=False,  # 禁用RevIN，避免维度不匹配
            affine=False,
            subtract_last=False,
            **{k: v for k, v in kwargs.items()}
        )
        
        # 兼容LLaVA架构的属性
        self.is_loaded = False
        self.hidden_size = d_model  # 每个patch的特征维度
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（单样本）
        
        Args:
            x: [n_vars, seq_len] 单个样本的时间序列
            
        Returns:
            features: [n_vars, patch_num, d_model] 时序特征
        """
        # 确保输入是2D: [n_vars, seq_len]
        if x.dim() == 3:
            assert x.size(0) == 1, "Wrapper只支持单样本编码"
            x = x.squeeze(0)
        
        n_vars, seq_len = x.shape
        assert seq_len == self.context_window, \
            f"输入序列长度{seq_len}与context_window{self.context_window}不匹配"
        
        # 使用encoder的第一个backbone进行特征提取
        # PatchTST_backbone有多个尺度的backbone，我们只使用第一个
        backbone = self.encoder.backbones[0]
        p_len = self.encoder.patch_len_list[0]
        s_len = self.encoder.stride_list[0]
        
        # 添加batch维度: [1, n_vars, seq_len]
        x = x.unsqueeze(0)
        
        # Patching: 遵循PatchTST源码的实现
        # unfold: [bs, n_vars, patch_num, patch_len]
        x_unfolded = x.unfold(dimension=-1, size=p_len, step=s_len)
        # permute to: [bs, n_vars, patch_len, patch_num]
        x_patched = x_unfolded.permute(0, 1, 3, 2)
        
        # 使用TSTiEncoder的forward方法
        # 输入: [bs, n_vars, patch_len, patch_num]
        # 输出: [bs, n_vars, d_model, patch_num]
        z = backbone(x_patched)
        
        # 转换为 [bs, n_vars, patch_num, d_model]
        z = z.permute(0, 1, 3, 2)
        
        # 移除batch维度: [n_vars, patch_num, d_model]
        z = z.squeeze(0)
        
        return z
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载预训练权重
        
        Args:
            checkpoint_path: checkpoint文件路径
        """
        if not os.path.exists(checkpoint_path):
            print(f"警告：checkpoint文件不存在: {checkpoint_path}")
            return
        
        print(f"从 {checkpoint_path} 加载PatchTST权重...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 提取模型权重
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 加载权重（只加载匹配的层）
        if self.encoder is not None:
            # 过滤不匹配的键
            model_dict = self.encoder.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.encoder.load_state_dict(model_dict, strict=False)
            print(f"成功加载 {len(pretrained_dict)}/{len(state_dict)} 个参数")
        
        self.is_loaded = True
    
    @property
    def device(self):
        """获取设备"""
        if self.encoder is not None:
            return next(self.encoder.parameters()).device
        return torch.device('cpu')
    
    @property
    def dtype(self):
        """获取数据类型"""
        if self.encoder is not None:
            return next(self.encoder.parameters()).dtype
        return torch.float32
    
    @property
    def dummy_feature(self):
        """返回dummy特征（用于非多模态样本）"""
        return torch.zeros(1, self.d_model, device=self.device, dtype=self.dtype)
    
    def freeze(self):
        """冻结所有参数"""
        self.requires_grad_(False)
        print("已冻结PatchTST编码器")
    
    def unfreeze(self):
        """解冻所有参数"""
        self.requires_grad_(True)
        print("已解冻PatchTST编码器")


def build_ts_encoder(
    checkpoint_path: Optional[str] = None,
    freeze: bool = True,
    **config
) -> PatchTSTEncoderWrapper:
    """
    构建时序编码器
    
    Args:
        checkpoint_path: 预训练权重路径
        freeze: 是否冻结权重
        **config: 编码器配置参数
        
    Returns:
        encoder: 时序编码器实例
    """
    # 创建编码器
    encoder = PatchTSTEncoderWrapper(**config)
    
    # 加载权重
    if checkpoint_path is not None:
        encoder.load_checkpoint(checkpoint_path)
    
    # 冻结参数
    if freeze:
        encoder.freeze()
    
    return encoder
